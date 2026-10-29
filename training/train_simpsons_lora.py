#!/usr/bin/env python3
"""
Entraînement LoRA Stable Diffusion (SD1.5 par défaut, SDXL avec --model-family sdxl) pour le style Simpsons.

Remplace simpsonGenerator_FIXED.ipynb : mêmes étapes (installation optionnelle des
dépendances, préparation du dataset Kaggle, lancement de l'entraînement via le
script officiel diffusers), en script autonome, configurable en ligne de commande
et sans chemin codé en dur (le notebook utilisait /tf/workspace/diffusion/...,
spécifique à l'environnement cloud où il avait été écrit).

Usage :
    python train_simpsons_lora.py --install-deps      # première exécution sur une machine neuve
    python train_simpsons_lora.py                      # runs suivants
    python train_simpsons_lora.py --max-train-steps 1500 --resolution 768
    python train_simpsons_lora.py --skip-dataset-prep  # dataset déjà préparé dans --train-data-dir
    python train_simpsons_lora.py --resume-from-checkpoint latest

    python train_simpsons_lora.py --model-family sdxl   # SDXL : 1024 px, batch 1, nettement plus lent

Kaggle (2x T4, 15 Go chacun) : les 2 GPU sont détectés et utilisés automatiquement
(accelerate --multi_gpu). T4 = pas de bf16, donc fp16 (défaut). Voir le README.
    python train_simpsons_lora.py --install-deps --output-dir /kaggle/working/simpsons_lora_results

Dataset : kostastokis/simpsons-faces (Kaggle, visages Simpsons déjà recadrés,
sans légende par image -> DEFAULT_PROMPTS en rotation). --skip-upscale est
actif par défaut pour ce dataset : la passe "hires fix" (img2img SDXL, voir
upscale_dataset()) repeint les images vers l'esthétique générique de SDXL et
a dilué le style Simpsons lors d'un run précédent (sur Norod78/simpsons-blip-
captions, un dataset de scènes complètes avec légendes BLIP génériques,
utilisé un temps à la place de celui-ci).

Prérequis (si --install-deps n'est pas utilisé) :
    pip install accelerate transformers diffusers peft datasets pillow requests torch
    (+ bitsandbytes / xformers optionnels pour réduire l'empreinte mémoire GPU)

Le LoRA entraîné est écrit dans --output-dir (par défaut ./simpsons_lora_results),
le même dossier que lit simpsonGeneratorAPI.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_simpsons_lora")

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

# Épinglée volontairement (même version que diffusers==0.31.0 dans
# requirements.txt) : le script d'entraînement ci-dessous est téléchargé
# depuis le tag Git correspondant, pas depuis `main`, qui pointe vers une
# diffusers de développement (ex: 0.41.0.dev0) très en avance sur toute
# release PyPI et fait échouer check_min_version() au lancement — après
# que le pré-traitement du dataset (upscale hires-fix, ~2h sur Kaggle) a
# déjà tourné pour rien.
DIFFUSERS_VERSION = "0.31.0"
# Un script officiel par famille : le script SDXL gère le double text encoder et
# les embeddings "pooled" propres à SDXL, pas le script SD1.5.
TRAIN_SCRIPT_NAMES = {
    "sd15": "train_text_to_image_lora.py",
    "sdxl": "train_text_to_image_lora_sdxl.py",
}
# Valeurs par défaut selon la famille : (modèle de base, résolution, batch par GPU, VAE).
FAMILY_DEFAULTS = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", 512, 4, ""),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", 1024, 1, "madebyollin/sdxl-vae-fp16-fix"),
}


def train_script(family: str) -> tuple[str, Path]:
    name = TRAIN_SCRIPT_NAMES[family]
    url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/examples/text_to_image/{name}"
    return url, CURRENT_DIR / name

# Prompts utilisés en rotation pour légender les images du dataset : les varier
# (plutôt qu'une légende unique) aide le LoRA à généraliser le style plutôt que
# mémoriser une seule phrase.
DEFAULT_PROMPTS = [
    "a simpson character, yellow skin, cartoon style",
    "portrait of a simpson character with yellow skin",
    "simpson animated character, bright colors",
    "face of a simpsons tv show character",
    "cartoon simpson character portrait",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning LoRA Stable Diffusion — style Simpsons")

    p.add_argument(
        "--model-family",
        choices=["sd15", "sdxl"],
        default="sd15",
        help="sd15 (défaut, démo locale sur GPU 6 Go) ou sdxl. Détermine script, base, résolution et batch par défaut.",
    )
    p.add_argument("--base-model", default=None, help="Défaut : selon --model-family.")
    p.add_argument(
        "--pretrained-vae-model",
        default=None,
        help="SDXL : fix communautaire du VAE (NaN en fp16) par défaut. Vide pour utiliser le VAE du modèle de base.",
    )
    p.add_argument(
        "--dataset-id",
        default="kostastokis/simpsons-faces",
        help="Dataset Kaggle (via kagglehub) ou Hugging Face (via `datasets`, colonnes 'image' + 'text').",
    )
    p.add_argument("--train-data-dir", default=str(CURRENT_DIR / "train_data"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "simpsons_lora_results"),
                    help="Par défaut : simpsons_lora_results/ à la racine du dépôt, lu par l'API de service.")

    p.add_argument("--resolution", type=int, default=None, help="Défaut : 512 (sd15) ou 1024 (sdxl).")
    # SDXL est nettement plus lourd que SD1.5 : batch_size=1 + gradient checkpointing
    # + adam 8-bit (voir run_training) pour tenir sur un GPU Kaggle (T4/P100, 16 Go).
    p.add_argument("--train-batch-size", type=int, default=None, help="Par GPU. Défaut : 4 (sd15) ou 1 (sdxl).")
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Défaut : calculé pour un batch effectif de 4 (nb de GPU x batch x accumulation).",
    )
    p.add_argument("--max-train-steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    # Rang du LoRA : 4 (défaut diffusers) donnait un style Simpsons très faible (3 Mo) ;
    # 32 capture bien mieux le style pour ~25 Mo.
    p.add_argument("--rank", type=int, default=32)
    # Pondération Min-SNR : converge plus vite et plus stable (recommandé par diffusers).
    p.add_argument("--snr-gamma", type=float, default=5.0)
    p.add_argument("--num-processes", type=int, default=None, help="Nombre de GPU (défaut : tous ceux détectés).")
    p.add_argument("--lr-scheduler", default="cosine_with_restarts")
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--checkpointing-steps", type=int, default=500)
    p.add_argument("--validation-prompt", default="a simpson character with yellow skin")
    p.add_argument("--validation-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Par défaut : fp16 si un GPU CUDA est détecté, sinon no.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Chemin d'un checkpoint (ou 'latest') pour reprendre un entraînement interrompu.",
    )

    p.add_argument(
        "--upscale-model",
        default=None,
        help="Modèle utilisé pour la passe d'upscale SDXL (par défaut : le même que --base-model).",
    )
    p.add_argument(
        "--skip-upscale",
        action="store_true",
        default=True,
        help="Ne pas upscaler le dataset. Actif par défaut : la passe SDXL img2img repeint les images vers "
        "l'esthétique SDXL et dilue le style Simpsons (voir docstring d'upscale_dataset). "
        "Utiliser --no-skip-upscale pour la réactiver si le dataset source est en trop basse résolution.",
    )
    p.add_argument("--no-skip-upscale", dest="skip_upscale", action="store_false")

    p.add_argument("--install-deps", action="store_true", help="Installe les dépendances pip avant de lancer.")
    p.add_argument("--skip-dataset-prep", action="store_true", help="Réutilise --train-data-dir tel quel.")
    p.add_argument("--force-redownload", action="store_true", help="Force le re-téléchargement du dataset.")

    args = p.parse_args()
    base, resolution, batch, vae = FAMILY_DEFAULTS[args.model_family]
    args.base_model = args.base_model or base
    args.resolution = args.resolution or resolution
    args.train_batch_size = args.train_batch_size or batch
    if args.pretrained_vae_model is None:
        args.pretrained_vae_model = vae
    args.upscale_model = args.upscale_model or args.base_model
    return args


def install_dependencies() -> None:
    log.info("Installation des dépendances...")
    packages = [
        "accelerate", "transformers==4.47.1", f"diffusers=={DIFFUSERS_VERSION}",
        # Épinglé comme dans requirements.txt : un peft plus récent tente
        # d'importer transformers.integrations.tensor_parallel (ajouté dans une
        # version de transformers postérieure à 4.47.1) dès qu'on recharge le
        # LoRA dans le pipeline (set_peft_model_state_dict), ce qui fait planter
        # la validation finale avec ModuleNotFoundError une fois l'entraînement
        # terminé.
        "peft==0.14.0",
        "datasets", "kagglehub", "pillow", "requests", "bitsandbytes", "xformers",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    # `peft` récent refuse d'ajouter l'adaptateur LoRA si `torchao` est présent
    # en version trop ancienne (ex: 0.10.0 préinstallé sur les images Kaggle),
    # même quand on ne s'en sert pas du tout ici (pas de quantization) : voir
    # peft.tuners.lora.torchao.dispatch_torchao -> is_torchao_available(), qui
    # lève une ImportError bloquante au lieu de simplement désactiver ce
    # dispatch. On le désinstalle pour repasser dans le cas "non disponible".
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    log.info("Dépendances installées.")


def ensure_training_script(family: str) -> Path:
    url, path = train_script(family)
    if path.exists():
        # Un script mis en cache par un run précédent (autre version de
        # DIFFUSERS_VERSION) planterait sur check_min_version() malgré le
        # pin ci-dessus : on le détecte et on retélécharge plutôt que de
        # skipper silencieusement.
        cached = path.read_text(encoding="utf-8", errors="ignore")
        if f'check_min_version("{DIFFUSERS_VERSION}")' in cached:
            log.info("Script d'entraînement déjà présent (version à jour) : %s", path)
            return path
        log.info("Script d'entraînement en cache obsolète (autre version de diffusers), retéléchargement...")

    log.info(
        "Téléchargement du script officiel diffusers (train_text_to_image_lora_sdxl.py, v%s)...",
        DIFFUSERS_VERSION,
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    log.info("Script d'entraînement téléchargé : %s", path)
    return path


def prepare_dataset_kaggle(dataset_id: str, train_data_dir: Path, force_redownload: bool) -> None:
    """Dataset Kaggle (ex: kostastokis/simpsons-faces) : visages Simpsons déjà
    recadrés, sans légende par image -> DEFAULT_PROMPTS en rotation, comme dans
    l'ancien notebook."""
    import kagglehub

    log.info("Téléchargement du dataset Kaggle '%s'...", dataset_id)
    source = Path(kagglehub.dataset_download(dataset_id, force_download=force_redownload))
    if (source / "cropped").exists():
        source = source / "cropped"
    images = sorted(f for f in source.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))

    metadata_path = train_data_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for i, img in enumerate(images):
            shutil.copy(img, train_data_dir / img.name)
            caption = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            f.write(json.dumps({"file_name": img.name, "text": caption}) + "\n")

    log.info("%d images écrites dans %s", len(images), train_data_dir)


def prepare_dataset_hf(dataset_id: str, train_data_dir: Path, force_redownload: bool) -> None:
    """Dataset Hugging Face (colonnes 'image' + légende 'text' ou 'caption')."""
    from datasets import load_dataset

    log.info("Téléchargement du dataset Hugging Face '%s'...", dataset_id)
    download_mode = "force_redownload" if force_redownload else None
    dataset = load_dataset(dataset_id, split="train", download_mode=download_mode)

    metadata_path = train_data_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            file_name = f"{i:05d}.png"
            row["image"].convert("RGB").save(train_data_dir / file_name)
            caption = row.get("text") or row.get("caption") or DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            f.write(json.dumps({"file_name": file_name, "text": caption}) + "\n")

    log.info("%d images écrites dans %s", len(dataset), train_data_dir)


def prepare_dataset(dataset_id: str, train_data_dir: Path, force_redownload: bool) -> None:
    if train_data_dir.exists():
        shutil.rmtree(train_data_dir)
    train_data_dir.mkdir(parents=True)

    try:
        prepare_dataset_kaggle(dataset_id, train_data_dir, force_redownload)
    except Exception as e:
        log.info("Pas un dataset Kaggle valide (%s), tentative via Hugging Face...", e)
        prepare_dataset_hf(dataset_id, train_data_dir, force_redownload)


def upscale_dataset(train_data_dir: Path, target_resolution: int, upscale_model: str) -> None:
    """Porte les images du dataset à target_resolution via une passe img2img
    SDXL à faible denoise ('hires fix') plutôt qu'un simple resize : le
    modèle hallucine du détail plausible, alors qu'un redimensionnement
    bilinéaire/Lanczos n'ajoute rien que le script d'entraînement ne ferait
    déjà tout seul en chargeant les images à --resolution.
    """
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from PIL import Image

    image_files = sorted(
        f for f in train_data_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not image_files:
        log.warning("Aucune image à upscaler dans %s.", train_data_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("Pas de GPU détecté : l'upscale de %d images va être très lent sur CPU.", len(image_files))
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info("Chargement de %s pour l'upscale (%s)...", upscale_model, device)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(upscale_model, torch_dtype=dtype).to(device)

    log.info("Upscale de %d images vers %dx%d...", len(image_files), target_resolution, target_resolution)
    for i, path in enumerate(image_files, start=1):
        low_res = Image.open(path).convert("RGB").resize(
            (target_resolution, target_resolution), Image.Resampling.LANCZOS
        )
        upscaled = pipe(
            prompt="a simpson character, yellow skin, cartoon style, high quality, sharp details",
            image=low_res,
            strength=0.3,
            num_inference_steps=20,
            guidance_scale=4.0,
        ).images[0]
        upscaled.save(path)
        if i % 100 == 0 or i == len(image_files):
            log.info("Upscale : %d/%d", i, len(image_files))

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
    log.info("Upscale terminé.")


def detect_mixed_precision(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch
        return "fp16" if torch.cuda.is_available() else "no"
    except ImportError:
        return "no"


def detect_num_processes(explicit: int | None) -> int:
    if explicit:
        return explicit
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def run_training(args: argparse.Namespace, script_path: Path) -> int:
    mixed_precision = detect_mixed_precision(args.mixed_precision)
    num_processes = detect_num_processes(args.num_processes)
    accumulation = args.gradient_accumulation_steps or max(1, 4 // (num_processes * args.train_batch_size))
    log.info(
        "Lancement de l'entraînement (GPU=%d, mixed_precision=%s, batch effectif=%d)...",
        num_processes, mixed_precision, num_processes * args.train_batch_size * accumulation,
    )

    launcher = ["accelerate", "launch"]
    if num_processes > 1:
        launcher += ["--multi_gpu", f"--num_processes={num_processes}", f"--mixed_precision={mixed_precision}"]
    cmd = [
        *launcher, str(script_path),
        f"--pretrained_model_name_or_path={args.base_model}",
        f"--train_data_dir={args.train_data_dir}",
        "--dataset_name=imagefolder",
        "--caption_column=text",
        f"--resolution={args.resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={accumulation}",
        f"--rank={args.rank}",
        f"--snr_gamma={args.snr_gamma}",
        "--use_8bit_adam",
        f"--max_train_steps={args.max_train_steps}",
        f"--learning_rate={args.learning_rate}",
        f"--lr_scheduler={args.lr_scheduler}",
        f"--lr_warmup_steps={args.lr_warmup_steps}",
        f"--output_dir={args.output_dir}",
        f"--mixed_precision={mixed_precision}",
        f"--checkpointing_steps={args.checkpointing_steps}",
        f"--validation_prompt={args.validation_prompt}",
        f"--validation_epochs={args.validation_epochs}",
        # 4 par défaut ; 1 suffit pour contrôler et limite la pression VRAM.
        "--num_validation_images=1",
        f"--seed={args.seed}",
    ]
    if args.model_family == "sdxl":
        # Nécessaire pour faire tenir SDXL sur un GPU 16 Go (Kaggle T4/P100).
        cmd.append("--gradient_checkpointing")
    if args.pretrained_vae_model and args.model_family == "sdxl":
        cmd.append(f"--pretrained_vae_model_name_or_path={args.pretrained_vae_model}")
    if args.resume_from_checkpoint:
        cmd.append(f"--resume_from_checkpoint={args.resume_from_checkpoint}")

    log.info("Commande : %s", " ".join(cmd))
    # Réduit la fragmentation VRAM (OOM à la validation finale sur GPU 16 Go).
    env = {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    rc = subprocess.run(cmd, env=env).returncode
    # La validation finale peut OOM après que le LoRA a été sauvé : l'entraînement est alors bon.
    if rc != 0 and (Path(args.output_dir) / "pytorch_lora_weights.safetensors").exists():
        log.warning("Échec après sauvegarde du LoRA (validation finale ?) : entraînement considéré OK.")
        return 0
    return rc


def main() -> None:
    args = parse_args()
    train_data_dir = Path(args.train_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        install_dependencies()

    script_path = ensure_training_script(args.model_family)

    if not args.skip_dataset_prep:
        prepare_dataset(args.dataset_id, train_data_dir, args.force_redownload)
    else:
        if not train_data_dir.exists():
            raise SystemExit(f"--skip-dataset-prep utilisé mais {train_data_dir} n'existe pas.")
        log.info("Préparation du dataset ignorée (--skip-dataset-prep), réutilisation de %s", train_data_dir)

    if not args.skip_upscale:
        upscale_dataset(train_data_dir, args.resolution, args.upscale_model)
    else:
        log.info("Upscale du dataset ignoré (--skip-upscale).")

    exit_code = run_training(args, script_path)

    if exit_code == 0:
        log.info("Entraînement terminé. Poids LoRA dans : %s", output_dir)
    else:
        log.error("L'entraînement a échoué (code %d).", exit_code)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
