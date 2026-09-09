#!/usr/bin/env python3
"""
Entraînement LoRA Stable Diffusion pour le style Simpsons.

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

Dataset : Norod78/simpsons-blip-captions (Hugging Face, 755 images captionnées
par BLIP, scènes complètes — pas que des visages). Les images sources font
512x512 : une passe "hires fix" (img2img SDXL à faible denoise, voir
upscale_dataset()) les porte à --resolution avant l'entraînement, pour
apporter un vrai détail plausible plutôt que le simple redimensionnement
bilinéaire que ferait de toute façon le script d'entraînement sinon.

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
# Script SDXL dédié (pas train_text_to_image_lora.py, qui est pour SD1.5) :
# gère le double text encoder et les embeddings "pooled" propres à SDXL.
TRAIN_SCRIPT_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/"
    "examples/text_to_image/train_text_to_image_lora_sdxl.py"
)
TRAIN_SCRIPT_PATH = CURRENT_DIR / "train_text_to_image_lora_sdxl.py"

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

    p.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument(
        "--pretrained-vae-model",
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Fix communautaire du VAE SDXL (NaN en fp16). Vide pour utiliser le VAE du modèle de base.",
    )
    p.add_argument(
        "--dataset-id",
        default="Norod78/simpsons-blip-captions",
        help="Dataset Hugging Face (chargé via `datasets`), avec colonnes 'image' + 'text'.",
    )
    p.add_argument("--train-data-dir", default=str(CURRENT_DIR / "train_data"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "simpsons_lora_results"),
                    help="Par défaut : simpsons_lora_results/ à la racine du dépôt, lu par l'API de service.")

    p.add_argument("--resolution", type=int, default=1024)
    # SDXL est nettement plus lourd que SD1.5 : batch_size=1 + gradient checkpointing
    # + adam 8-bit (voir run_training) pour tenir sur un GPU Kaggle (T4/P100, 16 Go).
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=5e-5)
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
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Modèle utilisé pour la passe d'upscale (par défaut : le même que --base-model, déjà téléchargé).",
    )
    p.add_argument(
        "--skip-upscale",
        action="store_true",
        help="Ne pas upscaler le dataset (ex: déjà fait lors d'un run précédent avec --skip-dataset-prep).",
    )

    p.add_argument("--install-deps", action="store_true", help="Installe les dépendances pip avant de lancer.")
    p.add_argument("--skip-dataset-prep", action="store_true", help="Réutilise --train-data-dir tel quel.")
    p.add_argument("--force-redownload", action="store_true", help="Force le re-téléchargement du dataset.")

    return p.parse_args()


def install_dependencies() -> None:
    log.info("Installation des dépendances...")
    packages = [
        "accelerate", "transformers==4.47.1", f"diffusers=={DIFFUSERS_VERSION}", "peft", "datasets",
        "pillow", "requests", "bitsandbytes", "xformers",
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


def ensure_training_script() -> None:
    if TRAIN_SCRIPT_PATH.exists():
        # Un script mis en cache par un run précédent (autre version de
        # DIFFUSERS_VERSION) planterait sur check_min_version() malgré le
        # pin ci-dessus : on le détecte et on retélécharge plutôt que de
        # skipper silencieusement.
        cached = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8", errors="ignore")
        if f'check_min_version("{DIFFUSERS_VERSION}")' in cached:
            log.info("Script d'entraînement déjà présent (version à jour) : %s", TRAIN_SCRIPT_PATH)
            return
        log.info("Script d'entraînement en cache obsolète (autre version de diffusers), retéléchargement...")

    log.info(
        "Téléchargement du script officiel diffusers (train_text_to_image_lora_sdxl.py, v%s)...",
        DIFFUSERS_VERSION,
    )
    response = requests.get(TRAIN_SCRIPT_URL, timeout=30)
    response.raise_for_status()
    TRAIN_SCRIPT_PATH.write_bytes(response.content)
    log.info("Script d'entraînement téléchargé : %s", TRAIN_SCRIPT_PATH)


def prepare_dataset(dataset_id: str, train_data_dir: Path, force_redownload: bool) -> None:
    """Télécharge un dataset Hugging Face (colonnes 'image' + légende 'text' ou
    'caption') et l'écrit au format imagefolder + metadata.jsonl attendu par le
    script d'entraînement. Si le dataset n'a pas de légende par image, retombe
    sur DEFAULT_PROMPTS en rotation (comportement de l'ancien dataset Kaggle)."""
    from datasets import load_dataset

    log.info("Téléchargement du dataset Hugging Face '%s'...", dataset_id)
    download_mode = "force_redownload" if force_redownload else None
    dataset = load_dataset(dataset_id, split="train", download_mode=download_mode)

    if train_data_dir.exists():
        shutil.rmtree(train_data_dir)
    train_data_dir.mkdir(parents=True)

    metadata_path = train_data_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            file_name = f"{i:05d}.png"
            row["image"].convert("RGB").save(train_data_dir / file_name)
            caption = row.get("text") or row.get("caption") or DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            f.write(json.dumps({"file_name": file_name, "text": caption}) + "\n")

    log.info("%d images écrites dans %s", len(dataset), train_data_dir)


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


def run_training(args: argparse.Namespace) -> int:
    mixed_precision = detect_mixed_precision(args.mixed_precision)
    log.info("Lancement de l'entraînement (mixed_precision=%s)...", mixed_precision)

    cmd = [
        "accelerate", "launch", str(TRAIN_SCRIPT_PATH),
        f"--pretrained_model_name_or_path={args.base_model}",
        f"--train_data_dir={args.train_data_dir}",
        "--dataset_name=imagefolder",
        "--caption_column=text",
        f"--resolution={args.resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        # Nécessaires pour faire tenir un entraînement SDXL sur un GPU 16 Go
        # (Kaggle T4/P100) — inutiles mais inoffensifs sur un GPU plus large.
        "--gradient_checkpointing",
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
        f"--seed={args.seed}",
    ]
    if args.pretrained_vae_model:
        cmd.append(f"--pretrained_vae_model_name_or_path={args.pretrained_vae_model}")
    if args.resume_from_checkpoint:
        cmd.append(f"--resume_from_checkpoint={args.resume_from_checkpoint}")

    log.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    args = parse_args()
    train_data_dir = Path(args.train_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        install_dependencies()

    ensure_training_script()

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

    exit_code = run_training(args)

    if exit_code == 0:
        log.info("Entraînement terminé. Poids LoRA dans : %s", output_dir)
    else:
        log.error("L'entraînement a échoué (code %d).", exit_code)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
