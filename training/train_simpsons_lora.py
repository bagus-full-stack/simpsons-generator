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

Prérequis (si --install-deps n'est pas utilisé) :
    pip install accelerate transformers diffusers peft datasets kagglehub pillow requests torch
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
TRAIN_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "examples/text_to_image/train_text_to_image_lora.py"
)
TRAIN_SCRIPT_PATH = CURRENT_DIR / "train_text_to_image_lora.py"

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

    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--dataset-slug", default="kostastokis/simpsons-faces", help="Dataset Kaggle (kagglehub)")
    p.add_argument("--train-data-dir", default=str(CURRENT_DIR / "train_data"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "simpsons_lora_results"),
                    help="Par défaut : simpsons_lora_results/ à la racine du dépôt, lu par l'API de service.")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=2)
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

    p.add_argument("--install-deps", action="store_true", help="Installe les dépendances pip avant de lancer.")
    p.add_argument("--skip-dataset-prep", action="store_true", help="Réutilise --train-data-dir tel quel.")
    p.add_argument("--force-redownload", action="store_true", help="Force le re-téléchargement du dataset Kaggle.")

    return p.parse_args()


def install_dependencies() -> None:
    log.info("Installation des dépendances...")
    packages = [
        "accelerate", "transformers", "diffusers", "peft", "datasets",
        "kagglehub", "pillow", "requests", "bitsandbytes", "xformers",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    log.info("Dépendances installées.")


def ensure_training_script() -> None:
    if TRAIN_SCRIPT_PATH.exists():
        log.info("Script d'entraînement déjà présent : %s", TRAIN_SCRIPT_PATH)
        return
    log.info("Téléchargement du script officiel diffusers (train_text_to_image_lora.py)...")
    response = requests.get(TRAIN_SCRIPT_URL, timeout=30)
    response.raise_for_status()
    TRAIN_SCRIPT_PATH.write_bytes(response.content)
    log.info("Script d'entraînement téléchargé : %s", TRAIN_SCRIPT_PATH)


def prepare_dataset(dataset_slug: str, train_data_dir: Path, force_redownload: bool) -> None:
    import kagglehub

    log.info("Téléchargement du dataset Kaggle '%s'...", dataset_slug)
    try:
        cache_path = Path(kagglehub.dataset_download(dataset_slug, force_download=force_redownload))
    except TypeError:
        # Anciennes versions de kagglehub sans le paramètre force_download.
        cache_path = Path(kagglehub.dataset_download(dataset_slug))

    source = cache_path / "cropped" if (cache_path / "cropped").exists() else cache_path

    if train_data_dir.exists():
        shutil.rmtree(train_data_dir)
    train_data_dir.mkdir(parents=True)

    valid_extensions = (".png", ".jpg", ".jpeg")
    images = sorted(f for f in os.listdir(source) if f.lower().endswith(valid_extensions))
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans le dataset téléchargé ({source})")
    log.info("%d images trouvées.", len(images))

    metadata_path = train_data_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for i, img in enumerate(images):
            shutil.copy(source / img, train_data_dir / img)
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            f.write(json.dumps({"file_name": img, "text": prompt}) + "\n")

    log.info("Dataset prêt dans %s", train_data_dir)


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
        prepare_dataset(args.dataset_slug, train_data_dir, args.force_redownload)
    else:
        if not train_data_dir.exists():
            raise SystemExit(f"--skip-dataset-prep utilisé mais {train_data_dir} n'existe pas.")
        log.info("Préparation du dataset ignorée (--skip-dataset-prep), réutilisation de %s", train_data_dir)

    exit_code = run_training(args)

    if exit_code == 0:
        log.info("Entraînement terminé. Poids LoRA dans : %s", output_dir)
    else:
        log.error("L'entraînement a échoué (code %d).", exit_code)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
