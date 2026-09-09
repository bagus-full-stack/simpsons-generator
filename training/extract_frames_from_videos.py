#!/usr/bin/env python3
"""
Extraction de frames HD depuis tes propres fichiers vidéo (épisodes possédés
légalement — Disney+, Blu-ray...), pour obtenir un dataset 1024x1024 natif
sans artefact d'upscale IA. C'est l'alternative "Option B" à upscale_dataset()
dans train_simpsons_lora.py, qui compense la basse résolution (512x512) du
dataset Norod78 par une passe hires-fix (img2img SDXL) : ici, pas d'upscale
nécessaire puisque la source est déjà en 1080p+.

Pipeline par vidéo :
  1. ffmpeg échantillonne les frames à --fps, puis élimine au passage les
     frames quasi-identiques consécutives (filtre mpdecimate) : un plan fixe
     de plusieurs secondes ne produit ainsi qu'une poignée de frames au lieu
     d'une par intervalle d'échantillonnage.
  2. Un second passage en Python (perceptual hash via `imagehash`) élimine
     les doublons non consécutifs qu'mpdecimate ne voit pas : même plan
     répété plus tard dans l'épisode, générique, écran-titre, etc. — y
     compris d'un épisode à l'autre.
  3. Chaque frame conservée est redimensionnée (plus petit côté =
     --resolution) puis recadrée au centre en carré --resolution x
     --resolution — le même traitement que --center_crop appliquerait de
     toute façon au chargement dans train_text_to_image_lora_sdxl.py, mais
     fait ici une fois pour de vrai plutôt qu'à la volée à chaque epoch.
  4. Écriture au format imagefolder + metadata.jsonl attendu par
     train_simpsons_lora.py, avec les mêmes légendes en rotation
     (DEFAULT_PROMPTS) que l'ancien dataset Kaggle sans légende par image.

Le dossier de sortie se branche directement sur le script d'entraînement :
    python train_simpsons_lora.py --skip-dataset-prep --skip-upscale \
        --train-data-dir training/train_data_hd --resolution 1024

Prérequis :
  - ffmpeg installé et sur le PATH.
    Windows : `winget install Gyan.FFmpeg` (ou `choco install ffmpeg`).
    Vérifier avec : ffmpeg -version
  - pip install pillow imagehash   (ou --install-deps)

Rappel droit d'auteur (déjà dans le README du projet) : les frames extraites
restent des images d'épisodes protégés par le droit d'auteur. Usage fan art /
non commercial uniquement — même statut que le dataset Kaggle/HF utilisé
jusqu'ici, ce n'est pas un risque nouveau, juste à garder en tête.

Usage :
    python extract_frames_from_videos.py --input-dir "D:\\Simpsons\\Episodes"
    python extract_frames_from_videos.py --input-dir ... --fps 0.5 --resolution 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("extract_frames_from_videos")

CURRENT_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".webm"}

from train_simpsons_lora import DEFAULT_PROMPTS  # noqa: E402  (mêmes légendes que le dataset existant)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extraction de frames HD depuis des épisodes vidéo, pour dataset LoRA")

    p.add_argument("--input-dir", required=True, help="Dossier contenant les fichiers vidéo des épisodes.")
    p.add_argument("--output-dir", default=str(CURRENT_DIR / "train_data_hd"),
                    help="Dossier de sortie (format imagefolder + metadata.jsonl).")

    p.add_argument("--fps", type=float, default=1 / 3,
                    help="Fréquence d'échantillonnage (frames/seconde). Défaut : 1 frame toutes les 3s.")
    p.add_argument("--resolution", type=int, default=1024,
                    help="Résolution finale (carré, resolution x resolution).")

    p.add_argument("--hash-size", type=int, default=8,
                    help="Taille du perceptual hash (imagehash.phash) ; 8 = hash 64 bits, défaut d'imagehash.")
    p.add_argument("--hash-threshold", type=int, default=6,
                    help="Distance de Hamming max (sur 64 bits par défaut) pour considérer deux frames comme doublons.")

    p.add_argument("--jpeg", action="store_true",
                    help="Extraire en JPEG (plus léger) au lieu de PNG (défaut, sans perte).")
    p.add_argument("--ffmpeg-path", default="ffmpeg", help="Chemin vers l'exécutable ffmpeg.")

    p.add_argument("--install-deps", action="store_true", help="Installe pillow/imagehash avant de lancer.")
    p.add_argument("--force", action="store_true", help="Écrase --output-dir s'il existe déjà.")

    return p.parse_args()


def install_dependencies() -> None:
    log.info("Installation des dépendances (pillow, imagehash)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow", "imagehash"], check=True)
    log.info("Dépendances installées.")


def check_ffmpeg(ffmpeg_path: str) -> None:
    if shutil.which(ffmpeg_path) is None:
        raise SystemExit(
            f"ffmpeg introuvable ('{ffmpeg_path}' pas dans le PATH). "
            "Installe-le (Windows : winget install Gyan.FFmpeg) puis réessaie."
        )


def find_videos(input_dir: Path) -> list[Path]:
    videos = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"Aucune vidéo trouvée dans {input_dir} (extensions supportées : {sorted(VIDEO_EXTENSIONS)})")
    return videos


def extract_candidate_frames(video: Path, tmp_dir: Path, fps: float, jpeg: bool, ffmpeg_path: str) -> list[Path]:
    """Échantillonne + décime une vidéo avec ffmpeg. mpdecimate compare les
    frames déjà échantillonnées à --fps entre elles et laisse tomber celles
    quasi-identiques à la précédente (plan fixe), avant même que Python ne
    voie quoi que ce soit — bien plus rapide qu'un dédoublonnage tout-Python."""
    ext = "jpg" if jpeg else "png"
    pattern = str(tmp_dir / f"frame_%06d.{ext}")
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps},mpdecimate=hi=64*8:lo=64*3:frac=0.33",
        "-vsync", "vfr",
        *(["-q:v", "2"] if jpeg else []),
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(tmp_dir.glob(f"frame_*.{ext}"))


def resize_and_center_crop(img, resolution: int):
    from PIL import Image

    w, h = img.size
    scale = resolution / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    return img.crop((left, top, left + resolution, top + resolution))


class GlobalDeduper:
    """Dédoublonnage par perceptual hash à travers TOUTES les vidéos (attrape
    les génériques/écrans-titre qui se répètent d'un épisode à l'autre), sans
    comparer chaque nouvelle frame à tout l'historique : les hashes sont
    regroupés par préfixe binaire, donc deux frames visuellement proches
    tombent presque toujours dans le même seau et seul ce seau est comparé."""

    def __init__(self, hash_size: int, threshold: int):
        self.hash_size = hash_size
        self.threshold = threshold
        self._buckets: dict[int, list] = {}

    def _bucket_key(self, hash_value) -> int:
        # Clampé à 0 : pour un --hash-size assez petit (<5, hash sur moins de
        # 24 bits), la formule d'origine devenait négative et `>>` lève un
        # ValueError. On dégrade alors juste vers "toutes les frames dans le
        # même compartiment" plutôt que de planter.
        shift = max(self.hash_size * self.hash_size // 2 - 12, 0)
        return int(str(hash_value), 16) >> shift

    def is_duplicate_then_add(self, hash_value) -> bool:
        key = self._bucket_key(hash_value)
        bucket = self._buckets.setdefault(key, [])
        for existing in bucket:
            if hash_value - existing <= self.threshold:
                return True
        bucket.append(hash_value)
        return False


def process_video(
    video: Path,
    output_dir: Path,
    deduper: GlobalDeduper,
    args: argparse.Namespace,
    next_index: int,
    metadata_file,
) -> int:
    import imagehash
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        tmp_dir = Path(tmp)
        candidates = extract_candidate_frames(video, tmp_dir, args.fps, args.jpeg, args.ffmpeg_path)
        log.info("%s : %d frames candidates après mpdecimate", video.name, len(candidates))

        kept = 0
        for frame_path in candidates:
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                phash = imagehash.phash(img, hash_size=args.hash_size)
                if deduper.is_duplicate_then_add(phash):
                    continue

                final = resize_and_center_crop(img, args.resolution)
                file_name = f"{next_index:06d}.png"
                final.save(output_dir / file_name)

                caption = DEFAULT_PROMPTS[next_index % len(DEFAULT_PROMPTS)]
                metadata_file.write(json.dumps({"file_name": file_name, "text": caption}) + "\n")

                next_index += 1
                kept += 1

        log.info("%s : %d frames retenues après dédoublonnage perceptuel", video.name, kept)
    return next_index


def main() -> None:
    args = parse_args()

    if args.install_deps:
        install_dependencies()
    check_ffmpeg(args.ffmpeg_path)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"--input-dir introuvable : {input_dir}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"{output_dir} existe déjà. Utilise --force pour l'écraser.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    videos = find_videos(input_dir)
    log.info("%d vidéo(s) trouvée(s) dans %s", len(videos), input_dir)

    deduper = GlobalDeduper(args.hash_size, args.hash_threshold)
    metadata_path = output_dir / "metadata.jsonl"
    next_index = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for video in videos:
            log.info("Traitement de %s...", video.name)
            next_index = process_video(video, output_dir, deduper, args, next_index, metadata_file)

    log.info("Terminé : %d images écrites dans %s", next_index, output_dir)
    log.info(
        "Pour entraîner sur ce dataset : python train_simpsons_lora.py "
        "--skip-dataset-prep --skip-upscale --train-data-dir %s --resolution %d",
        output_dir, args.resolution,
    )


if __name__ == "__main__":
    main()
