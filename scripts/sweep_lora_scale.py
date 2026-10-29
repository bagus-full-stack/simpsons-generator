"""Compare LORA_SCALE : même prompt et même seed pour chaque valeur, une grille en sortie.

Usage : python -m scripts.sweep_lora_scale "Elon Musk" 1.0 0.8 0.6 [--out lora_sweep]
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from app.config import get_settings
from app.generation import generate_image
from app.pipelines import load_pipelines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("scales", nargs="+", type=float)
    p.add_argument("--out", default="lora_sweep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    load_pipelines()
    settings = get_settings()

    images = []
    for scale in args.scales:
        settings.lora_scale = scale
        torch.manual_seed(args.seed)
        image = generate_image("text", args.prompt, steps=30, turbo=False)
        image.save(out / f"scale_{scale}.png")
        images.append((scale, image))

    w, h = images[0][1].size
    grid = Image.new("RGB", (w * len(images), h))
    for i, (scale, image) in enumerate(images):
        grid.paste(image, (i * w, 0))
        ImageDraw.Draw(grid).text((i * w + 10, 10), f"LORA_SCALE={scale}", fill="red")
    grid.save(out / "grid.png")
    print(f"OK -> {out / 'grid.png'}")


if __name__ == "__main__":
    main()
