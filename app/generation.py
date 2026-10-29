"""Logique de génération d'image, extraite de run_generation() dans l'ancienne
API. Fonctions pures : elles reçoivent des bytes en entrée et renvoient une
image PIL, sans toucher au disque ni à la base — la persistance est gérée par
app.jobs.execute_job.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageOps

from .config import get_settings
from .pipelines import LOADED_ADAPTERS, get_pipeline

# Identiques à simpsonGenerator_FIXED.ipynb : style cartoon Simpsons, peau jaune.
NEGATIVE_PROMPT = (
    "realistic, photograph, photo, 3d render, ugly, deformed, "
    "blurry, low quality, distorted, bad anatomy, extra limbs"
)


def build_prompt(character_or_prompt: str) -> str:
    return f"{character_or_prompt} as a simpson character, yellow skin, cartoon style, high quality, detailed"


def process_image_bytes(data: bytes) -> Image.Image:
    resolution = get_settings().image_resolution
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return ImageOps.fit(img, (resolution, resolution), method=Image.Resampling.LANCZOS)


def process_mask_bytes(data: bytes) -> Image.Image:
    resolution = get_settings().image_resolution
    mask = Image.open(io.BytesIO(data)).convert("L")
    return ImageOps.fit(mask, (resolution, resolution), method=Image.Resampling.LANCZOS)


def get_canny_image(image: Image.Image) -> Image.Image:
    import cv2  # import différé : dépendance lourde, seulement nécessaire pour le mode "pose"

    array = np.array(image)
    canny = cv2.Canny(array, 100, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    return Image.fromarray(canny)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _hires_fix(image: Image.Image, prompt: str, steps: int, guidance: float) -> Image.Image:
    """Upscale puis img2img léger : affine les détails sans recomposer l'image."""
    settings = get_settings()
    size = tuple(int(d * settings.hires_scale) // 8 * 8 for d in image.size)
    img_pipe = get_pipeline("img2img")
    img_pipe.scheduler = img_pipe.scheduler_default
    return img_pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image.resize(size, Image.Resampling.LANCZOS),
        strength=settings.hires_strength,
        # int(steps × strength) doit rester >= 1, sinon diffusers refuse (steps=2, strength=0.4 -> 0).
        num_inference_steps=max(steps, math.ceil(1 / settings.hires_strength)),
        guidance_scale=guidance,
    ).images[0]


def generate_image(
    mode: str,
    prompt: str,
    steps: int,
    turbo: bool,
    input_image_bytes: bytes | None = None,
    mask_image_bytes: bytes | None = None,
    strength: float = 0.8,
) -> Image.Image:
    pipe = get_pipeline(mode)
    full_prompt = build_prompt(prompt)

    adapters = LOADED_ADAPTERS
    weight_by_adapter = {"simpson": get_settings().lora_scale, "lcm": 1.0 if turbo else 0.0}
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=[weight_by_adapter[a] for a in adapters])

    if turbo:
        pipe.scheduler = pipe.scheduler_lcm
        actual_steps, actual_guidance = 6, 1.5
    else:
        pipe.scheduler = pipe.scheduler_default
        actual_steps, actual_guidance = steps, 7.5

    args: dict = {
        "prompt": full_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": actual_steps,
        "guidance_scale": actual_guidance,
    }

    try:
        if mode == "img2img":
            if input_image_bytes is None:
                raise ValueError("input_image_bytes requis pour le mode img2img")
            args.update(image=process_image_bytes(input_image_bytes), strength=strength)
        elif mode == "pose":
            if input_image_bytes is None:
                raise ValueError("input_image_bytes requis pour le mode pose")
            edge_map = get_canny_image(process_image_bytes(input_image_bytes))
            args.update(image=edge_map, controlnet_conditioning_scale=1.0)
            # ControlNet reste sur CPU hors utilisation (voir pipelines.py) pour laisser
            # de la VRAM aux 3 autres modes sur un GPU 6 Go — remonté ici pour la durée du job.
            pipe.controlnet.to(pipe.unet.device)
        elif mode == "inpaint":
            if input_image_bytes is None or mask_image_bytes is None:
                raise ValueError("input_image_bytes et mask_image_bytes requis pour le mode inpaint")
            args.update(
                image=process_image_bytes(input_image_bytes),
                mask_image=process_mask_bytes(mask_image_bytes),
                strength=0.95,
            )
        else:
            settings = get_settings()
            args.update(height=settings.image_height, width=settings.image_resolution)

        image = pipe(**args).images[0]
        if mode == "text" and not turbo and get_settings().hires_fix:
            image = _hires_fix(image, full_prompt, actual_steps, actual_guidance)
        return image
    finally:
        # Toujours revenir au scheduler par défaut, même en cas d'erreur en plein
        # milieu d'une génération turbo (comportement repris de l'ancienne API).
        if hasattr(pipe, "scheduler_default"):
            pipe.scheduler = pipe.scheduler_default
        if mode == "pose" and hasattr(pipe, "controlnet"):
            import torch

            pipe.controlnet.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
