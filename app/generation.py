"""Logique de génération d'image, extraite de run_generation() dans l'ancienne
API. Fonctions pures : elles reçoivent des bytes en entrée et renvoient une
image PIL, sans toucher au disque ni à la base — la persistance est gérée par
app.jobs.execute_job.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from .pipelines import get_pipeline

NEGATIVE_PROMPT = "realistic, photo, ugly, deformed, blurry, bad anatomy"


def build_prompt(character_or_prompt: str) -> str:
    return f"{character_or_prompt} as a simpson character, yellow skin, cartoon style, high quality"


def process_image_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return ImageOps.fit(img, (512, 512), method=Image.Resampling.LANCZOS)


def process_mask_bytes(data: bytes) -> Image.Image:
    mask = Image.open(io.BytesIO(data)).convert("L")
    return ImageOps.fit(mask, (512, 512), method=Image.Resampling.LANCZOS)


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

    if turbo:
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 1.0])
        pipe.scheduler = pipe.scheduler_lcm
        actual_steps, actual_guidance = 6, 1.5
    else:
        pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])
        pipe.scheduler = pipe.scheduler_dpm
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
        elif mode == "inpaint":
            if input_image_bytes is None or mask_image_bytes is None:
                raise ValueError("input_image_bytes et mask_image_bytes requis pour le mode inpaint")
            args.update(
                image=process_image_bytes(input_image_bytes),
                mask_image=process_mask_bytes(mask_image_bytes),
                strength=0.95,
            )
        else:
            args.update(height=512, width=512)

        return pipe(**args).images[0]
    finally:
        # Toujours revenir au scheduler par défaut, même en cas d'erreur en plein
        # milieu d'une génération turbo (comportement repris de l'ancienne API).
        if hasattr(pipe, "scheduler_dpm"):
            pipe.scheduler = pipe.scheduler_dpm
