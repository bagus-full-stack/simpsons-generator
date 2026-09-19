"""Chargement des pipelines Stable Diffusion XL.

Repris de l'ancien simpsonGeneratorAPI.py (mêmes 4 pipelines, mêmes adaptateurs
LoRA simpson + LCM), migré de SD1.5 vers SDXL (meilleure qualité, 1024x1024
natif) tout en gardant les mêmes temps de génération grâce à LCM-LoRA-SDXL.
Les chemins et modèles sont lus depuis la config au lieu d'être codés en dur.
torch/diffusers ne sont importés qu'à l'intérieur de load_pipelines(), pour que
le reste de l'API (et les tests) puisse s'importer sans dépendre d'un GPU ni
d'un torch installé.
"""

from __future__ import annotations

import logging
import os

from .config import get_settings

log = logging.getLogger(__name__)

ML_MODELS: dict = {}

# Adaptateurs LoRA effectivement chargés sur text_pipe (et donc partagés par les
# autres pipelines, qui réutilisent les mêmes composants) — "simpson" peut être
# absent si les poids sont introuvables ou incompatibles (ex: encore un LoRA
# SD1.5 le temps de terminer le réentraînement sur base SDXL).
LOADED_ADAPTERS: list[str] = []

MODE_TO_PIPELINE_KEY = {
    "text": "text_pipe",
    "img2img": "img_pipe",
    "pose": "canny_pipe",
    "inpaint": "inpaint_pipe",
}


def load_pipelines() -> None:
    settings = get_settings()

    if settings.skip_model_load:
        log.warning("SKIP_MODEL_LOAD actif : pipelines non chargés (mode test/CI).")
        return

    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DPMSolverMultistepScheduler,
        LCMScheduler,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )

    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info("Chargement des modèles sur %s...", device)

    pipe_kwargs: dict = {"torch_dtype": dtype}
    if settings.vae_model_id:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(settings.vae_model_id, torch_dtype=dtype)

    text_pipe = StableDiffusionXLPipeline.from_pretrained(settings.base_model_id, **pipe_kwargs).to(device)
    # VAE decode of a full 1024x1024 SDXL latent is memory-hungry enough to OOM even
    # when the denoising loop itself fits in VRAM. vae is shared across all 4 pipelines
    # (see text_pipe.components below), so enabling tiling once here covers all of them.
    text_pipe.enable_vae_tiling()
    text_pipe.enable_vae_slicing()

    LOADED_ADAPTERS.clear()
    lora_weights = settings.lora_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        try:
            text_pipe.load_lora_weights(
                str(settings.lora_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="simpson"
            )
            LOADED_ADAPTERS.append("simpson")
        except Exception:
            log.exception(
                "Échec du chargement du LoRA Simpsons (%s) : incompatible avec %s ? "
                "(ex: encore un LoRA SD1.5 le temps de réentraîner sur SDXL) — "
                "le style Simpsons ne sera pas appliqué.",
                lora_weights,
                settings.base_model_id,
            )
    else:
        log.warning("Poids LoRA introuvables (%s) : le style Simpsons ne sera pas appliqué.", lora_weights)

    text_pipe.load_lora_weights(settings.lcm_lora_id, adapter_name="lcm")
    LOADED_ADAPTERS.append("lcm")
    text_pipe.set_adapters(LOADED_ADAPTERS, adapter_weights=[0.0 for _ in LOADED_ADAPTERS])

    dpm_scheduler = DPMSolverMultistepScheduler.from_config(text_pipe.scheduler.config)
    lcm_scheduler = LCMScheduler.from_config(text_pipe.scheduler.config)

    def configure_pipe(pipe):
        pipe.scheduler_dpm = dpm_scheduler
        pipe.scheduler_lcm = lcm_scheduler
        pipe.scheduler = dpm_scheduler
        return pipe

    configure_pipe(text_pipe)

    img_pipe = configure_pipe(StableDiffusionXLImg2ImgPipeline(**text_pipe.components))
    inpaint_pipe = configure_pipe(StableDiffusionXLInpaintPipeline(**text_pipe.components))

    log.info("Chargement de ControlNet Canny (SDXL)...")
    controlnet = ControlNetModel.from_pretrained(settings.controlnet_model_id, torch_dtype=dtype).to(device)
    canny_pipe = configure_pipe(
        StableDiffusionXLControlNetPipeline(controlnet=controlnet, **text_pipe.components).to(device)
    )

    ML_MODELS.update(text_pipe=text_pipe, img_pipe=img_pipe, canny_pipe=canny_pipe, inpaint_pipe=inpaint_pipe)
    log.info("Pipelines prêtes (adaptateurs chargés : %s).", LOADED_ADAPTERS)


def get_pipeline(mode: str):
    key = MODE_TO_PIPELINE_KEY.get(mode)
    if key is None:
        raise ValueError(f"Mode de génération inconnu : {mode!r}")
    pipe = ML_MODELS.get(key)
    if pipe is None:
        raise RuntimeError(f"Pipeline '{key}' non chargée (SKIP_MODEL_LOAD actif, ou démarrage en cours).")
    return pipe


def get_loaded_adapters() -> list[str]:
    """Adaptateurs LoRA effectivement disponibles sur les pipelines chargées
    (voir LOADED_ADAPTERS) — utilisé par generation.py pour n'activer que ce
    qui a réellement été chargé (ex: "simpson" peut être absent)."""
    return LOADED_ADAPTERS


def unload_pipelines() -> None:
    ML_MODELS.clear()
    LOADED_ADAPTERS.clear()
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
