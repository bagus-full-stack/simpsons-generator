"""Chargement des pipelines Stable Diffusion.

Repris de l'ancien simpsonGeneratorAPI.py (mêmes 4 pipelines, mêmes adaptateurs
LoRA simpson + LCM), avec les chemins et le modèle de base lus depuis la config
au lieu d'être codés en dur. torch/diffusers ne sont importés qu'à l'intérieur
de load_pipelines(), pour que le reste de l'API (et les tests) puisse s'importer
sans dépendre d'un GPU ni d'un torch installé.
"""

from __future__ import annotations

import logging
import os

from .config import get_settings

log = logging.getLogger(__name__)

ML_MODELS: dict = {}

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
        ControlNetModel,
        DPMSolverMultistepScheduler,
        LCMScheduler,
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )

    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info("Chargement des modèles sur %s...", device)

    text_pipe = StableDiffusionPipeline.from_pretrained(
        settings.base_model_id, torch_dtype=dtype, safety_checker=None
    ).to(device)

    lora_weights = settings.lora_dir / "pytorch_lora_weights.safetensors"
    if lora_weights.exists():
        text_pipe.load_lora_weights(
            str(settings.lora_dir), weight_name="pytorch_lora_weights.safetensors", adapter_name="simpson"
        )
    else:
        log.warning("Poids LoRA introuvables (%s) : le style Simpsons ne sera pas appliqué.", lora_weights)

    text_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
    text_pipe.set_adapters(["simpson", "lcm"], adapter_weights=[1.0, 0.0])

    dpm_scheduler = DPMSolverMultistepScheduler.from_config(text_pipe.scheduler.config)
    lcm_scheduler = LCMScheduler.from_config(text_pipe.scheduler.config)

    def configure_pipe(pipe):
        pipe.scheduler_dpm = dpm_scheduler
        pipe.scheduler_lcm = lcm_scheduler
        pipe.scheduler = dpm_scheduler
        return pipe

    configure_pipe(text_pipe)

    img_pipe = configure_pipe(StableDiffusionImg2ImgPipeline(**text_pipe.components))
    inpaint_pipe = configure_pipe(StableDiffusionInpaintPipeline(**text_pipe.components))

    log.info("Chargement de ControlNet Canny...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype).to(device)
    canny_pipe = configure_pipe(
        StableDiffusionControlNetPipeline(
            controlnet=controlnet,
            vae=text_pipe.vae,
            text_encoder=text_pipe.text_encoder,
            tokenizer=text_pipe.tokenizer,
            unet=text_pipe.unet,
            scheduler=text_pipe.scheduler,
            safety_checker=None,
            feature_extractor=text_pipe.feature_extractor,
        ).to(device)
    )

    ML_MODELS.update(text_pipe=text_pipe, img_pipe=img_pipe, canny_pipe=canny_pipe, inpaint_pipe=inpaint_pipe)
    log.info("Pipelines prêtes.")


def get_pipeline(mode: str):
    key = MODE_TO_PIPELINE_KEY.get(mode)
    if key is None:
        raise ValueError(f"Mode de génération inconnu : {mode!r}")
    pipe = ML_MODELS.get(key)
    if pipe is None:
        raise RuntimeError(f"Pipeline '{key}' non chargée (SKIP_MODEL_LOAD actif, ou démarrage en cours).")
    return pipe


def unload_pipelines() -> None:
    ML_MODELS.clear()
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
