"""Chargement des pipelines Stable Diffusion 1.5 (Realistic Vision).

Repris de l'ancien simpsonGeneratorAPI.py (mêmes 4 pipelines, mêmes adaptateurs
LoRA simpson + LCM), basé sur SD1.5 (Realistic Vision) comme le notebook, avec LCM-LoRA SD1.5 pour le mode turbo.
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


def _lora_matches_unet(weights_path, expected_cross_attention_dim: int) -> bool:
    """Vérifie la dim du cross-attention du LoRA (2048 pour SDXL, 768 pour SD1.5)
    sans charger les tenseurs, pour éviter l'assign=True destructif de load_lora_weights()
    sur un checkpoint incompatible (voir commentaire dans load_pipelines)."""
    from safetensors import safe_open

    with safe_open(str(weights_path), framework="pt") as f:
        for key in f.keys():
            if "attn2.to_k.lora_A" in key or "attn2.to_v.lora_A" in key:
                return f.get_slice(key).get_shape()[-1] == expected_cross_attention_dim
    return True


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

    # safety_checker=None : la modération est faite en amont (app.moderation).
    pipe_kwargs: dict = {"torch_dtype": dtype, "safety_checker": None}
    if settings.vae_model_id:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(settings.vae_model_id, torch_dtype=dtype)

    text_pipe = StableDiffusionPipeline.from_pretrained(settings.base_model_id, **pipe_kwargs).to(device)
    # Slicing seul : économise la VRAM sans changer le rendu (le tiling, lui, oui).
    text_pipe.enable_vae_slicing()

    LOADED_ADAPTERS.clear()
    lora_weights = settings.lora_dir / settings.lora_file
    if not lora_weights.exists():
        log.warning("Poids LoRA introuvables (%s) : le style Simpsons ne sera pas appliqué.", lora_weights)
    elif not _lora_matches_unet(lora_weights, text_pipe.unet.config.cross_attention_dim):
        # load_lora_weights() utilise load_state_dict(strict=False, assign=True), qui assigne
        # les tenseurs du checkpoint par référence AVANT de lever l'erreur de shape mismatch —
        # un LoRA incompatible (ex: encore entraîné sur SD1.5) laisse donc l'UNet partagé (utilisé
        # par tous les pipelines via text_pipe.components) avec des tenseurs meta (vides), qui
        # plantent au .to(device) suivant. Un check de shape en amont évite d'y toucher.
        log.warning(
            "LoRA Simpsons (%s) incompatible avec %s (LoRA entraîné sur une autre base que SD1.5 ?) — "
            "chargement ignoré, le style Simpsons ne sera pas appliqué.",
            lora_weights,
            settings.base_model_id,
        )
    else:
        text_pipe.load_lora_weights(
            str(settings.lora_dir), weight_name=settings.lora_file, adapter_name="simpson"
        )
        LOADED_ADAPTERS.append("simpson")

    text_pipe.load_lora_weights(settings.lcm_lora_id, adapter_name="lcm")
    LOADED_ADAPTERS.append("lcm")
    text_pipe.set_adapters(LOADED_ADAPTERS, adapter_weights=[0.0 for _ in LOADED_ADAPTERS])

    # DPMSolverMultistep (comme le notebook) + sigmas Karras : détails plus nets à steps égaux.
    default_scheduler = DPMSolverMultistepScheduler.from_config(
        text_pipe.scheduler.config, use_karras_sigmas=True
    )
    lcm_scheduler = LCMScheduler.from_config(text_pipe.scheduler.config)

    def configure_pipe(pipe):
        pipe.scheduler_default = default_scheduler
        pipe.scheduler_lcm = lcm_scheduler
        pipe.scheduler = default_scheduler
        return pipe

    configure_pipe(text_pipe)

    img_pipe = configure_pipe(StableDiffusionImg2ImgPipeline(**text_pipe.components))
    inpaint_pipe = configure_pipe(StableDiffusionInpaintPipeline(**text_pipe.components))

    log.info("Chargement de ControlNet Canny (SD1.5)...")
    # Reste sur CPU : ~2.5 Go en fp16, gardé en permanence sur un GPU 6 Go ne laisse plus
    # de marge pour les 3 autres modes qui ne s'en servent jamais. generate_image() le
    # bascule sur le device au moment d'un job "pose" puis le redescend juste après.
    controlnet = ControlNetModel.from_pretrained(settings.controlnet_model_id, torch_dtype=dtype)
    canny_pipe = configure_pipe(StableDiffusionControlNetPipeline(controlnet=controlnet, **text_pipe.components))

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
