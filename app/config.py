"""Configuration centralisée, lue depuis l'environnement (ou un fichier .env).

Toutes les valeurs qui étaient codées en dur dans l'ancien simpsonGeneratorAPI.py
(chemins, modèle de base, CORS...) passent désormais par des variables d'environnement,
pour permettre de faire tourner la même image Docker en dev/staging/prod.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Environnement ---
    environment: str = "development"  # "development" | "production"
    # Coupe le chargement des pipelines diffusers (utilisé par la suite de tests,
    # pour ne pas dépendre d'un GPU/de torch pour valider la logique API).
    skip_model_load: bool = False

    # --- Modèle ---
    # SD1.5 de base, identique à simpsonGenerator_FIXED.ipynb (celui qui a entraîné le LoRA).
    # Temps courts en mode turbo grâce à lcm_lora_id (voir pipelines.py).
    base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # VAE affiné : couleurs et détails un peu meilleurs que celui livré avec SD1.5.
    vae_model_id: str | None = "stabilityai/sd-vae-ft-mse"
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_canny"
    image_resolution: int = 512
    # Hauteur du mode texte : 512x512 comme le notebook.
    image_height: int = 512
    lora_dir: Path = BASE_DIR / "simpsons_lora_results"
    lora_file: str = "sd15_lora.safetensors"
    # Force du LoRA Simpsons. 0.8 : plus net que 1.0 (yeux/visages moins déformés) sans perdre
    # le style (0.6 le dilue). Mesuré avec scripts/sweep_lora_scale.py ; à refaire pour un nouveau LoRA.
    lora_scale: float = 0.8
    # "Hires fix" (mode texte, hors turbo) : upscale puis 2e passe img2img légère.
    hires_fix: bool = True
    hires_scale: float = 1.5
    hires_strength: float = 0.4
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # --- Base de données (historique des jobs) ---
    database_url: str = f"sqlite:///{BASE_DIR / 'app.db'}"

    # --- File d'attente ---
    queue_backend: str = "inline"  # "inline" (synchrone, sans dépendance) | "rq" (Redis)
    redis_url: str = "redis://localhost:6379/0"

    # --- Stockage des images ---
    storage_backend: str = "local"  # "local" | "s3"
    output_folder: Path = BASE_DIR / "generated_simpsons"
    public_base_url: str = "http://localhost:8000"
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_public_base_url: str | None = None
    s3_prefix: str = ""

    # --- Sécurité ---
    cors_allow_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    api_keys: Annotated[list[str], NoDecode] = []
    rate_limit: str = "20/minute"

    # --- Modération de contenu ---
    enable_nsfw_filter: bool = True
    nsfw_model_id: str = "Falconsai/nsfw_image_detection"
    nsfw_threshold: float = 0.85
    moderation_blocklist: Annotated[list[str], NoDecode] = ["child", "minor", "underage", "loli"]

    # --- Observabilité ---
    sentry_dsn: str | None = None

    @field_validator("cors_allow_origins", "api_keys", "moderation_blocklist", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
