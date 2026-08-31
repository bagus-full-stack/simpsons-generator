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
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    lora_dir: Path = BASE_DIR / "simpsons_lora_results"
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
