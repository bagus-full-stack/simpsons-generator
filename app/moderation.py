"""Modération de contenu.

L'ancienne API tournait avec `safety_checker=None` et une bibliothèque de
prompts ciblant des personnalités réelles (voir training/prompts.md) — sans
aucun filtre, ouvrir ça au public est un risque légal et réputationnel direct.

Deux niveaux, tous deux activables/désactivables indépendamment :
  1. `check_prompt`  — liste de blocage textuelle sur la requête, avant tout calcul GPU.
  2. `check_image`   — classifieur NSFW sur l'image produite, avant l'écriture en stockage.

Ce n'est pas un remplacement pour un vrai service de modération (Azure Content
Safety, AWS Rekognition, Sightengine, ...) : c'est un filet de sécurité minimal
qui pose le bon endroit dans le pipeline pour brancher un vrai fournisseur.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)


class ModerationError(Exception):
    """Levée quand un prompt ou une image est rejeté par la politique de contenu."""


def check_prompt(prompt: str) -> None:
    settings = get_settings()
    lowered = prompt.lower()
    for term in settings.moderation_blocklist:
        if term.lower() in lowered:
            log.warning("Prompt rejeté par la liste de blocage.")
            raise ModerationError("Ce prompt enfreint la politique de contenu.")


_nsfw_classifier = None


def check_image(image: "Image.Image") -> None:
    settings = get_settings()
    if not settings.enable_nsfw_filter:
        return

    global _nsfw_classifier
    if _nsfw_classifier is None:
        from transformers import pipeline as hf_pipeline

        log.info("Chargement du classifieur NSFW '%s'...", settings.nsfw_model_id)
        _nsfw_classifier = hf_pipeline("image-classification", model=settings.nsfw_model_id)

    results = _nsfw_classifier(image)
    top = max(results, key=lambda r: r["score"])
    if top["label"].lower() == "nsfw" and top["score"] >= settings.nsfw_threshold:
        log.warning("Image générée rejetée par le filtre NSFW (score=%.2f).", top["score"])
        raise ModerationError("L'image générée a été rejetée par le filtre de contenu.")
