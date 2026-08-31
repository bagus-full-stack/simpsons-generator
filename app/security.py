"""Authentification par clé API + rate limiting.

L'ancienne API n'avait ni l'un ni l'autre : n'importe qui pouvait déclencher
des générations GPU depuis n'importe quel site (CORS ouvert à "*").
"""

from __future__ import annotations

import logging

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import get_settings

log = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

_warned_no_api_key = False


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()

    if not settings.api_keys:
        global _warned_no_api_key
        if not _warned_no_api_key:
            log.warning(
                "API_KEYS n'est pas configuré : les endpoints de génération sont "
                "accessibles sans authentification. À utiliser en dev uniquement."
            )
            _warned_no_api_key = True
        return

    if x_api_key is None or x_api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API invalide ou manquante.")
