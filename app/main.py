"""Point d'entrée de l'API : `uvicorn app.main:app`.

Remplace simpsonGeneratorAPI.py — mêmes 4 modes de génération, mais découplés
de l'inférence (voir app/routes/jobs.py), avec auth, rate limiting, CORS
restreint, logs structurés et métriques.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import get_settings
from .db import init_db
from .generation import warm_up_pipelines
from .jobs import fail_interrupted_jobs
from .logging_config import configure_logging
from .metrics import CONTENT_TYPE_LATEST, generate_latest
from .moderation import warm_up
from .pipelines import load_pipelines, unload_pipelines
from .routes.jobs import router as jobs_router
from .security import limiter

configure_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.environment == "production" and not settings.api_keys:
        raise RuntimeError(
            "ENVIRONMENT=production nécessite au moins une clé dans API_KEYS "
            "(sinon les endpoints de génération restent ouverts sans authentification)."
        )

    init_db()
    # Avec QUEUE_BACKEND=rq, c'est le worker qui génère : l'API ne doit pas occuper la VRAM avec une 2e copie.
    if settings.queue_backend == "inline":
        fail_interrupted_jobs()
        load_pipelines()
        warm_up()
        warm_up_pipelines()

    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1, environment=settings.environment)
        log.info("Sentry initialisé.")

    yield

    unload_pipelines()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="Simpson Generator API", version="2.0.0", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.storage_backend == "local":
        settings.output_folder.mkdir(parents=True, exist_ok=True)
        app.mount("/images", StaticFiles(directory=str(settings.output_folder)), name="images")

    app.include_router(jobs_router)

    @app.get("/health")
    def health() -> dict:
        # uvicorn n'ouvre le port qu'après le lifespan (modèles chargés) : si ça répond, l'API est prête.
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _rate_limit_handler(request, exc: RateLimitExceeded) -> Response:
    return Response(
        content='{"detail":"Trop de requêtes, réessayez plus tard."}',
        media_type="application/json",
        status_code=429,
    )


app = create_app()
