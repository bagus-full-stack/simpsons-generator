"""Exécution d'un job de génération.

execute_job() est le point d'entrée unique appelé aussi bien par l'exécuteur
"inline" (même process que l'API, cf. queue_backend.InlineExecutor) que par un
worker RQ séparé (queue_backend.RQExecutor + app/worker.py) : c'est le seul
endroit qui touche au GPU, à la modération et au stockage.
"""

from __future__ import annotations

import logging
import time

from .db import SessionLocal
from .generation import generate_image, image_to_png_bytes
from .metrics import generation_seconds, jobs_total
from .models_db import Job, JobStatus
from .moderation import ModerationError, check_image, check_prompt
from .storage import get_storage

log = logging.getLogger(__name__)


def create_job(
    db,
    *,
    mode: str,
    prompt: str,
    steps: int = 30,
    turbo: bool = False,
    strength: float = 0.8,
    input_image_key: str | None = None,
    mask_image_key: str | None = None,
) -> Job:
    job = Job(
        mode=mode,
        prompt=prompt,
        steps=steps,
        turbo=turbo,
        strength=strength,
        input_image_key=input_image_key,
        mask_image_key=mask_image_key,
        status=JobStatus.QUEUED.value,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def execute_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.warning("Job introuvable : %s", job_id)
            return

        job.status = JobStatus.RUNNING.value
        db.commit()

        storage = None
        start = time.time()
        try:
            storage = get_storage()
            check_prompt(job.prompt)

            input_bytes = storage.load(job.input_image_key) if job.input_image_key else None
            mask_bytes = storage.load(job.mask_image_key) if job.mask_image_key else None

            image = generate_image(
                mode=job.mode,
                prompt=job.prompt,
                steps=job.steps,
                turbo=job.turbo,
                input_image_bytes=input_bytes,
                mask_image_bytes=mask_bytes,
                strength=job.strength,
            )

            check_image(image)

            key = f"generated/{job.id}.png"
            storage.save(image_to_png_bytes(image), key)

            job.image_url = storage.url_for(key)
            job.duration_ms = int((time.time() - start) * 1000)
            job.status = JobStatus.DONE.value

            jobs_total.labels(mode=job.mode, status="done").inc()
            generation_seconds.labels(mode=job.mode).observe(job.duration_ms / 1000)

        except ModerationError as exc:
            job.status = JobStatus.REJECTED.value
            job.error = str(exc)
            jobs_total.labels(mode=job.mode, status="rejected").inc()

        except Exception:  # noqa: BLE001 — on veut logger *toute* erreur d'inférence
            log.exception("Échec du job %s", job_id)
            job.status = JobStatus.FAILED.value
            job.error = "La génération a échoué."  # détails (chemins, CUDA...) dans les logs, pas chez le client
            jobs_total.labels(mode=job.mode, status="failed").inc()

        db.commit()

        # Les entrées ne servent plus : on ne garde que l'image générée.
        for key in (job.input_image_key, job.mask_image_key):
            if key and storage:
                try:
                    storage.delete(key)
                except Exception:  # noqa: BLE001 — un fichier orphelin ne doit pas faire échouer un job terminé
                    log.warning("Suppression de %s impossible.", key)
    finally:
        db.close()


def mark_job_failed(job, connection, exc_type, exc_value, traceback) -> None:
    """Callback RQ (on_failure) : le worker a été tué avant que execute_job puisse noter l'échec."""
    db = SessionLocal()
    try:
        row = db.get(Job, job.args[0])
        if row is not None and row.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            row.status = JobStatus.FAILED.value
            row.error = "La génération a été interrompue (timeout ou arrêt du worker)."
            log.error("Job %s interrompu : %s: %s", row.id, exc_type.__name__, exc_value)
            db.commit()
    finally:
        db.close()
