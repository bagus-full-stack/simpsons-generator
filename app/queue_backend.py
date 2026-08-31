"""Abstraction de file d'attente.

- InlineExecutor (par défaut) : exécute le job tout de suite, dans le process
  API. Zéro dépendance externe, comportement le plus proche de l'ancienne API
  pour le dev local — mais un seul job à la fois, comme avant.
- RQExecutor : dépose le job dans Redis, consommé par un ou plusieurs process
  `python -m app.worker` séparés. C'est ce backend qui permet de scaler
  horizontalement le nombre de workers GPU indépendamment de l'API.

Les deux implémentent la même interface (`submit(job_id)`), donc les routes
n'ont pas besoin de savoir laquelle est active.
"""

from __future__ import annotations

from typing import Protocol

from .config import get_settings


class Executor(Protocol):
    def submit(self, job_id: str) -> None: ...


class InlineExecutor:
    def submit(self, job_id: str) -> None:
        from .jobs import execute_job

        execute_job(job_id)


class RQExecutor:
    def __init__(self, redis_url: str, queue_name: str = "simpsons-generation") -> None:
        import redis
        from rq import Queue

        self._queue = Queue(queue_name, connection=redis.from_url(redis_url))

    def submit(self, job_id: str) -> None:
        self._queue.enqueue("app.jobs.execute_job", job_id, job_timeout=600)


def get_executor() -> Executor:
    settings = get_settings()
    if settings.queue_backend == "rq":
        return RQExecutor(settings.redis_url)
    return InlineExecutor()
