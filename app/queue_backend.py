"""Abstraction de file d'attente.

- InlineExecutor (par défaut) : exécute le job dans un thread dédié du process
  API, un job à la fois (les autres attendent en file) : la requête HTTP
  répond tout de suite. Zéro dépendance externe, idéal pour le dev local.
- RQExecutor : dépose le job dans Redis, consommé par un ou plusieurs process
  `python -m app.worker` séparés. C'est ce backend qui permet de scaler
  horizontalement le nombre de workers GPU indépendamment de l'API.

Les deux implémentent la même interface (`submit(job_id)`), donc les routes
n'ont pas besoin de savoir laquelle est active.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Protocol

from .config import get_settings


class Executor(Protocol):
    def submit(self, job_id: str) -> None: ...


log = logging.getLogger(__name__)

# Un seul thread : les pipelines (scheduler, poids LoRA) sont partagés et la VRAM est limitée.
_inline_pool = ThreadPoolExecutor(max_workers=1)


class InlineExecutor:
    def submit(self, job_id: str) -> None:
        from .jobs import execute_job

        future = _inline_pool.submit(execute_job, job_id)
        future.add_done_callback(lambda f: f.exception() and log.error("Job %s : %r", job_id, f.exception()))


class RQExecutor:
    def __init__(self, redis_url: str, queue_name: str = "simpsons-generation") -> None:
        import redis
        from rq import Queue

        self._queue = Queue(queue_name, connection=redis.from_url(redis_url))

    def submit(self, job_id: str) -> None:
        from rq.job import Callback

        from .jobs import mark_job_failed

        # on_failure : si le worker est tué (timeout, mémoire), execute_job ne peut plus noter l'échec.
        self._queue.enqueue(
            "app.jobs.execute_job", job_id, job_timeout=600, on_failure=Callback(mark_job_failed)
        )


@lru_cache
def get_executor() -> Executor:
    settings = get_settings()
    if settings.queue_backend == "rq":
        return RQExecutor(settings.redis_url)
    return InlineExecutor()
