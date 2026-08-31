"""Worker RQ : `python -m app.worker`.

Ne sert que si QUEUE_BACKEND=rq. Charge les pipelines une fois au démarrage
(comme l'API), puis consomme la file Redis en continu. Peut tourner sur une
machine séparée de l'API — c'est ce qui permet de scaler le nombre de workers
GPU indépendamment du nombre d'instances API.
"""

from __future__ import annotations

import logging

import redis
from rq import Queue, Worker

from .config import get_settings
from .db import init_db
from .logging_config import configure_logging
from .pipelines import load_pipelines

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()

    if settings.queue_backend != "rq":
        raise RuntimeError(
            f"QUEUE_BACKEND doit valoir 'rq' pour lancer un worker (actuellement: {settings.queue_backend!r})"
        )

    init_db()
    load_pipelines()

    connection = redis.from_url(settings.redis_url)
    queue = Queue("simpsons-generation", connection=connection)
    log.info("Worker prêt, écoute la file '%s' sur %s", queue.name, settings.redis_url)

    worker = Worker([queue], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
