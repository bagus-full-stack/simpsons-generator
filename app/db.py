"""Accès base de données. SQLite par défaut (zéro dépendance externe pour le
dev local), Postgres en prod via DATABASE_URL — voir .env.example.
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

_settings = get_settings()
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}

engine = create_engine(_settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models_db  # noqa: F401  (enregistre les modèles sur Base.metadata)

    # shortcut: create_all ne migre pas les tables existantes ; passer à Alembic au premier changement de schéma.
    Base.metadata.create_all(bind=engine)
    # shortcut: colonne ajoutée à la main pour les bases déjà déployées ; Alembic au prochain changement.
    if "client_id" not in {c["name"] for c in inspect(engine).get_columns("jobs")}:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN client_id VARCHAR(64)"))
            conn.execute(text("CREATE INDEX ix_jobs_client_id ON jobs (client_id)"))
