"""Modèle de persistance des jobs de génération.

Remplace l'ancien /history qui reconstruisait le prompt en tronquant le nom de
fichier sur disque (f.split("_")[0]) : chaque génération est maintenant une
ligne en base, avec son statut, sa durée réelle et l'URL de l'image produite.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"  # bloqué par la modération de contenu


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    mode: Mapped[str] = mapped_column(String(16))
    prompt: Mapped[str] = mapped_column(Text)
    steps: Mapped[int] = mapped_column(Integer, default=30)
    turbo: Mapped[bool] = mapped_column(Boolean, default=False)
    strength: Mapped[float] = mapped_column(Float, default=0.8)

    # Clés de stockage (backend local ou S3) des entrées, pas exposées à l'API.
    input_image_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mask_image_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value)
    image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
