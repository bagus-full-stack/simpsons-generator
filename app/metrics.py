"""Métriques Prometheus minimales — l'ancienne API n'exposait rien de mesurable
à part des print() avec des emojis."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

jobs_total = Counter("simpsons_jobs_total", "Jobs de génération traités", ["mode", "status"])
generation_seconds = Histogram("simpsons_generation_seconds", "Durée de génération en secondes", ["mode"])

__all__ = ["jobs_total", "generation_seconds", "generate_latest", "CONTENT_TYPE_LATEST"]
