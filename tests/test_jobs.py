from types import SimpleNamespace

import pytest

from app import jobs
from app.db import SessionLocal, init_db
from app.jobs import create_job, execute_job, mark_job_failed
from app.storage import get_storage


def test_execute_job_deletes_inputs_even_when_generation_fails():
    init_db()
    storage = get_storage()
    storage.save(b"x", "uploads/test_cleanup.png")
    db = SessionLocal()
    job = create_job(db, mode="img2img", prompt="Homer", input_image_key="uploads/test_cleanup.png")
    execute_job(job.id)
    db.refresh(job)
    assert job.status == "failed"  # SKIP_MODEL_LOAD : pas de pipeline
    with pytest.raises(FileNotFoundError):
        storage.load("uploads/test_cleanup.png")
    db.close()


def test_mark_job_failed_flags_job_killed_mid_run():
    init_db()
    db = SessionLocal()
    job = create_job(db, mode="text", prompt="Homer")
    mark_job_failed(SimpleNamespace(args=[job.id]), None, RuntimeError, RuntimeError("killed"), None)
    db.refresh(job)
    assert job.status == "failed" and "interrompue" in job.error
    db.close()


def test_execute_job_fails_cleanly_when_storage_config_is_broken(monkeypatch):
    def broken():
        raise RuntimeError("S3_BUCKET manquant")

    monkeypatch.setattr(jobs, "get_storage", broken)
    init_db()
    db = SessionLocal()
    job = create_job(db, mode="text", prompt="Homer")
    execute_job(job.id)
    db.refresh(job)
    assert job.status == "failed"
    assert "S3_BUCKET" not in job.error
    db.close()
