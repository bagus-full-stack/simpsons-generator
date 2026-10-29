"""Tests de contrat pour /jobs, sans dépendre d'un GPU.

SKIP_MODEL_LOAD=1 (conftest.py) laisse ML_MODELS vide : la génération échoue
donc toujours avec un job "failed", ce qui est suffisant pour valider le
routing, la validation Pydantic et la persistance — pas la qualité d'image.
"""

import time

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_submit_text_job_then_poll_until_failure():
    with TestClient(app) as client:
        submit = client.post("/jobs/text", json={"character_name": "Homer Simpson", "steps": 10})
        assert submit.status_code == 200
        job = submit.json()
        assert job["mode"] == "text"
        assert job["status"] in ("queued", "running", "failed")

        for _ in range(50):  # le job tourne dans un thread : on attend son statut final
            detail = client.get(f"/jobs/{job['id']}")
            if detail.json()["status"] in ("failed", "done"):
                break
            time.sleep(0.1)
        assert detail.status_code == 200
        assert detail.json()["status"] == "failed"
        assert detail.json()["error"]
        assert "text_pipe" not in detail.json()["error"]  # pas de détail interne côté client


def test_submit_text_job_rejects_empty_prompt():
    with TestClient(app) as client:
        response = client.post("/jobs/text", json={"character_name": ""})
        assert response.status_code == 422


def test_get_unknown_job_returns_404():
    with TestClient(app) as client:
        response = client.get("/jobs/does-not-exist")
        assert response.status_code == 404


def test_list_jobs_only_returns_done_jobs():
    with TestClient(app) as client:
        client.post("/jobs/text", json={"character_name": "Marge Simpson"})
        response = client.get("/jobs")
        assert response.status_code == 200
        assert all(job["status"] == "done" for job in response.json())


def test_upload_over_limit_is_rejected_with_413():
    with TestClient(app) as client:
        big = b"0" * (8 * 1024 * 1024 + 1)
        response = client.post("/jobs/img2img", data={"prompt": "Homer"}, files={"file": ("a.png", big, "image/png")})
        assert response.status_code == 413


def test_file_routes_bound_steps():
    with TestClient(app) as client:
        response = client.post(
            "/jobs/pose", data={"prompt": "Homer", "steps": 100000}, files={"file": ("a.png", b"x", "image/png")}
        )
        assert response.status_code == 422


def test_job_routes_require_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "api_keys", ["secret"])
    with TestClient(app) as client:
        assert client.post("/jobs/text", json={"character_name": "Homer"}).status_code == 401
        assert client.get("/jobs").status_code == 401
        assert client.get("/jobs/whatever").status_code == 401
        headers = {"X-API-Key": "secret"}
        assert client.post("/jobs/text", json={"character_name": "Homer"}, headers=headers).status_code == 200
