"""Tests de contrat pour /jobs, sans dépendre d'un GPU.

SKIP_MODEL_LOAD=1 (conftest.py) laisse ML_MODELS vide : la génération échoue
donc toujours avec un job "failed", ce qui est suffisant pour valider le
routing, la validation Pydantic et la persistance — pas la qualité d'image.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_submit_text_job_then_poll_until_failure():
    with TestClient(app) as client:
        submit = client.post("/jobs/text", json={"character_name": "Homer Simpson", "steps": 10})
        assert submit.status_code == 200
        job = submit.json()
        assert job["mode"] == "text"
        assert job["status"] in ("queued", "running", "failed")

        detail = client.get(f"/jobs/{job['id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "failed"
        assert detail.json()["error"]


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
