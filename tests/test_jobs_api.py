"""Tests de contrat pour /jobs, sans dépendre d'un GPU.

SKIP_MODEL_LOAD=1 (conftest.py) laisse ML_MODELS vide : la génération échoue
donc toujours avec un job "failed", ce qui est suffisant pour valider le
routing, la validation Pydantic et la persistance — pas la qualité d'image.
"""

import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.config import get_settings
from app.db import SessionLocal
from app.jobs import create_job
from app.main import app
from app.routes import jobs as jobs_routes
from app.storage import get_storage


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


def test_submit_is_refused_when_queue_is_full(monkeypatch):
    monkeypatch.setattr(jobs_routes, "MAX_PENDING_JOBS", 1)
    with TestClient(app) as client:  # le démarrage marque les anciens jobs en échec : la file est vide
        db = SessionLocal()
        finished = create_job(db, mode="text", prompt="Homer")
        finished.status = "done"
        db.commit()
        # un job terminé ne compte pas dans la file
        assert client.post("/jobs/text", json={"character_name": "Homer"}).status_code == 200
        create_job(db, mode="text", prompt="Marge")  # personne ne l'exécute : il reste queued
        response = client.post("/jobs/text", json={"character_name": "Bart"})
        assert response.status_code == 503
        assert "occupé" in response.json()["detail"]
        db.close()


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_ignores_the_client_filename():
    with TestClient(app) as client:
        long_name = "a" * 300 + ".png"  # 500 avant : le nom servait de clé de stockage
        response = client.post(
            "/jobs/img2img", data={"prompt": "Homer"}, files={"file": (long_name, _png(), "image/png")}
        )
        assert response.status_code == 200


def test_upload_that_is_not_an_image_is_refused():
    with TestClient(app) as client:
        response = client.post(
            "/jobs/img2img", data={"prompt": "Homer"}, files={"file": ("a.png", b"not an image", "image/png")}
        )
        assert response.status_code == 422


def test_inpaint_with_a_bad_mask_stores_nothing():
    uploads = get_storage().root / "uploads"
    before = {p.name for p in uploads.glob("*")} if uploads.exists() else set()
    with TestClient(app) as client:
        response = client.post(
            "/jobs/inpaint",
            data={"prompt": "Homer"},
            files={"image_file": ("i.png", _png(), "image/png"), "mask_file": ("m.png", b"x", "image/png")},
        )
    assert response.status_code == 422
    assert {p.name for p in uploads.glob("*")} <= before  # l'image valide n'est pas restée sur le disque


def test_blocked_prompt_is_refused_at_submit_time():
    with TestClient(app) as client:
        text = client.post("/jobs/text", json={"character_name": "a minor character"})
        assert text.status_code == 400 and "politique" in text.json()["detail"]
        png = ("a.png", _png(), "image/png")
        forms = {"img2img": {"file": png}, "pose": {"file": png}, "inpaint": {"image_file": png, "mask_file": png}}
        for route, files in forms.items():
            response = client.post(f"/jobs/{route}", data={"prompt": "loli"}, files=files)
            assert response.status_code == 400, route
        metrics = client.get("/metrics").text
        for mode in ("text", "img2img", "pose", "inpaint"):
            assert f'simpsons_jobs_total{{mode="{mode}",status="rejected"}}' in metrics, mode  # refus comptés
