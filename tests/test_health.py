from fastapi.testclient import TestClient

from app.main import app


def test_health_is_always_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_format():
    # simpsons_jobs_total n'apparaît qu'après le premier job (les métriques à
    # labels ne sont émises qu'une fois .labels(...) appelé) ; on vérifie donc
    # juste que l'endpoint répond au format texte Prometheus attendu.
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"# HELP" in response.content
