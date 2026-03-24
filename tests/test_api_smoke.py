"""Smoke tests for FastAPI service."""

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    """Ensure health endpoint is reachable."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert isinstance(data.get("model_loaded"), bool)
