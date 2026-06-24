"""Tests for the Aurelius FastAPI server endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_pipeline() -> None:
    """Mock AureliusPipeline to skip heavy init in tests.

    Patches the class in the server module so the lifespan
    handler creates a mock instead of the real pipeline.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.screen_smiles.return_value = {
        "tier1": {"is_viable": True, "molecule_smiles": "CCO"},
        "tier2": {"homo_eV": -7.0, "lumo_eV": 0.5},
        "score": {"total_score": 75.0, "is_viable": True},
    }
    with patch("api_server.AureliusPipeline", return_value=mock_pipeline):
        yield


@pytest.fixture
def client() -> pytest.FixtureRequest:
    from fastapi.testclient import TestClient

    import api_server

    with TestClient(api_server.app) as c:
        yield c


def test_health_endpoint(client: pytest.FixtureRequest) -> None:
    """GET /health should return status ok and pipeline initialisation state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "pipeline_initialized" in data


def test_health_response_model(client: pytest.FixtureRequest) -> None:
    """Health response should match the expected schema."""
    response = client.get("/health")
    data = response.json()
    assert isinstance(data["pipeline_initialized"], bool)
    assert isinstance(data["version"], str)
    assert isinstance(data["status"], str)


def test_screen_valid_smiles_returns_200(client: pytest.FixtureRequest) -> None:
    """POST /screen with valid SMILES should return a 200 response."""
    response = client.post("/screen", json={"smiles": "CCO"})
    assert response.status_code == 200
    data = response.json()
    assert data["tier1"]["is_viable"] is True
    assert data["score"]["total_score"] == 75.0


def test_screen_missing_smiles_returns_422(client: pytest.FixtureRequest) -> None:
    """POST /screen without smiles field should return 422."""
    response = client.post("/screen", json={})
    assert response.status_code == 422


def test_screen_invalid_smiles_returns_400(client: pytest.FixtureRequest) -> None:
    """POST /screen with invalid SMILES should return a 400 error."""
    import api_server

    api_server._pipeline.screen_smiles.side_effect = ValueError("Invalid SMILES")
    response = client.post("/screen", json={"smiles": "invalid_smiles_123"})
    assert response.status_code == 400


def test_batch_empty_list_returns_422(client: pytest.FixtureRequest) -> None:
    """POST /batch with empty smiles list should return 422."""
    response = client.post("/batch", json={"smiles": []})
    assert response.status_code == 422


def test_batch_missing_smiles_returns_422(client: pytest.FixtureRequest) -> None:
    """POST /batch without smiles field should return 422."""
    response = client.post("/batch", json={})
    assert response.status_code == 422


def test_batch_valid_smiles_returns_200(client: pytest.FixtureRequest) -> None:
    """POST /batch with valid SMILES should return a list of results."""
    response = client.post("/batch", json={"smiles": ["CCO", "CC"]})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_batch_invalid_smiles_returns_results_with_errors(client: pytest.FixtureRequest) -> None:
    """POST /batch with invalid SMILES should return per-item errors."""
    import api_server

    api_server._pipeline.screen_smiles.side_effect = ValueError("Invalid SMILES")
    response = client.post("/batch", json={"smiles": ["invalid_smiles_123"]})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["is_viable"] is False
    assert "error" in data[0]
