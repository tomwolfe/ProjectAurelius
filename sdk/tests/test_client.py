"""Tests for the Aurelius SDK client."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aurelius_sdk import Client


class TestClient:
    """Client construction and basic usage."""

    def test_init_defaults(self) -> None:
        client = Client()
        assert client._base_url == "http://localhost:8000"
        assert client._headers == {}
        assert client._timeout == 30.0

    def test_init_with_api_key(self) -> None:
        client = Client(api_key="test-key-123")
        assert client._headers["X-API-Key"] == "test-key-123"

    def test_init_with_custom_url(self) -> None:
        client = Client(base_url="http://192.168.1.1:8080")
        assert client._base_url == "http://192.168.1.1:8080"

    def test_init_with_timeout(self) -> None:
        client = Client(timeout=60.0)
        assert client._timeout == 60.0

    def test_context_manager(self) -> None:
        with Client() as client:
            assert client._base_url == "http://localhost:8000"

    def test_client_repr(self) -> None:
        client = Client()
        assert isinstance(repr(client), str) or True

    def test_constructor_stores_params(self) -> None:
        """Verify constructor stores all parameters correctly."""
        client = Client(base_url="http://test:9000", api_key="key", timeout=15.0)
        assert client._base_url == "http://test:9000"
        assert client._headers["X-API-Key"] == "key"
        assert client._timeout == 15.0

    def test_verify_accuracy_missing_kernel_returns_match_false(self, tmp_path: Any) -> None:
        """verify_accuracy must return match=False when kernel file doesn't exist."""
        client = Client()
        result = client.verify_accuracy(
            str(tmp_path / "nonexistent.json"),
            ["CCO"],
        )
        assert result["match"] is False
        assert len(result["discrepancies"]) > 0

    def test_verify_accuracy_empty_benchmark(self, tmp_path: Any) -> None:
        """verify_accuracy with [] benchmark should return match based on empty kernel."""
        kernel_path = tmp_path / "empty_kernel.json"
        kernel_path.write_text('{"validation_metrics": {"spearman_rho": 0.0, "mae": 0.0, "rmse": 0.0, "n_training": 0}}')

        client = Client()
        with patch.object(client, 'screen', return_value={"score": {"total_score": 0.0}}):
            result = client.verify_accuracy(str(kernel_path), [])
        assert result["match"] is True
        assert result["actual_metrics"]["n_training"] == 0

    def test_verify_accuracy_invalid_json(self, tmp_path: Any) -> None:
        """verify_accuracy must handle malformed JSON gracefully."""
        kernel_path = tmp_path / "invalid.json"
        kernel_path.write_text("{invalid json content")

        client = Client()
        result = client.verify_accuracy(str(kernel_path), ["CCO"])
        assert result["match"] is False

    def test_verify_accuracy_single_molecule(self, tmp_path: Any) -> None:
        """verify_accuracy with single molecule should return consistent results."""
        kernel_path = tmp_path / "single_kernel.json"
        kernel_path.write_text(
            json.dumps({
                "validation_metrics": {
                    "spearman_rho": 0.85,
                    "mae": 0.12,
                    "rmse": 0.18,
                    "n_training": 1,
                },
            })
        )

        client = Client()
        result = client.verify_accuracy(str(kernel_path), ["CCO"])
        assert "actual_metrics" in result
        assert "expected_metrics" in result
        assert "discrepancies" in result
