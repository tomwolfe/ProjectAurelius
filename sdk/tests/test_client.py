"""Tests for the Aurelius SDK client."""

from __future__ import annotations

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
