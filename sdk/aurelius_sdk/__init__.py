"""Aurelius SDK — Python client for the Project Aurelius discovery engine API.

Usage:
    from aurelius_sdk import Client

    client = Client(base_url="http://localhost:8000")
    result = client.screen("CCO")
    print(result["homo_eV"])
"""

from __future__ import annotations

from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class Client:
    """Client for the Aurelius engine REST API.

    Parameters
    ----------
    base_url : str
        Base URL of the Aurelius engine API server (default: ``http://localhost:8000``).
    api_key : str, optional
        API key for authenticated endpoints. Sent as ``X-API-Key`` header.
    timeout : float
        Request timeout in seconds (default: 30.0).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if httpx is None:
            raise ImportError(
                "The `httpx` library is required. Install with: pip install aurelius-sdk"
            )
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key is not None:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._base_url, headers=self._headers, timeout=self._timeout)

    def screen(self, smiles: str) -> dict[str, Any]:
        """Screen a single molecule and return its full evaluation result.

        Parameters
        ----------
        smiles : str
            Canonical SMILES string of the molecule.

        Returns
        -------
        dict
            Full evaluation result from the Aurelius pipeline.
        """
        response = self._client.post("/screen", json={"smiles": smiles})
        response.raise_for_status()
        return dict(response.json())

    def screen_batch(self, smiles_list: list[str]) -> list[dict[str, Any]]:
        """Screen multiple molecules in a single batch request.

        Parameters
        ----------
        smiles_list : list of str
            List of canonical SMILES strings.

        Returns
        -------
        list of dict
            Evaluation results in the same order as the input.
        """
        response = self._client.post("/batch", json={"smiles": smiles_list})
        response.raise_for_status()
        return [dict(r) for r in response.json()]

    def health(self) -> dict[str, Any]:
        """Check the health / status of the engine API server.

        Returns
        -------
        dict with keys 'status', 'version', and 'pipeline_initialized'.
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return dict(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


__all__ = ["Client"]
