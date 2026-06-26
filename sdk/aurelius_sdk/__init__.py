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

    def verify_accuracy(
        self,
        kernel_path: str,
        benchmark_subset: list[str],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Verify that the kernel's validation_metrics match re-evaluated results.

        The method sends each SMILES in *benchmark_subset* to the engine,
        collects the returned metrics, and compares them against the
        ``validation_metrics`` embedded in the kernel loaded from
        *kernel_path*.

        Parameters
        ----------
        kernel_path : str
            Path to a signed kernel JSON file on the local filesystem.
        benchmark_subset : list of str
            SMILES strings to re-screen for accuracy verification.
        timeout : float, optional
            Per-request timeout in seconds. Defaults to the client timeout.

        Returns
        -------
        dict
            A dictionary with keys ``"match"`` (bool), ``"expected_metrics"``,
            ``"actual_metrics"``, and ``"discrepancies"`` (list of strings).

        Example
        -------
        >>> client = Client(base_url="http://localhost:8000")
        >>> result = client.verify_accuracy("my_kernel.json", ["CCO", "CC=O"])
        >>> result["match"]
        True
        """
        import json as _json
        import hashlib as _hashlib

        # Load and verify kernel
        try:
            with open(kernel_path) as _f:
                kernel = _json.load(_f)
        except (OSError, _json.JSONDecodeError) as _exc:
            return {
                "match": False,
                "expected_metrics": None,
                "actual_metrics": None,
                "discrepancies": [f"Failed to load kernel: {_exc}"],
            }

        expected_metrics = kernel.get("validation_metrics", {})
        expected_hash = _hashlib.sha256(
            _json.dumps(expected_metrics, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # Re-evaluate benchmark subset
        actual_metrics: dict[str, Any] = {
            "spearman_rho": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "n_training": len(benchmark_subset),
        }
        discrepancies: list[str] = []

        for smi in benchmark_subset:
            try:
                result = self.screen(smi)
            except Exception as _exc:
                discrepancies.append(f"Screen failed for {smi}: {_exc}")
                continue

            score = result.get("score", {})
            total = score.get("total_score", 0.0)
            actual_metrics["mae"] = total
            actual_metrics["rmse"] = total
            actual_metrics["spearman_rho"] = score.get("is_viable", False)

        # Compare metrics
        for key in ("spearman_rho", "mae", "rmse"):
            exp = expected_metrics.get(key, 0.0)
            act = actual_metrics.get(key, 0.0)
            if abs(exp - act) > 1e-6:
                discrepancies.append(
                    f"Metric mismatch for '{key}': expected {exp}, got {act}; "
                    f"diff={act - exp:+.6f}"
                )

        actual_hash = _hashlib.sha256(
            _json.dumps(actual_metrics, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        match = expected_hash == actual_hash and len(discrepancies) == 0

        return {
            "match": match,
            "expected_metrics": expected_metrics,
            "actual_metrics": actual_metrics,
            "discrepancies": discrepancies,
        }

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


__all__ = ["Client"]
