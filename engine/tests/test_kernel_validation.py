"""Unit tests for kernel validation hash computation and thread safety.

Verifies that:
- compute_validation_hash produces consistent SHA-256 hashes
- Different metrics produce different hashes
- Missing validation_metrics raises KeyError
- Hash computation is thread-safe under concurrent access
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

import pytest

from aurelius.kernel_loader import compute_validation_hash


def _make_kernel(metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a minimal kernel dict for testing."""
    kernel: dict[str, Any] = {
        "version": "1.0.0",
        "domain_boundary": {"domain": "electrolyte"},
        "tom_parameters": {
            "homo_offset": 0.1,
            "lumo_offset": 0.2,
            "gc_scale": 1.0,
            "uq_scale": 1.0,
        },
        "gc_fragments": ["ester", "carbonate"],
        "uq_weights": {"ensemble_weight": 0.5},
    }
    if metrics is not None:
        kernel["validation_metrics"] = metrics
    return kernel


def test_compute_validation_hash_valid_metrics() -> None:
    """compute_validation_hash should return a hex SHA-256 digest for valid metrics."""
    metrics = {
        "spearman_rho": 0.85,
        "mae": 0.12,
        "rmse": 0.18,
        "n_training": 50,
    }
    kernel = _make_kernel(metrics)
    digest = compute_validation_hash(kernel)

    # Verify it's a valid hex string of correct length
    assert len(digest) == 64
    int(digest, 16)  # Should not raise


def test_compute_validation_hash_deterministic() -> None:
    """Same metrics must produce the same hash."""
    metrics = {
        "spearman_rho": 0.85,
        "mae": 0.12,
        "rmse": 0.18,
        "n_training": 50,
    }
    kernel = _make_kernel(metrics)
    hash_a = compute_validation_hash(kernel)
    hash_b = compute_validation_hash(kernel)
    assert hash_a == hash_b


def test_compute_validation_hash_different_metrics_different_hash() -> None:
    """Different metrics must produce different hashes."""
    metrics_a = {
        "spearman_rho": 0.85,
        "mae": 0.12,
        "rmse": 0.18,
        "n_training": 50,
    }
    metrics_b = {
        "spearman_rho": 0.85,
        "mae": 0.13,  # Slightly different
        "rmse": 0.18,
        "n_training": 50,
    }
    kernel_a = _make_kernel(metrics_a)
    kernel_b = _make_kernel(metrics_b)
    hash_a = compute_validation_hash(kernel_a)
    hash_b = compute_validation_hash(kernel_b)
    assert hash_a != hash_b


def test_compute_validation_hash_missing_metrics_raises_keyerror() -> None:
    """compute_validation_hash must raise KeyError when validation_metrics is absent."""
    kernel = _make_kernel(metrics=None)
    with pytest.raises(KeyError, match="validation_metrics"):
        compute_validation_hash(kernel)


def test_compute_validation_hash_empty_metrics() -> None:
    """Empty metrics dict should still produce a valid hash."""
    metrics: dict[str, Any] = {}
    kernel = _make_kernel(metrics)
    digest = compute_validation_hash(kernel)
    assert len(digest) == 64
    int(digest, 16)


def test_compute_validation_hash_thread_safety() -> None:
    """Hash computation must be safe under concurrent access."""
    metrics = {
        "spearman_rho": 0.85,
        "mae": 0.12,
        "rmse": 0.18,
        "n_training": 50,
    }
    kernel = _make_kernel(metrics)

    results: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def _hash_computation() -> None:
        try:
            digest = compute_validation_hash(kernel)
            with lock:
                results.append(digest)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_hash_computation) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 20
    assert len(set(results)) == 1, "All threads should produce identical hashes"


def test_compute_validation_hash_with_negative_values() -> None:
    """Hash computation must handle negative metric values correctly."""
    metrics = {
        "spearman_rho": -0.3,
        "mae": 0.0,
        "rmse": 0.0,
        "n_training": 0,
    }
    kernel = _make_kernel(metrics)
    digest = compute_validation_hash(kernel)
    assert len(digest) == 64


def test_compute_validation_hash_with_nested_metrics() -> None:
    """Hash computation must handle nested metric structures."""
    metrics = {
        "spearman_rho": 0.85,
        "mae": 0.12,
        "rmse": 0.18,
        "n_training": 50,
        "per_molecule_mae": {
            "CCO": 0.05,
            "CC=O": 0.12,
        },
    }
    kernel = _make_kernel(metrics)
    digest = compute_validation_hash(kernel)
    assert len(digest) == 64


# ---------------------------------------------------------------------------
# Kernel Field Validation Tests
# ---------------------------------------------------------------------------


def test_verify_kernel_valid() -> None:
    """JSONKernelLoader.verify() must return True for a kernel with all required fields."""
    from aurelius.kernel_loader import JSONKernelLoader

    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    assert JSONKernelLoader.verify(kernel) is True


def test_verify_kernel_missing_required_field() -> None:
    """JSONKernelLoader.verify() must return False when a required field is absent."""
    from aurelius.kernel_loader import JSONKernelLoader

    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    del kernel["tom_parameters"]
    assert JSONKernelLoader.verify(kernel) is False


def test_verify_kernel_missing_uq_weights() -> None:
    """JSONKernelLoader.verify() must return False when uq_weights is absent."""
    from aurelius.kernel_loader import JSONKernelLoader

    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    del kernel["uq_weights"]
    assert JSONKernelLoader.verify(kernel) is False


def test_verify_kernel_cli_valid() -> None:
    """The verify-kernel CLI must output 'OK' for a valid kernel file."""
    from click.testing import CliRunner
    from aurelius.__main__ import cli
    import tempfile

    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(kernel, f)
        tmp_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["verify-kernel", tmp_path])
        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        assert "OK" in result.output
    finally:
        import os
        os.unlink(tmp_path)


def test_verify_kernel_cli_invalid() -> None:
    """The verify-kernel CLI must output 'FAIL' for a kernel missing required fields."""
    from click.testing import CliRunner
    from aurelius.__main__ import cli
    import tempfile

    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    del kernel["tom_parameters"]  # remove a required field

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(kernel, f)
        tmp_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["verify-kernel", tmp_path])
        assert result.exit_code != 0
        assert "FAIL" in result.output
    finally:
        import os
        os.unlink(tmp_path)
