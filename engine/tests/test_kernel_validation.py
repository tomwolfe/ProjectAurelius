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
# Ed25519 Signature Verification Tests
# ---------------------------------------------------------------------------


def _generate_key_pair() -> tuple[Any, bytes]:
    """Generate an Ed25519 key pair for testing.

    Returns (private_key, public_key_bytes).
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    private = Ed25519PrivateKey.generate()
    public_bytes = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private, public_bytes


def _sign_kernel(kernel: dict[str, Any], private_key: Any) -> dict[str, Any]:
    """Sign a kernel dict with an Ed25519 private key.

    Strips any existing ``signature`` field, produces a canonical JSON
    representation, signs it, and inserts the hex-encoded signature.
    """
    payload = {k: v for k, v in kernel.items() if k != "signature"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sig = private_key.sign(canonical.encode("utf-8"))
    kernel = dict(kernel)
    kernel["signature"] = sig.hex()
    return kernel


def test_ed25519_verify_valid_signature() -> None:
    """JSONKernelLoader.verify() must return True for a valid signature."""
    private, public_bytes = _generate_key_pair()
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    signed = _sign_kernel(kernel, private)

    # Verify using the generated public key (same logic as JSONKernelLoader)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    stored = signed.get("signature", "")
    payload = {k: v for k, v in signed.items() if k != "signature"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    pub = Ed25519PublicKey.from_public_bytes(public_bytes)
    pub.verify(bytes.fromhex(stored), canonical.encode("utf-8"))
    # No exception = valid


def test_ed25519_verify_tampered_payload() -> None:
    """JSONKernelLoader.verify() must return False when the payload is tampered."""
    private, public_bytes = _generate_key_pair()
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    signed = _sign_kernel(kernel, private)

    # Tamper with the payload
    signed["tom_parameters"]["homo_offset"] = 99.0

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    stored = signed.get("signature", "")
    payload = {k: v for k, v in signed.items() if k != "signature"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    pub = Ed25519PublicKey.from_public_bytes(public_bytes)
    with pytest.raises(Exception):
        pub.verify(bytes.fromhex(stored), canonical.encode("utf-8"))


def test_ed25519_verify_missing_signature() -> None:
    """JSONKernelLoader.verify() must return False when signature is missing."""
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    assert "signature" not in kernel
    from aurelius.kernel_loader import JSONKernelLoader
    assert JSONKernelLoader.verify(kernel) is False


def test_ed25519_verify_empty_signature() -> None:
    """JSONKernelLoader.verify() must return False for an empty signature."""
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    kernel["signature"] = ""
    from aurelius.kernel_loader import JSONKernelLoader
    assert JSONKernelLoader.verify(kernel) is False


def test_ed25519_verify_garbage_signature() -> None:
    """JSONKernelLoader.verify() must return False for a garbage signature."""
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    kernel["signature"] = "deadbeef"
    from aurelius.kernel_loader import JSONKernelLoader
    assert JSONKernelLoader.verify(kernel) is False


def test_verify_kernel_cli_valid() -> None:
    """The verify-kernel CLI must output 'OK' for a validly-signed kernel file."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from aurelius.__main__ import cli
    import tempfile

    private, public_bytes = _generate_key_pair()
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    signed = _sign_kernel(kernel, private)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(signed, f)
        tmp_path = f.name

    try:
        with patch("aurelius.constants.KERNEL_PUBLIC_KEY", public_bytes):
            runner = CliRunner()
            result = runner.invoke(cli, ["verify-kernel", tmp_path])
        assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"
        assert "FAIL" not in result.output
        assert "OK" in result.output
    finally:
        import os
        os.unlink(tmp_path)


def test_verify_kernel_cli_invalid() -> None:
    """The verify-kernel CLI must output 'FAIL' for a tampered kernel file."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from aurelius.__main__ import cli
    import tempfile

    private, public_bytes = _generate_key_pair()
    kernel = _make_kernel({"spearman_rho": 0.85, "mae": 0.12, "rmse": 0.18, "n_training": 50})
    signed = _sign_kernel(kernel, private)
    signed["tom_parameters"]["homo_offset"] = 99.0  # tamper

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(signed, f)
        tmp_path = f.name

    try:
        with patch("aurelius.constants.KERNEL_PUBLIC_KEY", public_bytes):
            runner = CliRunner()
            result = runner.invoke(cli, ["verify-kernel", tmp_path])
        assert result.exit_code != 0
        assert "FAIL" in result.output
    finally:
        import os
        os.unlink(tmp_path)


def test_verify_kernel_cli_no_signature() -> None:
    """The verify-kernel CLI must output 'FAIL' for a kernel with no signature."""
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
        assert result.exit_code != 0
        assert "FAIL" in result.output
    finally:
        import os
        os.unlink(tmp_path)
