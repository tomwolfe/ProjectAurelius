"""Kernel loading and verification for signed Aurelius kernels.

Extracted from pipeline.py to improve modularity.  Provides the
``KernelLoader`` ABC, ``JSONKernelLoader`` with Ed25519 signature
verification, and the ``_load_demo_kernel`` helper that supports
dynamic downloading from GitHub Releases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["KernelLoader", "JSONKernelLoader", "compute_validation_hash"]

logger = logging.getLogger(__name__)

_DEMO_KERNEL_URL: str = (
    "https://github.com/anomalyco/ProjectAurelius/releases/latest/download/"
    "carbonate_high_voltage.json"
)

_DEMO_KERNEL_CANDIDATES: list[str] = []


class KernelLoader(ABC):
    """Abstract base for kernel parameter loaders.

    Implementations load a signed kernel from any source (local file,
    database, API) and return a dict of calibrated parameters.
    """

    @abstractmethod
    def load(self, path: str) -> dict[str, Any] | None:
        """Load a kernel from the given *path*.

        Returns a dict of kernel parameters (without the ``signature``
        field) on success, or ``None`` if loading/verification fails.
        """
        ...

    @abstractmethod
    def verify(self, kernel: dict[str, Any]) -> bool:
        """Verify the Ed25519 signature of *kernel*.

        Returns ``True`` if the signature is valid, ``False`` otherwise.
        """
        ...


class JSONKernelLoader(KernelLoader):
    """Load and verify a kernel from a signed JSON file.

    This is the default kernel loader used by the pipeline. It loads
    a ``.json`` file, verifies its Ed25519 signature, and returns the
    kernel parameters.
    """

    def load(self, path: str) -> dict[str, Any] | None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            from aurelius.constants import KERNEL_PUBLIC_KEY

            with open(path) as f:
                kernel = json.load(f)

            stored = kernel.get("signature", "")
            if not stored:
                logger.warning("Kernel %s: no signature field — using defaults.", path)
                return None

            if not self.verify(kernel):
                logger.warning(
                    "Kernel %s: signature verification failed — using defaults.",
                    path,
                )
                return None

            logger.info("Kernel %s: signature verified successfully.", path)
            return {k: v for k, v in kernel.items() if k != "signature"}
        except ImportError:
            logger.warning(
                "Kernel %s: cryptography not installed — cannot verify signature. Using defaults.",
                path,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Kernel %s: loading failed (%s) — using defaults.",
                path, exc,
            )
            return None

    @staticmethod
    def verify(kernel: dict[str, Any]) -> bool:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            from aurelius.constants import KERNEL_PUBLIC_KEY

            stored = kernel.get("signature", "")
            if not stored:
                return False

            payload = {k: v for k, v in kernel.items() if k != "signature"}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

            pub = Ed25519PublicKey.from_public_bytes(KERNEL_PUBLIC_KEY)
            pub.verify(bytes.fromhex(stored), canonical.encode("utf-8"))
            return True
        except Exception:
            return False


def _resolve_demo_kernel_paths() -> list[str]:
    """Return a list of candidate file-system paths for the demo kernel.

    Searches relative to the engine source tree and the home directory.
    """
    paths: list[str] = []
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(module_dir, "..", "..", "..", "docs", "examples", "kernels", "carbonate_high_voltage.json"),
        os.path.join(module_dir, "..", "..", "docs", "examples", "kernels", "carbonate_high_voltage.json"),
        os.path.join(module_dir, "..", "docs", "examples", "kernels", "carbonate_high_voltage.json"),
        os.path.join(os.path.expanduser("~"), ".aurelius", "kernels", "carbonate_high_voltage.json"),
    ]
    for c in candidates:
        resolved = os.path.abspath(c)
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _download_kernel(url: str, dest_path: str) -> bool:
    """Download a kernel JSON file from *url* to *dest_path*.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logger.info("Downloading demo kernel from %s …", url)
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Demo kernel downloaded to %s", dest_path)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        logger.debug("Failed to download demo kernel from %s: %s", url, exc)
        return False


def compute_validation_hash(kernel: dict[str, Any]) -> str:
    """Compute a SHA-256 hash of the validation_metrics embedded in *kernel*.

    The hash is computed over a canonical JSON serialisation of the
    ``validation_metrics`` value (sorted keys, compact separators) so
    that any change to the metrics — even a single floating-point digit
    — will produce a different hash.

    Parameters
    ----------
    kernel : dict
        A kernel dictionary that must contain a ``validation_metrics`` key.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest of the validation_metrics.

    Raises
    ------
    KeyError
        If *kernel* does not contain a ``validation_metrics`` key.
    """
    metrics = kernel.get("validation_metrics")
    if metrics is None:
        raise KeyError("'validation_metrics' is required in the kernel dict.")
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_demo_kernel() -> dict[str, Any] | None:
    """Load the pre-certified carbonate high-voltage demo kernel.

    Tries local file-system paths first, then attempts to download
    the latest certified kernel from GitHub Releases.  Falls back to
    ``None`` (use defaults) if all attempts fail.

    Returns kernel parameters dict or ``None`` if not found.
    """
    loader = JSONKernelLoader()

    for path in _resolve_demo_kernel_paths():
        if os.path.exists(path):
            result = loader.load(path)
            if result is not None:
                return result
            logger.warning("Demo kernel at %s failed verification — trying next.", path)

    dest_dir = os.path.join(os.path.expanduser("~"), ".aurelius", "kernels")
    dest_path = os.path.join(dest_dir, "carbonate_high_voltage.json")
    if _download_kernel(_DEMO_KERNEL_URL, dest_path):
        result = loader.load(dest_path)
        if result is not None:
            return result

    logger.warning("Demo kernel not found at any path and could not be downloaded.")
    return None
