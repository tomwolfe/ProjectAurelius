"""Centralized dependency detection and framework routing for Project Aurelius.

This module provides a single source of truth for framework availability
checks (MLX, PyTorch, RDKit, HuggingFace), version validation, and
structured fallback routing. All modules should import from this manager
rather than maintaining their own try/except ImportError blocks.

Design principles:
    - Single point of truth: HAS_MLX, HAS_TORCH, HAS_RDKIT are defined once.
    - Version-aware: validates minimum versions via importlib.metadata.
    - Logging-first: logs INFO/WARNING when fallbacks activate using
      logging.getLogger(__name__).
    - Backward-compatible: re-exports HAS_* booleans so existing code
      that checks `if HAS_MLX:` continues to work.

Usage:
    >>> from aurelius.utils.dependencies import DependencyManager
    >>> deps = DependencyManager()
    >>> deps.report_status()
    {'mlx': {'available': True, 'fallback_active': False, 'version': '0.21.0'}, ...}
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralized framework detection (single source of truth)
# ---------------------------------------------------------------------------

# MLX detection
try:
    import mlx.core as _mx  # noqa: F401

    _HAS_MLX = True
    _MLX_VERSION: str | None = None
    try:
        from importlib.metadata import version as _pkg_version

        _MLX_VERSION = _pkg_version("mlx")
    except Exception:
        _MLX_VERSION = "unknown"
except ImportError:
    _HAS_MLX = False
    _MLX_VERSION = None
    _mx = None  # type: ignore[assignment, unused-ignore]

# PyTorch detection
try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
    _TORCH_VERSION: str | None = None
    try:
        from importlib.metadata import version as _pkg_version

        _TORCH_VERSION = _pkg_version("torch")
    except Exception:
        _TORCH_VERSION = "unknown"
except ImportError:
    _HAS_TORCH = False
    _TORCH_VERSION = None
    _torch = None  # type: ignore[assignment, unused-ignore]

# RDKit detection
try:
    from rdkit import Chem as _rdkit_chem  # noqa: F401, N813

    _HAS_RDKIT = True
    _RDKIT_VERSION: str | None = None
    try:
        from importlib.metadata import version as _pkg_version

        _RDKIT_VERSION = _pkg_version("rdkit")
    except Exception:
        _RDKIT_VERSION = "unknown"
except ImportError:
    _HAS_RDKIT = False
    _RDKIT_VERSION = None
    _rdkit_chem = None  # type: ignore[assignment, unused-ignore]

# HuggingFace Hub detection
try:
    __import__("huggingface_hub")  # noqa: F401
    _HAS_HF_HUB = True
    _HF_HUB_VERSION: str | None = None
    try:
        from importlib.metadata import version as _pkg_version

        _HF_HUB_VERSION = _pkg_version("huggingface-hub")
    except Exception:
        _HF_HUB_VERSION = "unknown"
except ImportError:
    _HAS_HF_HUB = False
    _HF_HUB_VERSION = None

# Datasets detection
try:
    __import__("datasets")  # noqa: F401
    _HAS_DATASETS = True
    _DATASETS_VERSION: str | None = None
    try:
        from importlib.metadata import version as _pkg_version

        _DATASETS_VERSION = _pkg_version("datasets")
    except Exception:
        _DATASETS_VERSION = "unknown"
except ImportError:
    _HAS_DATASETS = False
    _DATASETS_VERSION = None

# Minimum version requirements
_MIN_VERSIONS: dict[str, str] = {
    "mlx": "0.15.0",
    "torch": "2.3.0",
    "rdkit": "2023.9.0",
    "huggingface-hub": "0.20.0",
    "datasets": "2.16.0",
}


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of integers.

    Handles versions like "0.21.0", "2024.3.2", "2.12.0.dev".

    Args:
        version_str: Version string to parse.

    Returns:
        Tuple of integers for version comparison, or (0,) on failure.
    """
    parts: list[int] = []
    for part in version_str.split("."):
        # Extract leading digits
        num_str = ""
        for ch in part:
            if ch.isdigit():
                num_str += ch
            else:
                break
        parts.append(int(num_str) if num_str else 0)
    return tuple(parts) if parts else (0,)


def _version_gte(version: str, minimum: str) -> bool:
    """Check if `version` is >= `minimum`."""
    return _parse_version(version) >= _parse_version(minimum)


# ---------------------------------------------------------------------------
# Public boolean exports (backward-compatible)
# ---------------------------------------------------------------------------

HAS_MLX: bool = _HAS_MLX
HAS_TORCH: bool = _HAS_TORCH
HAS_RDKIT: bool = _HAS_RDKIT
HAS_HF_HUB: bool = _HAS_HF_HUB
HAS_DATASETS: bool = _HAS_DATASETS

# ---------------------------------------------------------------------------
# DependencyManager class
# ---------------------------------------------------------------------------


class DependencyManager:
    """Centralized dependency detection and framework routing.

    Handles framework detection, validates minimum versions, provides
    structured fallback routing, and logs when fallbacks activate.

    This class is a singleton-like manager. Create one instance and
    share it, or call the module-level functions below.
    """

    def __init__(self) -> None:
        """Initialize the dependency manager.

        Detects all frameworks and validates versions. Logs warnings
        for missing or outdated dependencies.
        """
        self._status_cache: dict[str, dict[str, Any]] | None = None

    def check_framework(self, name: str) -> dict[str, Any]:
        """Check availability and version of a single framework.

        Args:
            name: Framework name (one of "mlx", "torch", "rdkit",
                "huggingface-hub", "datasets").

        Returns:
            Dict with keys: available (bool), version (str | None),
                meets_minimum (bool), min_version (str).
        """
        version_map: dict[str, str | None] = {
            "mlx": _MLX_VERSION,
            "torch": _TORCH_VERSION,
            "rdkit": _RDKIT_VERSION,
            "huggingface-hub": _HF_HUB_VERSION,
            "datasets": _DATASETS_VERSION,
        }
        availability_map: dict[str, bool] = {
            "mlx": _HAS_MLX,
            "torch": _HAS_TORCH,
            "rdkit": _HAS_RDKIT,
            "huggingface-hub": _HAS_HF_HUB,
            "datasets": _HAS_DATASETS,
        }

        available = availability_map.get(name, False)
        version = version_map.get(name)
        min_ver = _MIN_VERSIONS.get(name, "0.0.0")
        meets_minimum = False
        if available and version and version != "unknown":
            meets_minimum = _version_gte(version, min_ver)
            if not meets_minimum:
                logger.warning(
                    "Framework '%s' version %s is below minimum %s. Upgrade with: pip install --upgrade %s",
                    name,
                    version,
                    min_ver,
                    name,
                )

        return {
            "available": available,
            "version": version,
            "meets_minimum": meets_minimum,
            "min_version": min_ver,
        }

    def report_status(self) -> dict[str, dict[str, Any]]:
        """Report the status of all frameworks.

        Logs INFO for frameworks that are available and WARNING when
        fallbacks activate (i.e., a framework is missing).

        Returns:
            Dict of {framework: {available: bool, version: str | None,
                meets_minimum: bool, min_version: str}}.
        """
        if self._status_cache is not None:
            return self._status_cache

        status: dict[str, dict[str, Any]] = {}
        for name in _MIN_VERSIONS:
            info = self.check_framework(name)
            status[name] = {
                "available": info["available"],
                "version": info["version"],
                "meets_minimum": info["meets_minimum"],
                "min_version": info["min_version"],
            }
            if info["available"]:
                logger.info(
                    "Framework '%s' available (version %s, meets minimum %s).",
                    name,
                    info["version"] or "unknown",
                    info["min_version"],
                )
            else:
                logger.warning(
                    "Framework '%s' is NOT available. "
                    "Install with: pip install %s (or check optional dependency group).",
                    name,
                    name,
                )

        self._status_cache = status
        return status

    def is_mlx_available(self) -> bool:
        """Return True if MLX is available."""
        return _HAS_MLX

    def is_torch_available(self) -> bool:
        """Return True if PyTorch is available."""
        return _HAS_TORCH

    def is_rdkit_available(self) -> bool:
        """Return True if RDKit is available."""
        return _HAS_RDKIT

    def is_hf_hub_available(self) -> bool:
        """Return True if huggingface-hub is available."""
        return _HAS_HF_HUB

    def is_datasets_available(self) -> bool:
        """Return True if datasets is available."""
        return _HAS_DATASETS

    def routing_info(self) -> dict[str, str]:
        """Determine optimal execution path for each framework.

        Returns:
            Dict mapping framework names to their routing strategy:
                "native" = available and meets minimum
                "fallback" = available but below minimum
                "unavailable" = not installed
        """
        routing: dict[str, str] = {}
        for name in _MIN_VERSIONS:
            info = self.check_framework(name)
            if info["available"] and info["meets_minimum"]:
                routing[name] = "native"
            elif info["available"]:
                routing[name] = "fallback"
            else:
                routing[name] = "unavailable"
        return routing

    def clear_cache(self) -> None:
        """Clear the status cache (useful for testing)."""
        self._status_cache = None


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward-compatible)
# ---------------------------------------------------------------------------


def check_framework(name: str) -> dict[str, Any]:
    """Convenience function to check a single framework.

    Args:
        name: Framework name.

    Returns:
        Dict with availability and version info.
    """
    return DependencyManager().check_framework(name)


def report_status() -> dict[str, dict[str, Any]]:
    """Convenience function to report all framework statuses.

    Returns:
        Status dict for all frameworks.
    """
    return DependencyManager().report_status()


def routing_info() -> dict[str, str]:
    """Convenience function to get routing info for all frameworks.

    Returns:
        Dict mapping framework names to routing strategy.
    """
    return DependencyManager().routing_info()
