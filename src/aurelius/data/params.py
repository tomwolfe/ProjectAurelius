"""Centralized force field parameters with thread-safe singleton loading.

This module provides a single, unified source for all force field parameters
used across the Aurelius pipeline, eliminating duplicated lazy-loading logic
from scoring and solvation engines.
"""

from __future__ import annotations

import json
import os
import threading
from importlib import resources
from typing import Any, ClassVar, cast


def _default_ff_path() -> str:
    """Return the file-system path to the force field params JSON.

    The path is resolved via importlib.resources so it works correctly
    even when the package is bundled in a zip or other non-standard
    location.
    """
    return str(
        resources.files("aurelius.data").joinpath("force_field_params.json")
    )


class ForceFieldParams:
    """Thread-safe singleton for lazy-loaded force field parameters.

    Parameters are loaded once from the bundled JSON resource and cached.
    All subsequent accesses return the cached data without re-reading the file.

    Usage:
        >>> params = ForceFieldParams.get()
        >>> scoring = params.get_scoring()
        >>> solvation = params.get_solvation()
    """

    _instance: ClassVar[ForceFieldParams | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _data: dict[str, Any] | None = None

    @classmethod
    def get(cls) -> ForceFieldParams:
        """Get the singleton instance, loading params on first access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = object.__new__(cls)
                    instance._load()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the loaded params cache (useful for testing)."""
        cls._instance = None
        cls._data = None

    def _load(self) -> None:
        """Load force field parameters from the packaged JSON resource."""
        self._data = _load_force_field_params()

    def get_scoring(self) -> dict[str, Any]:
        """Get scoring parameters section."""
        return self._data.get("scoring_parameters", {}) if self._data else {}

    def get_solvation(self) -> dict[str, Any]:
        """Get solvation parameters section."""
        return self._data.get("solvation", {}) if self._data else {}

    def get_lennard_jones(self) -> dict[str, Any]:
        """Get Lennard-Jones parameters section."""
        return self._data.get("lennard_jones", {}) if self._data else {}

    def get_partial_charges(self) -> dict[str, Any]:
        """Get partial charges section."""
        return self._data.get("partial_charges", {}) if self._data else {}

    def get_defaults(self) -> dict[str, Any]:
        """Get default values section."""
        return self._data.get("defaults", {}) if self._data else {}

    def get_all(self) -> dict[str, Any]:
        """Get all force field parameters."""
        return self._data or {}


def _load_force_field_params(path: str | None = None) -> dict[str, Any]:
    """Load force field parameters from JSON resource.

    Args:
        path: Optional path to force field params JSON file.

    Returns:
        Dictionary of force field parameters.
    """
    ff_path = path or _default_ff_path()
    if os.path.isfile(ff_path):
        try:
            with open(ff_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return {}
