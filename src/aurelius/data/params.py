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

from pydantic import BaseModel, Field


def _default_ff_path() -> str:
    """Return the file-system path to the force field params JSON.

    The path is resolved via importlib.resources so it works correctly
    even when the package is bundled in a zip or other non-standard
    location.
    """
    return str(
        resources.files("aurelius.data").joinpath("force_field_params.json")
    )


class ForceFieldConfig(BaseModel):
    """Strict Pydantic model for force field parameters.

    Provides type-safe access to all force field parameters used across
    the Aurelius pipeline. Loaded once from the bundled JSON resource
    and cached for repeated access.

    Usage:
        >>> config = ForceFieldConfig.from_json_file()
        >>> scoring = config.get_scoring()
        >>> solvation = config.get_solvation()
    """

    class Config:
        """Pydantic model configuration."""
        extra = "ignore"

    _instance: ClassVar[ForceFieldConfig | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _data: dict[str, Any] | None = None
    _path: str | None = None

    @classmethod
    def from_json_file(cls, path: str | None = None) -> "ForceFieldConfig":
        """Load force field parameters from a JSON file.

        Args:
            path: Path to force field params JSON file. If None, uses
                the bundled default resource.

        Returns:
            A new ForceFieldConfig instance loaded from the file.
        """
        instance = object.__new__(cls)
        instance._path = path
        instance._load()
        return instance

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForceFieldConfig":
        """Create a ForceFieldConfig from a dict.

        Args:
            data: Dictionary of force field parameters.

        Returns:
            A new ForceFieldConfig instance.
        """
        instance = object.__new__(cls)
        instance._data = data
        return instance

    @classmethod
    def get(cls) -> "ForceFieldConfig":
        """Get the singleton instance, loading params on first access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = object.__new__(cls)
                    instance._path = None
                    instance._load()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the loaded params cache (useful for testing)."""
        cls._instance = None
        cls._data = None

    def _load(self) -> None:
        """Load force field parameters from the bundled JSON resource."""
        self._data = _load_force_field_params(self._path)

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


# Backward compatibility alias
ForceFieldParams = ForceFieldConfig


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
