"""Data resources package for Project Aurelius.

This package provides fallback data files for training and screening
when external datasets are unavailable (e.g. no network, missing
HuggingFace access).

Usage:
    from importlib import resources
    esol = resources.files("aurelius.data").joinpath("esol_fallback.csv")
"""

from __future__ import annotations

from importlib import resources as _resources
from pathlib import Path


def _default_ff_path() -> str:
    """Return the file-system path to the force field params JSON.

    The path is resolved via importlib.resources so it works correctly
    even when the package is bundled in a zip or other non-standard
    location.
    """
    return str(
        _resources.files("aurelius.data").joinpath("force_field_params.json")
    )


from aurelius.data.params import (
    ForceFieldParams,
    _default_ff_path,
    _load_force_field_params,
)

__all__ = [
    "_default_ff_path",
    "_load_force_field_params",
    "ForceFieldParams",
]
