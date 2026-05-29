"""Solvation analysis package.

The MWSE engine was removed in v8.0 as part of the refactoring
to replace fake physics with real ML-based oracles.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str) -> None:
    """Provide a helpful error when deleted modules are imported."""
    raise ImportError(
        "The MWSE solvation engine was removed in v8.0. "
        "Use `from aurelius.scoring.oracle import PretrainedGNNOracle` "
        "for ML-based property evaluation instead."
    )


def __dir__() -> list[str]:
    return []
