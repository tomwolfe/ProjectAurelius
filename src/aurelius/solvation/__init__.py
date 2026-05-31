"""Solvation analysis package (deprecated — functionality moved to oracle in v9.0)."""

from __future__ import annotations


def __getattr__(name: str) -> None:
    raise ImportError(
        "The MWSE solvation engine was removed in v9.0. "
        "Use `from aurelius.scoring.oracle import PropertyOracle` "
        "for ML-based property evaluation instead."
    )


def __dir__() -> list[str]:
    return []
