"""Centralized dependency detection — fail fast if RDKit is missing.

Removed legacy silent-degradation path. RDKit is required.
"""

from __future__ import annotations

HAS_RDKIT: bool = False

try:
    from rdkit import Chem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    msg = (
        "RDKit is required for Project Aurelius. "
        "Install with: conda install -c conda-forge rdkit"
    )
    raise ImportError(msg) from None
