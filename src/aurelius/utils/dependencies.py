"""Centralized dependency detection for Project Aurelius.

Provides simple boolean availability flags for optional frameworks.
"""

from __future__ import annotations

HAS_RDKIT: bool = False

try:
    from rdkit import Chem  # noqa: F401

    HAS_RDKIT = True
except ImportError:
    pass
