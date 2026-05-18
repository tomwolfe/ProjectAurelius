"""Utility functions for Project Aurelius.

Re-exports shared utilities from submodules.
"""

from __future__ import annotations

from aurelius.utils.dependencies import (
    HAS_MLX,
    HAS_RDKIT,
    HAS_TORCH,
    DependencyManager,
    check_framework,
    report_status,
    routing_info,
)
from aurelius.utils.descriptors import _generate_molecular_descriptors, _hash_descriptors

__all__ = [
    "DependencyManager",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "_generate_molecular_descriptors",
    "_hash_descriptors",
    "check_framework",
    "report_status",
    "routing_info",
]
