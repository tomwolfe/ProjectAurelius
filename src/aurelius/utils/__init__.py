"""Utility functions for Project Aurelius.

Re-exports shared utilities from submodules.
"""

from __future__ import annotations

from aurelius.utils.chem_utils import (
    HAS_RDKIT,
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
    _serialize_fp,
    _tanimoto,
)
from aurelius.utils.dependencies import (
    HAS_MLX,
    HAS_TORCH,
    check_framework,
    report_status,
    routing_info,
)
from aurelius.utils.descriptors import generate_molecular_descriptors

__all__ = [
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "_deserialize_fp",
    "generate_molecular_descriptors",
    "_is_valid_mol",
    "_mol_to_fp",
    "_safe_mol_from_smiles",
    "_serialize_fp",
    "_tanimoto",
    "check_framework",
    "report_status",
    "routing_info",
]
