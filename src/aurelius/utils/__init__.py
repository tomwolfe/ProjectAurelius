"""Utility functions for Project Aurelius.

Re-exports shared utilities from submodules.
"""

from __future__ import annotations

from aurelius.utils.chem import (
    HAS_RDKIT,
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
    _serialize_fp,
    _tanimoto,
)
from aurelius.utils.chem_utils import (
    HAS_RDKIT as HAS_RDKIT_UTILS,
)
from aurelius.utils.chem_utils import (
    _deserialize_fp as _deserialize_fp_utils,
)
from aurelius.utils.chem_utils import (
    _is_valid_mol as _is_valid_mol_utils,
)
from aurelius.utils.chem_utils import (
    _mol_to_fp as _mol_to_fp_utils,
)
from aurelius.utils.chem_utils import (
    _safe_mol_from_smiles as _safe_mol_from_smiles_utils,
)
from aurelius.utils.chem_utils import (
    _serialize_fp as _serialize_fp_utils,
)
from aurelius.utils.chem_utils import (
    _tanimoto as _tanimoto_utils,
)
from aurelius.utils.dependencies import (
    HAS_MLX,
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
    "HAS_RDKIT_UTILS",
    "HAS_TORCH",
    "_deserialize_fp",
    "_deserialize_fp_utils",
    "_generate_molecular_descriptors",
    "_hash_descriptors",
    "_is_valid_mol",
    "_is_valid_mol_utils",
    "_mol_to_fp",
    "_mol_to_fp_utils",
    "_safe_mol_from_smiles",
    "_safe_mol_from_smiles_utils",
    "_serialize_fp",
    "_serialize_fp_utils",
    "_tanimoto",
    "_tanimoto_utils",
    "check_framework",
    "report_status",
    "routing_info",
]
