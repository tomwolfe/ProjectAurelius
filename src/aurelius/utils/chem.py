"""RDKit helper functions for molecular manipulation and fingerprinting.

Re-exports from chem_utils for backward compatibility.
"""

from __future__ import annotations

# Re-export all RDKit utilities from chem_utils for backward compatibility
from aurelius.utils.chem_utils import (  # noqa: F401
    HAS_RDKIT,
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
    _serialize_fp,
    _tanimoto,
)

__all__ = [
    "HAS_RDKIT",
    "_deserialize_fp",
    "_is_valid_mol",
    "_mol_to_fp",
    "_safe_mol_from_smiles",
    "_serialize_fp",
    "_tanimoto",
]
