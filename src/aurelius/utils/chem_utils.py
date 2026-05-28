"""RDKit helper functions for molecular manipulation and fingerprinting.

Provides shared utilities for:
- SMILES parsing and validation
- Fingerprint generation, serialization, and similarity computation
- Graceful degradation when RDKit is unavailable

All functions handle the case where RDKit is not installed by returning
safe defaults (None, False, or 0.0) rather than raising exceptions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aurelius.constants import FINGERPRINT_SIZE
from aurelius.utils.dependencies import HAS_RDKIT

# ---------------------------------------------------------------------------
# RDKit import guard
# ---------------------------------------------------------------------------
if HAS_RDKIT:
    from rdkit import Chem as _Chem  # type: ignore[import-not-found, unused-ignore]
    from rdkit.Chem import Descriptors as _Descriptors  # type: ignore[import-not-found, unused-ignore]
    from rdkit.DataStructs import (
        BitVectToText,  # type: ignore[import-not-found, unused-ignore]
        CreateFromBitString,  # type: ignore[import-not-found, unused-ignore]
        ExplicitBitVect,  # type: ignore[import-not-found, unused-ignore]
    )
    from rdkit.DataStructs import (
        FingerprintSimilarity as _FingerprintSimilarity,  # type: ignore[import-not-found, unused-ignore]
    )
else:
    _Chem = None  # type: ignore[assignment]
    _Descriptors = None  # type: ignore[assignment]
    BitVectToText = None  # type: ignore[assignment, misc]
    CreateFromBitString = None  # type: ignore[assignment, misc]
    ExplicitBitVect = None  # type: ignore[assignment, misc]
    _FingerprintSimilarity = None  # type: ignore[assignment, misc]


def _safe_mol_from_smiles(smiles: str) -> Any | None:
    """Return RDKit Mol or None.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Sanitized RDKit Mol object, or None if parsing fails.
    """
    if not HAS_RDKIT:
        return None
    try:
        mol = _Chem.MolFromSmiles(smiles)  # type: ignore[union-attr]
        if mol is not None:
            _Chem.SanitizeMol(mol)  # type: ignore[union-attr]
        return mol
    except Exception:
        return None


def _is_valid_mol(mol: Any) -> bool:
    """Check chemical validity and molecular weight < 450 Da.

    Args:
        mol: RDKit Mol object.

    Returns:
        True if molecule is valid and MW < 450.
    """
    try:
        _Chem.SanitizeMol(mol)
    except Exception:
        return False
    mw = _Descriptors.ExactMolWt(mol)  # type: ignore[union-attr, attr-defined]
    return bool(mw < 450.0)  # type: ignore[no-any-return]


def generate_ecfp4_fingerprint(smiles: str, n_bits: int = FINGERPRINT_SIZE) -> np.ndarray[Any, Any]:
    """Generate a 2048-bit ECFP4 (Morgan radius=2) fingerprint from SMILES.

    Uses RDKit's GetMorganFingerprintAsBitVect for production-grade
    fingerprints. Raises RuntimeError when RDKit is unavailable.

    Args:
        smiles: SMILES string of the molecule.
        n_bits: Fingerprint size (default 2048).

    Returns:
        numpy float32 array of shape (n_bits,) with values 0.0 or 1.0.

    Raises:
        RuntimeError: If RDKit is unavailable or SMILES is invalid.
    """
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AllChem

    mol = _Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"RDKit failed to parse SMILES '{smiles}'. Invalid molecule structure.",
        )
    fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)  # type: ignore[union-attr, attr-defined]
    bit_list = fp.ToList()
    arr = np.array(bit_list, dtype=np.float32)
    if len(arr) < n_bits:
        padded = np.zeros(n_bits, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded
    return arr[:n_bits]


def generate_molecular_descriptors(smiles: str) -> dict[str, float]:
    """Generate simple molecular descriptors from SMILES for Tier 0 prediction.

    Produces a minimal feature vector encoding structural properties
    relevant to SEI formation activation energies. When RDKit is
    available, uses real descriptors; raises RuntimeError when RDKit
    is unavailable.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dictionary of descriptor name -> value.

    Raises:
        RuntimeError: When RDKit is unavailable.
    """
    if not HAS_RDKIT:
        raise RuntimeError("RDKit is required for molecular descriptor generation. Install RDKit: pip install rdkit")

    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"RDKit failed to parse SMILES '{smiles}'. Invalid molecule structure.",
        )

    try:
        return {
            "mol_weight": float(_Descriptors.ExactMolWt(mol)),  # type: ignore[union-attr, attr-defined]
            "num_h_donors": int(_Descriptors.NumHDonors(mol)),  # type: ignore[union-attr, attr-defined]
            "num_h_acceptors": int(_Descriptors.NumHAcceptors(mol)),  # type: ignore[union-attr, attr-defined]
            "num_rotatable_bonds": int(_Descriptors.NumRotatableBonds(mol)),  # type: ignore[union-attr, attr-defined]
            "logp": float(_Descriptors.MolLogP(mol)),  # type: ignore[union-attr, attr-defined]
            "tpsa": float(_Descriptors.TPSA(mol)),  # type: ignore[union-attr, attr-defined]
        }
    except Exception as e:
        raise RuntimeError(f"Descriptor generation failed: {e}") from e


def _mol_to_fp(mol: Any) -> Any:
    """Compute ECFP4 (radius=2) fingerprint using Morgan generator.

    Args:
        mol: RDKit Mol object.

    Returns:
        Morgan fingerprint object (radius=2).
    """
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.GetHashedMorganFingerprint(mol, 2, 2048)  # type: ignore[no-any-return]


def _serialize_fp(fp: Any) -> str:
    """Serialize an RDKit fingerprint to a hex-like text string.

    Args:
        fp: RDKit fingerprint object.

    Returns:
        Serialized fingerprint string.
    """
    ev = ExplicitBitVect(2048)
    for idx in fp.GetNonzeroElements():
        ev.SetBit(idx)
    return BitVectToText(ev)  # type: ignore[no-any-return]


def _deserialize_fp(hex_str: str) -> Any:
    """Reconstruct an RDKit fingerprint from serialized text.

    Args:
        hex_str: Serialized fingerprint string.

    Returns:
        RDKit fingerprint object.
    """
    return CreateFromBitString(hex_str)


def _tanimoto(fp1: Any, fp2: Any) -> float:
    """Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.

    Returns:
        Tanimoto similarity coefficient in [0, 1].
    """
    if _FingerprintSimilarity is None:
        return 0.0
    # Convert UIntSparseIntVect to ExplicitBitVect for compatibility
    if not hasattr(fp1, "GetNumBits"):
        ev1 = ExplicitBitVect(2048)
        for idx in fp1.GetNonzeroElements():
            ev1.SetBit(idx)
        fp1 = ev1
    if not hasattr(fp2, "GetNumBits"):
        ev2 = ExplicitBitVect(2048)
        for idx in fp2.GetNonzeroElements():
            ev2.SetBit(idx)
        fp2 = ev2
    return float(_FingerprintSimilarity(fp1, fp2))  # type: ignore[no-any-return, no-untyped-call]


__all__ = [
    "HAS_RDKIT",
    "_safe_mol_from_smiles",
    "_is_valid_mol",
    "_mol_to_fp",
    "_serialize_fp",
    "_deserialize_fp",
    "_tanimoto",
]
