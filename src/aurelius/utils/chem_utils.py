"""RDKit helper functions — serialization and validation only.

All molecular parsing and featurization is handled by ``MoleculeContext``.
This module only retains unique utilities not duplicated in MoleculeContext:
fingerprint serialization/deserialization and a limited set of helpers for
the mutation engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Fingerprint serialization / deserialization (used by MutationEngine)
# ---------------------------------------------------------------------------


def _serialize_fp(fp: Any) -> str:
    """Serialize an RDKit fingerprint to a hex-like text string.

    Args:
        fp: RDKit fingerprint object.

    Returns:
        Serialized fingerprint string.
    """
    from rdkit.DataStructs import BitVectToText, ExplicitBitVect

    ev = ExplicitBitVect(2048)
    for idx in fp.GetNonzeroElements():
        ev.SetBit(idx)
    return str(BitVectToText(ev))


def _deserialize_fp(hex_str: str) -> Any:
    """Reconstruct an RDKit fingerprint from serialized text.

    Args:
        hex_str: Serialized fingerprint string.

    Returns:
        RDKit fingerprint object.
    """
    from rdkit.DataStructs import CreateFromBitString

    return CreateFromBitString(hex_str)


def _tanimoto(fp1: Any, fp2: Any) -> float:
    """Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.

    Returns:
        Tanimoto similarity coefficient in [0, 1].
    """
    from rdkit.DataStructs import ExplicitBitVect, FingerprintSimilarity

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
    return float(FingerprintSimilarity(fp1, fp2))


__all__ = [
    "_serialize_fp",
    "_deserialize_fp",
    "_tanimoto",
]
