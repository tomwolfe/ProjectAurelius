"""RDKit helper functions — serialization and validation only.

All molecular parsing and featurization is handled by ``MoleculeContext``.
This module only retains unique utilities not duplicated in MoleculeContext:
fingerprint serialization/deserialization and a limited set of helpers for
the mutation engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem

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


def electrolyte_synthetic_accessibility(ctx: Any) -> float:
    """Custom synthetic accessibility score for battery electrolytes.

    Returns a score from 1 (easy) to 10 (hard to synthesise).

    Unlike RDKit's ChEMBL-trained SA score, this function:
      - Rewards common electrolyte motifs (carbonates, sulfones,
        nitriles, fluorinated groups, ethers)
      - Penalises ring strain (3-4 membered rings), stereocenters,
        peroxides, and reactive functional groups
      - Reflects real industrial electrolyte synthesis difficulty

    Args:
        ctx: Pre-parsed MoleculeContext.

    Returns:
        SA score in [1.0, 10.0] (lower = easier to synthesise).
    """
    mol = ctx.mol

    # Base: electrolytes are generally simpler than drug-like molecules
    score = 3.0

    # --- Penalties (harder to synthesise) ---

    # Ring strain: 3- or 4-membered rings
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) <= 4:
            score += 1.0

    # Stereocenters increase synthetic difficulty
    n_stereo = Chem.rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    score += 0.5 * n_stereo

    # Peroxides: highly unstable, extremely hard to formulate
    peroxide = Chem.MolFromSmarts("[OX2][OX2]")
    if peroxide is not None and mol.HasSubstructMatch(peroxide):
        score += 3.0

    # Aldehydes: reactive, prone to oxidation
    aldehyde = Chem.MolFromSmarts("[CH](=O)")
    if aldehyde is not None and mol.HasSubstructMatch(aldehyde):
        score += 1.0

    # Acid chlorides, anhydrides — highly reactive
    anhydride = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])")
    if anhydride is not None and mol.HasSubstructMatch(anhydride):
        score += 2.0

    # --- Rewards (easier to synthesise — common electrolyte precursors) ---

    # Carbonates: workhorse electrolyte solvents
    carbonate = Chem.MolFromSmarts("O=C([OX2])[OX2]")
    if carbonate is not None and mol.HasSubstructMatch(carbonate):
        score -= 0.5

    # Ethers: common co-solvents
    ether = Chem.MolFromSmarts("[OD2]([CX4])[CX4]")
    if ether is not None and mol.HasSubstructMatch(ether):
        score -= 0.3

    # Sulfones: high-voltage electrolyte components
    sulfone = Chem.MolFromSmarts("S(=O)(=O)[CX4]")
    if sulfone is not None and mol.HasSubstructMatch(sulfone):
        score -= 0.3

    # Nitriles: common electrolyte additives
    nitrile = Chem.MolFromSmarts("[C]#[N]")
    if nitrile is not None and mol.HasSubstructMatch(nitrile):
        score -= 0.2

    # Fluorinated groups: ubiquitous in modern electrolytes
    f_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    score -= 0.15 * min(f_count, 4)

    # Epoxides: useful industrial precursors for ring-opening polymerisation
    epoxide = Chem.MolFromSmarts("[OX2]1[CX4][CX4]1")
    if epoxide is not None and mol.HasSubstructMatch(epoxide):
        score -= 0.5

    return float(np.clip(score, 1.0, 10.0))


__all__ = [
    "_serialize_fp",
    "_deserialize_fp",
    "_tanimoto",
    "electrolyte_synthetic_accessibility",
]
