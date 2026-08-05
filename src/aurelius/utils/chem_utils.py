"""RDKit helper functions — serialization, SA scoring, and validation.

All molecular parsing and featurization is handled by ``MoleculeContext``.
This module only retains unique utilities not duplicated in MoleculeContext:
fingerprint serialization/deserialization, SA scoring via a data-driven rule
registry, Tanimoto helpers for the mutation engine, and canonical-SMILES
caching with a size-bounded persistent store.
"""

from __future__ import annotations

import os
import shelve
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.constants import (
    ALDEHYDE_PATTERN as _ALDEHYDE_PATTERN,
)
from aurelius.constants import (
    ANHYDRIDE_PATTERN as _ANHYDRIDE_PATTERN,
)
from aurelius.constants import (
    CARBONATE_PATTERN as _CARBONATE_PATTERN,
)
from aurelius.constants import (
    EPOXIDE_PATTERN as _EPOXIDE_PATTERN,
)
from aurelius.constants import (
    ETHER_PATTERN as _ETHER_PATTERN,
)
from aurelius.constants import (
    NITRILE_PATTERN as _NITRILE_PATTERN,
)
from aurelius.constants import (
    PEROXIDE_PATTERN as _PEROXIDE_PATTERN,
)
from aurelius.constants import (
    SULFONE_SA_PATTERN as _SULFONE_SA_PATTERN,
)

_SMILES_CACHE_MAXSIZE = 4096
_SMILES_PERSISTENT_PATH = "smiles_cache.shelve"


@lru_cache(maxsize=_SMILES_CACHE_MAXSIZE)
def _canonical_smiles_lru(smiles: str) -> str:
    """Return canonical SMILES, cached in an LRU cache."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return ""
    return Chem.MolToSmiles(mol)


def _smiles_persistent_store() -> shelve.Shelf | None:
    """Open the persistent SMILES cache shelve, or return None on failure."""
    try:
        return shelve.open(_SMILES_PERSISTENT_PATH, writeback=False)
    except Exception:
        return None


def get_canonical_smiles(smiles: str) -> str:
    """Return the canonical SMILES for a given SMILES string.

    Checks the persistent cache first, then the in-memory LRU cache,
    and finally parses the SMILES fresh. Results are written back to
    both caches.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or empty string if parsing fails.
    """
    persistent = _smiles_persistent_store()
    if persistent is not None:
        try:
            cached = persistent.get(smiles)
            if cached is not None:
                persistent.close()
                return str(cached)
        except Exception:
            pass
        try:
            persistent.close()
        except Exception:
            pass

    result = _canonical_smiles_lru(smiles)

    persistent = _smiles_persistent_store()
    if persistent is not None:
        try:
            persistent[smiles] = result
            persistent.sync()
        except Exception:
            pass
        try:
            persistent.close()
        except Exception:
            pass

    return result


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


# ---------------------------------------------------------------------------
# Data-driven synthetic accessibility rule registry
# ---------------------------------------------------------------------------
# Each rule returns a score delta relative to a base of 3.0.
# Positive delta = harder to synthesise (penalty).
# Negative delta = easier to synthesise (reward).
# This replaces a 30-line sequential if/elif chain with a declarative,
# composable rule set.

_SA_RULES: list[tuple[str, Callable[[Any], float]]] = []


def _register_sa(fn: Callable[[Any], float]) -> Callable[[Any], float]:
    _SA_RULES.append((fn.__name__, fn))
    return fn


@_register_sa
def _ring_strain_penalty(ctx: Any) -> float:
    mol = ctx.mol
    ring_info = mol.GetRingInfo()
    score = 0.0
    for ring in ring_info.AtomRings():
        if len(ring) <= 4:
            score += 1.0
    return score


@_register_sa
def _stereocenter_penalty(ctx: Any) -> float:
    return 0.5 * Chem.rdMolDescriptors.CalcNumAtomStereoCenters(ctx.mol)


@_register_sa
def _peroxide_penalty(ctx: Any) -> float:
    if _PEROXIDE_PATTERN is not None and ctx.mol.HasSubstructMatch(_PEROXIDE_PATTERN):
        n = len(ctx.mol.GetSubstructMatches(_PEROXIDE_PATTERN))
        return 3.0 * min(n, 3)
    return 0.0


@_register_sa
def _aldehyde_penalty(ctx: Any) -> float:
    if _ALDEHYDE_PATTERN is not None and ctx.mol.HasSubstructMatch(_ALDEHYDE_PATTERN):
        return 1.0
    return 0.0


@_register_sa
def _anhydride_penalty(ctx: Any) -> float:
    if _ANHYDRIDE_PATTERN is not None and ctx.mol.HasSubstructMatch(_ANHYDRIDE_PATTERN):
        return 2.0
    return 0.0


@_register_sa
def _poly_peroxide_penalty(ctx: Any) -> float:
    """Additional penalty when ≥2 O-O bonds detected via graph traversal."""
    n = sum(
        1 for a in ctx.mol.GetAtoms()
        if a.GetAtomicNum() == 8 and a.GetDegree() == 2
        and any(n.GetAtomicNum() == 8 for n in a.GetNeighbors())
    )
    return 2.0 if n >= 2 else 0.0


@_register_sa
def _hypervalent_halogen_penalty(ctx: Any) -> float:
    for atom in ctx.mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in (9, 17, 35) and atom.GetExplicitValence() > 1:
            return 3.0
    return 0.0


@_register_sa
def _long_alkyl_chain_penalty(ctx: Any) -> float:
    n_c = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 6)
    if n_c > 12 and ctx.tpsa < 40:
        return min((n_c - 12) * 0.2, 2.0)
    return 0.0


@_register_sa
def _carbonate_reward(ctx: Any) -> float:
    if _CARBONATE_PATTERN is not None and ctx.mol.HasSubstructMatch(_CARBONATE_PATTERN):
        return -0.5
    return 0.0


@_register_sa
def _ether_reward(ctx: Any) -> float:
    if _ETHER_PATTERN is not None and ctx.mol.HasSubstructMatch(_ETHER_PATTERN):
        return -0.3
    return 0.0


@_register_sa
def _sulfone_reward(ctx: Any) -> float:
    if _SULFONE_SA_PATTERN is not None and ctx.mol.HasSubstructMatch(_SULFONE_SA_PATTERN):
        return -0.3
    return 0.0


@_register_sa
def _nitrile_reward(ctx: Any) -> float:
    if _NITRILE_PATTERN is not None and ctx.mol.HasSubstructMatch(_NITRILE_PATTERN):
        return -0.2
    return 0.0


@_register_sa
def _fluorine_reward(ctx: Any) -> float:
    f_count = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 9)
    return -0.15 * min(f_count, 4)


@_register_sa
def _epoxide_reward(ctx: Any) -> float:
    if _EPOXIDE_PATTERN is not None and ctx.mol.HasSubstructMatch(_EPOXIDE_PATTERN):
        return -0.5
    return 0.0


def electrolyte_synthetic_accessibility(ctx: Any) -> float:
    """Custom synthetic accessibility score for battery electrolytes.

    Returns a score from 1 (easy) to 10 (hard to synthesise).

    Unlike RDKit's ChEMBL-trained SA score, this function:
      - Rewards common electrolyte motifs (carbonates, sulfones,
        nitriles, fluorinated groups, ethers)
      - Penalises ring strain (3-4 membered rings), stereocenters,
        peroxides, and reactive functional groups
      - Reflects real industrial electrolyte synthesis difficulty

    Uses a data-driven rule registry (``_SA_RULES``) instead of a
    sequential if/elif chain, making individual rules auditable and
    independently testable.

    Args:
        ctx: Pre-parsed MoleculeContext.

    Returns:
        SA score in [1.0, 10.0] (lower = easier to synthesise).
    """
    score = 3.0
    for _name, rule_fn in _SA_RULES:
        score += rule_fn(ctx)
    return float(np.clip(score, 1.0, 10.0))


__all__ = [
    "_serialize_fp",
    "_deserialize_fp",
    "_tanimoto",
    "electrolyte_synthetic_accessibility",
    "get_canonical_smiles",
]
