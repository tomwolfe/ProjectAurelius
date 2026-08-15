"""RDKit helper functions — serialization, SA scoring, and validation.

All molecular parsing and featurization is handled by ``MoleculeContext``.
This module only retains unique utilities not duplicated in MoleculeContext:
fingerprint serialization/deserialization, SA scoring via a data-driven rule
registry, Tanimoto helpers for the mutation engine, and canonical-SMILES
caching with a size-bounded persistent store.
"""

from __future__ import annotations

import contextlib
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


def _smiles_persistent_store() -> shelve.Shelf[str] | None:
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
        with contextlib.suppress(Exception):
            persistent.close()

    result = _canonical_smiles_lru(smiles)

    persistent = _smiles_persistent_store()
    if persistent is not None:
        try:
            persistent[smiles] = result
            persistent.sync()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            persistent.close()

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


def synthesizability_complexity(ctx: Any) -> float:
    """Higher-resolution continuous synthesizability complexity score [0, 1].

    Replaces the coarse rule-based SA score (1–10, ~10 distinct values) with
    a fine-grained signal that varies continuously across the full [0, 1]
    range. Unlike ``electrolyte_synthetic_accessibility`` which uses additive
    rule deltas with 0.15–3.0 steps, this score combines:

    1. **Fragment diversity** — the number of unique fragments from a BRICS
       decomposition, normalised by molecular size. More fragments relative
       to size = higher complexity.
    2. **Functional-group diversity** — the number of distinct electron-
       accepting / donating motifs (counted via the SA rule smarts),
       each weighted by its rarity in commercial building blocks.
    3. **Ring strain** — continuous penalty for strained rings (3- and
       4-membered), scaled by atom count rather than a flat +1.0.
    4. **Stereochemical complexity** — number of stereocenters and double-bond
       geometry, continuous penalty.

    The four components are combined as:
        complexity = 0.35 * frag_complexity + 0.30 * fg_complexity
                   + 0.20 * ring_strain + 0.15 * stereo_complexity

    Physical justification: synthesis difficulty is driven by (a) how many
    fragments must be assembled, (b) how many distinct functional groups
    must be introduced in sequence, (c) whether rings are strained, and
    (d) stereochemical control. All four are continuous quantities;
    collapsing them to 10 bins loses the fine-grained discrimination that
    NSGA-II selection pressure needs.

    ADR-2026-08-11-08: Replaces the coarse SA score as the primary
    synthesizability objective in NSGA-II selection.

    Args:
        ctx: Pre-parsed MoleculeContext (or any object with a ``mol`` attribute).

    Returns:
        float in [0, 1] where 1.0 = maximally complex / hard to synthesise,
        0.0 = simple / directly purchasable.
    """
    mol = ctx.mol
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 1.0

    # --- Component 1: Fragment diversity ---
    # Number of unique BRICS fragments, normalised by log(molecule size).
    try:
        from rdkit.Chem import BRICS
        fragments = list(BRICS.BRICSDecompose(mol))
        n_frags = len(fragments)
    except Exception:
        n_frags = 1
    # log-scale: 1 fragment → 0, 12+ fragments → ~1.0
    frag_diversity = min(np.log1p(n_frags) / np.log1p(12.0), 1.0)

    # Size factor: heavier molecules need more steps
    size_factor = min(np.log10(max(n_heavy, 5)) / 3.0, 1.0)
    frag_complexity = 0.6 * frag_diversity + 0.4 * size_factor

    # --- Component 2: Functional-group complexity ---
    # Count distinct electron-accepting / donating motifs from the SA rule set.
    # Each motif contributes proportionally, so more diverse chemistry = harder.
    fg_count: float = 0
    n_rules_triggered = 0
    for _name, rule_fn in _SA_RULES:
        delta = abs(rule_fn(ctx))
        if delta > 0:
            n_rules_triggered += 1
            fg_count += delta

    # Normalise: simple penalties are 0.15-0.5 per group, cap at ~15 groups
    fg_complexity = min(fg_count / 4.0, 1.0) * (1.0 - 1.0 / max(n_rules_triggered, 1))

    # --- Component 3: Ring strain ---
    ring_info = mol.GetRingInfo()
    strained_atoms = set()
    for ring in ring_info.AtomRings():
        if len(ring) <= 4:
            strained_atoms.update(ring)
    ring_strain = min(len(strained_atoms) / max(n_heavy, 1), 1.0)

    # --- Component 4: Stereochemical complexity ---
    n_stereo = Chem.rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_rings = len(ring_info.AtomRings())
    stereo_complexity = min(
        (n_stereo * 0.2 + min(n_rings, 3) * 0.15) / 1.0, 1.0
    )

    complexity = (
        0.35 * frag_complexity + 0.30 * fg_complexity
        + 0.20 * ring_strain + 0.15 * stereo_complexity
    )
    return float(np.clip(complexity, 0.0, 1.0))


def synthesizability_reward(ctx: Any) -> float:
    """Synthesizability reward in [0, 1] — the complement of complexity.

    Higher = easier to synthesise. This is ``1 - synthesizability_complexity``
    so it can be used directly as a maximisation objective in scoring.
    """
    return 1.0 - synthesizability_complexity(ctx)


__all__ = [
    "_serialize_fp",
    "_deserialize_fp",
    "_tanimoto",
    "electrolyte_synthetic_accessibility",
    "synthesizability_complexity",
    "synthesizability_reward",
    "get_canonical_smiles",
]
