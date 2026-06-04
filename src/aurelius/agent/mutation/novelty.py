"""NoveltyValidator — standalone novelty and diversity checking.

Extracted from MutationEngine to satisfy Single Responsibility Principle.
Handles all novelty-related checks: exact SMILES dedup, scaffold novelty,
commercial fingerprint novelty, trivial alkyl extensions, and the combined
novelty gate.

Delegates RDKit parsing to an injected ``get_ctx`` callable to maintain
single-point parsing via MoleculeContext.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.types import MoleculeContext

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_SEED_TRIVIAL_NUMS: set[int] = {7, 8, 9, 15, 16, 17, 35}

# SMARTS patterns for generic commercial electrolyte motifs that are NOT
# already caught by ``is_trivial_alkyl_extension`` (which handles simple
# alkyl homologation of known molecules). These patterns catch multi-group
# commercial motifs (e.g. glymes, sulfone-nitriles) where the heteroatom
# profile changes, escaping the simpler carbon-count-based check.
# Molecules matching these patterns without a novel Murcko scaffold are
# penalised to prevent rediscovery of known commercial electrolytes.
_COMMERCIAL_MOTIF_PATTERNS: list[tuple[Chem.Mol, str]] = []
for _smi, _desc in [
    ("[OX2][CX4][CX4][OX2]", "glyme_backbone"),
    ("[CX4][SX4](=O)(=O)[CX4]", "dialkyl_sulfone"),
    ("[CX4][C]#[N]", "alkyl_nitrile"),
]:
    _m = Chem.MolFromSmarts(_smi)
    if _m is not None:
        _COMMERCIAL_MOTIF_PATTERNS.append((_m, _desc))


def _heteroatom_profile(mol: Chem.Mol) -> dict[int, int]:
    profile: dict[int, int] = {}
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z in _SEED_TRIVIAL_NUMS:
            profile[z] = profile.get(z, 0) + 1
    return profile


def _count_carbons(mol: Chem.Mol) -> int:
    return sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


class NoveltyValidator:
    """Validates molecular novelty against seed and commercial reference sets.

    Three gates, applied in order of decreasing strictness:
      1. Exact SMILES match against seeds / known / generated
      2. Murcko scaffold novelty vs seed scaffolds
      3. ECFP4 Tanimoto similarity against commercial fingerprint database
    """

    def __init__(
        self,
        seed_smiles: set[str],
        seed_scaffolds: set[str],
        seed_fingerprints: list[Any],
        seed_pool: list[str],
        commercial_fps: list[Any],
        known_smiles: set[str],
        generated_smiles: set[str],
        get_ctx: Callable[[str], MoleculeContext | None],
    ) -> None:
        self._seed_smiles = seed_smiles
        self._seed_scaffolds = seed_scaffolds
        self._seed_fingerprints = seed_fingerprints
        self._seed_pool = seed_pool
        self._commercial_fps = commercial_fps
        self._known_smiles = known_smiles
        self._generated_smiles = generated_smiles
        self._get_ctx = get_ctx
        self._seed_trivial_cache: list[tuple[dict[int, int], int]] | None = None

    def is_known_smiles(self, canon: str) -> bool:
        return canon in self._seed_smiles or canon in self._known_smiles or canon in self._generated_smiles

    def is_novel_scaffold(self, ctx: MoleculeContext) -> bool:
        if MurckoScaffold is None or not self._seed_scaffolds:
            return True
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
            if scaffold and scaffold in self._seed_scaffolds:
                return False
        except Exception:
            pass
        return True

    def is_novel_vs_commercial(self, fp: Any, threshold: float = 0.85) -> bool:
        if not self._commercial_fps:
            return True
        sims = BulkTanimotoSimilarity(fp, self._commercial_fps)
        return not any(s >= threshold for s in sims)

    @staticmethod
    def _motif_hetero_count(mol: Chem.Mol, pat: Chem.Mol) -> int:
        """Count heteroatoms in substructure matches of a pattern."""
        return sum(
            1 for match in mol.GetSubstructMatches(pat)
            for idx in match
            if mol.GetAtomWithIdx(idx).GetAtomicNum() in {7, 8, 15, 16}
        )

    def is_commercial_motif(self, ctx: MoleculeContext) -> bool:
        """Check if molecule matches a generic commercial electrolyte motif.

        Returns True if the molecule matches any of the pre-compiled commercial
        motif patterns AND:
          - Does not possess a truly novel Murcko scaffold
          - Has NO additional heteroatoms beyond what the matched motif provides

        This prevents false-positives on hybrid molecules (e.g. sulfone-ethers)
        that contain a commercial substructure but also have other functional
        groups making them genuinely novel.
        """
        mol = ctx.mol
        for pat, _desc in _COMMERCIAL_MOTIF_PATTERNS:
            if not mol.HasSubstructMatch(pat):
                continue
            total_het = sum(
                a.GetAtomicNum() in {7, 8, 15, 16} for a in mol.GetAtoms()
            )
            if total_het > self._motif_hetero_count(mol, pat):
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if not scaffold or scaffold in self._seed_scaffolds:
                    return True
            except Exception:
                continue
        return False

    def is_trivial_alkyl_extension(self, ctx: MoleculeContext, min_extra_carbons: int = 2) -> bool:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
        if scaffold and scaffold not in self._seed_scaffolds:
            return False
        ctx_profile = _heteroatom_profile(ctx.mol)
        ctx_c = _count_carbons(ctx.mol)

        if self._seed_trivial_cache is None:
            self._seed_trivial_cache = []
            for seed_smi in self._seed_pool:
                seed_ctx = self._get_ctx(seed_smi)
                if seed_ctx is None:
                    continue
                self._seed_trivial_cache.append(
                    (_heteroatom_profile(seed_ctx.mol), _count_carbons(seed_ctx.mol))
                )

        for seed_profile, seed_c in self._seed_trivial_cache:
            if ctx_profile == seed_profile and ctx_c >= seed_c + min_extra_carbons:
                return True
        return False

    def novelty_check(self, ctx: MoleculeContext, check_scaffold: bool = True, force_exploration: bool = False) -> bool:
        try:
            canon = Chem.MolToSmiles(ctx.mol)
        except Exception:
            return False
        if self.is_known_smiles(canon):
            return False
        if check_scaffold and not self.is_novel_scaffold(ctx):
            return False
        if check_scaffold and self.is_trivial_alkyl_extension(ctx):
            return False
        if check_scaffold and self.is_commercial_motif(ctx):
            return False
        fp = ctx.get_ecfp4()
        threshold = 0.90 if force_exploration else 0.85
        return self.is_novel_vs_commercial(fp, threshold=threshold)
