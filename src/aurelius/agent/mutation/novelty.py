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

from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.types import MoleculeContext

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None

logger = logging.getLogger(__name__)

_SEED_TRIVIAL_NUMS: set[int] = {7, 8, 9, 15, 16, 17, 35}


def _heteroatom_profile(mol: Chem.Mol) -> dict[int, int]:
    profile: dict[int, int] = {}
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z in _SEED_TRIVIAL_NUMS:
            profile[z] = profile.get(z, 0) + 1
    return profile


def _count_carbons(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


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
        seed_fingerprints: list,
        seed_pool: list[str],
        commercial_fps: list,
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

    def is_novel_vs_commercial(self, fp: object, threshold: float = 0.85) -> bool:
        import numpy as np
        if not self._commercial_fps:
            return True
        sims = BulkTanimotoSimilarity(fp, self._commercial_fps)
        return not any(s >= threshold for s in sims)

    def is_trivial_alkyl_extension(self, ctx: MoleculeContext, min_extra_carbons: int = 2) -> bool:
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
        fp = ctx.get_ecfp4()
        threshold = 0.90 if force_exploration else 0.85
        if not self.is_novel_vs_commercial(fp, threshold=threshold):
            return False
        return True
