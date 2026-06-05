"""FragmentHarvester — dynamic BRICS fragment harvesting and pool management.

Extracted from MutationEngine to satisfy Single Responsibility Principle.
Manages a pool of harvested BRICS fragments from high-scoring molecules,
with eligibility filtering, deduplication, and fitness-based eviction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.agent.mutation.brics import _MAX_HARVESTED_FRAGMENTS
from aurelius.agent.mutation.smarts import ELECTROLYTE_FRAGMENT_POOL
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


class FragmentHarvester:
    """Manages a pool of harvested BRICS fragments from high-scoring molecules.

    Fragments are harvested via BRICS decomposition of discovered molecules,
    filtered for eligibility (MW, HBD, similarity), and stored with associated
    scores. The pool size is bounded; the lowest-scoring fragment is evicted
    when the cap is reached.
    """

    def __init__(
        self,
        get_ctx: Callable[[str], MoleculeContext | None],
    ) -> None:
        self._get_ctx = get_ctx
        self._harvested_fragments: list[str] = []
        self._harvested_fragment_set: set[str] = set()
        self._harvested_fragment_scores: dict[str, float] = {}

    def fragment_pool_size(self) -> int:
        return len(ELECTROLYTE_FRAGMENT_POOL) + len(self._harvested_fragments)

    def get_harvested_fragments(self) -> list[str]:
        return list(self._harvested_fragments)

    def get_harvested_score(self, frag_smi: str) -> float:
        return self._harvested_fragment_scores.get(frag_smi, 0.0)

    def _eligible_for_harvest(self, frag_smi: str) -> MoleculeContext | None:
        if frag_smi in self._harvested_fragment_set:
            return None
        if frag_smi in ELECTROLYTE_FRAGMENT_POOL:
            return None
        f_ctx = self._get_ctx(frag_smi)
        if f_ctx is None or f_ctx.mw > 250 or f_ctx.hbd > 0:
            return None
        if self._harvested_fragments and self._fragment_too_similar(frag_smi, self._harvested_fragments, threshold=0.85):
            return None
        return f_ctx

    def _fragment_too_similar(self, new_smi: str, existing_smis: list[str], threshold: float = 0.85) -> bool:
        new_ctx = self._get_ctx(new_smi)
        if new_ctx is None:
            return False
        new_fp = Chem.RDKFingerprint(new_ctx.mol)
        existing_fps = []
        for old_smi in existing_smis:
            old_ctx = self._get_ctx(old_smi)
            if old_ctx is None:
                continue
            existing_fps.append(Chem.RDKFingerprint(old_ctx.mol))
        if not existing_fps:
            return False
        sims = BulkTanimotoSimilarity(new_fp, existing_fps)
        return any(s >= threshold for s in sims)

    def harvest_fragments(self, smiles: str, score: float = 65.0) -> None:
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return
        try:
            for frag_smi in BRICS.BRICSDecompose(ctx.mol):
                if frag_smi in self._harvested_fragment_set:
                    if score > self._harvested_fragment_scores.get(frag_smi, 0):
                        self._harvested_fragment_scores[frag_smi] = score
                    continue
                if self._eligible_for_harvest(frag_smi) is None:
                    continue
                self._harvested_fragments.append(frag_smi)
                self._harvested_fragment_set.add(frag_smi)
                self._harvested_fragment_scores[frag_smi] = score
                if len(self._harvested_fragments) > _MAX_HARVESTED_FRAGMENTS:
                    worst = min(
                        self._harvested_fragments,
                        key=lambda f: self._harvested_fragment_scores.get(f, 0),
                    )
                    self._harvested_fragments.remove(worst)
                    self._harvested_fragment_set.discard(worst)
                    self._harvested_fragment_scores.pop(worst, None)
        except Exception:
            logger.debug("Failed to harvest fragments from %s", smiles, exc_info=True)
