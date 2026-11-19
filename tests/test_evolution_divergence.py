"""Task 2: evolution fires below absolute threshold + multi-seed divergence.

- _evolve_seed_pool must grow the pool on low-score batches (~22) via the
  relative top-decile bar, not only on >= 65.0 discoveries.
- _find_novel_scaffold_candidates must return candidates below 65.0.
- Different seeds must explore different basins early (candidate sets diverge).
"""
from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.state import LoopState
from aurelius.constants import DISCOVERY_THRESHOLD
from aurelius.types import MoleculeContext


def _make_loop(seed: int) -> DiscoveryLoop:
    engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"], seed=seed)
    state = LoopState(path="/tmp/test_evo_state.json")
    state.clear()

    class _StubPipeline:
        def screen_batch(self, ctxs):
            out = []
            for c in ctxs:
                out.append({"score": {"total_score": 20.0, "is_viable": False, "rejection_reasons": []}, "tier2": {}})
            return out

        def screen_molecule(self, ctx):
            return self.screen_batch([ctx])[0]

    return DiscoveryLoop(
        pipeline=_StubPipeline(), engine=engine, state=state,
        max_generations=3, batch_size=8, seed=seed,
    )


def test_evolve_seed_pool_fires_below_threshold():
    loop = _make_loop(seed=42)
    n0 = len(loop.engine.seed_pool)
    ctxs = [MoleculeContext.from_smiles(s) for s in ["CCO", "CCC", "CCCC", "CCCO", "CCN"]]
    ctxs = [c for c in ctxs if c is not None]
    scores = [21.0, 22.5, 20.1, 19.8, 22.0]
    assert max(scores) < DISCOVERY_THRESHOLD
    loop._evolve_seed_pool(ctxs, scores)
    assert len(loop.engine.seed_pool) > n0, "relative bar must evolve pool below 65"


def test_relative_bar_is_top_decile():
    loop = _make_loop(seed=42)
    bar = loop._relative_evolution_bar([10.0] * 9 + [22.5])
    assert bar == 22.5  # top decile of 10 is top-1 (the max)
    bar2 = loop._relative_evolution_bar([22.5, 21.0, 20.0, 19.0])
    assert bar2 == 22.5  # top 10% of 4 is top-1
    # decile cutoff moves with distribution: 20 scores, top-2 gate
    scores20 = [float(x) for x in range(20)]
    bar20 = loop._relative_evolution_bar(scores20)
    assert bar20 == sorted(scores20, reverse=True)[1]


def test_novel_scaffold_candidates_below_threshold():
    loop = _make_loop(seed=42)
    # Seed all_results with low-score novel molecules
    from aurelius.types import ScreeningResult

    for smi, sc in [("c1ccccc1", 22.0), ("C1CCCCC1", 21.0), ("CC(=O)C", 20.0)]:
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None
        loop.all_results.append(
            ScreeningResult(smiles=smi, total_score=sc, is_viable=False,
                            rejection_reasons=[], fingerprint=ctx.get_feature_vector())
        )
    cands = loop._find_novel_scaffold_candidates(set(), set())
    assert len(cands) > 0, "novel scaffolds must be found below 65"


def test_truncate_preserves_top_not_recency():
    loop = _make_loop(seed=42)
    # Fill pool beyond cap with known order, score top at front
    from aurelius.types import ScreeningResult

    pool = ["C" * 1 for _ in range(5)]  # placeholder replaced below
    base = ["CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
    loop.engine.seed_pool = [f"CCO{i}" for i in range(250)]
    # Give first 5 high scores, rest zero
    for i, smi in enumerate(loop.engine.seed_pool[:5]):
        ctx = MoleculeContext.from_smiles("CCO")
        loop.all_results.append(
            ScreeningResult(smiles=smi, total_score=90.0 - i, is_viable=True,
                            rejection_reasons=[], fingerprint=b"")
        )
    loop._truncate_seed_pool_archive()
    assert len(loop.engine.seed_pool) <= 200
    # Top discoveries preserved (score-aware, not last-200)
    for r in loop.all_results[:5]:
        assert r.smiles in loop.engine.seed_pool


def test_multi_seed_divergence():
    from aurelius.pipeline import AureliusPipeline

    seen_sets = []
    for seed in (42, 123):
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1", "CS(=O)(=O)C"], seed=seed)
        state = LoopState(path="/tmp/test_div_state.json")
        state.clear()
        pipeline = AureliusPipeline()
        pipeline.initialize()
        loop = DiscoveryLoop(pipeline=pipeline, engine=engine, state=state,
                             max_generations=1, batch_size=12, seed=seed)
        cands = loop._generate_candidates(1)
        seen_sets.append(set(cands))
    a, b = seen_sets
    assert a != b, "different seeds must explore different basins early"
    jaccard = len(a & b) / max(len(a | b), 1)
    assert jaccard < 1.0
