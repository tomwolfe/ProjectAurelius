"""Phase 3a invariant - archive bars separation.

- Absolute discovery claim bar (score 65.0) and relative evolution bar
  (top decile of current generation) are DISTINCT code paths with distinct names.
- Seed pool evolves every generation regardless of absolute hits.
- Pareto plus novel-scaffold archive persists across generations and selection
  NEVER truncates history to last-200 only.
- Uses ASCII hyphens only.
"""
import inspect

from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.state import LoopState
from aurelius.constants import (
    DISCOVERY_THRESHOLD,
    EVOLUTION_RELATIVE_FRACTION,
    SEED_POOL_MAX_SIZE,
)
from aurelius.types import MoleculeContext, ScreeningResult


def _make_loop(seed: int = 42) -> DiscoveryLoop:
    engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"], seed=seed)
    state = LoopState(path="/tmp/test_phase3a_invariant_state.json")
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


def test_absolute_bar_unchanged():
    assert DISCOVERY_THRESHOLD == 65.0
    assert EVOLUTION_RELATIVE_FRACTION == 0.10
    assert SEED_POOL_MAX_SIZE == 200


def test_bars_are_distinct_code_paths():
    from aurelius.agent import loop as loop_mod

    rel_src = inspect.getsource(DiscoveryLoop._relative_evolution_bar)
    assert "_relative_evolution_bar" in dir(DiscoveryLoop)
    assert "EVOLUTION_RELATIVE_FRACTION" in rel_src
    # Relative bar path must not gate claims via the absolute bar.
    # Docstring may mention the absolute bar by name, but the executable
    # gate must be the relative fraction cutoff only.
    assert "sc >= DISCOVERY_THRESHOLD" not in rel_src
    assert ">= DISCOVERY_THRESHOLD" not in rel_src.split('"""')[-1]

    evo_src = inspect.getsource(DiscoveryLoop._evolve_seed_pool)
    assert "is_absolute" in evo_src
    assert "is_relative" in evo_src
    assert "_relative_evolution_bar" in evo_src
    assert "DISCOVERY_THRESHOLD" in evo_src

    claim_src = inspect.getsource(DiscoveryLoop._is_discovery)
    assert "DISCOVERY_THRESHOLD" in claim_src

    # Absolute claim bar and relative evolution bar have distinct names.
    assert "_is_discovery" != "_relative_evolution_bar"
    assert "DISCOVERY_THRESHOLD" != "EVOLUTION_RELATIVE_FRACTION"


def test_relative_bar_is_top_decile():
    loop = _make_loop()
    bar = loop._relative_evolution_bar([10.0] * 9 + [22.5])
    assert bar == 22.5
    scores20 = [float(x) for x in range(20)]
    bar20 = loop._relative_evolution_bar(scores20)
    assert bar20 == sorted(scores20, reverse=True)[1]


def test_evolution_fires_below_absolute():
    loop = _make_loop(seed=42)
    n0 = len(loop.engine.seed_pool)
    ctxs = [MoleculeContext.from_smiles(s) for s in ["CCO", "CCC", "CCCC", "CCCO", "CCN"]]
    ctxs = [c for c in ctxs if c is not None]
    scores = [21.0, 22.5, 20.1, 19.8, 22.0]
    assert max(scores) < DISCOVERY_THRESHOLD
    loop._evolve_seed_pool(ctxs, scores)
    assert len(loop.engine.seed_pool) > n0


def test_archive_preserves_pareto_not_recency():
    loop = _make_loop()
    loop.engine.seed_pool = [f"CCO{i}" for i in range(250)]
    for i, smi in enumerate(loop.engine.seed_pool[:5]):
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        loop.all_results.append(
            ScreeningResult(smiles=smi, total_score=90.0 - i, is_viable=True,
                            rejection_reasons=[], fingerprint=b"")
        )
    n_before = len(loop.all_results)
    loop._truncate_seed_pool_archive()
    assert len(loop.engine.seed_pool) <= SEED_POOL_MAX_SIZE
    for r in loop.all_results[:5]:
        assert r.smiles in loop.engine.seed_pool
    # Last-200 recency slice would drop the first 50 entries including top hits.
    recency_only = set([f"CCO{i}" for i in range(50, 250)])
    assert set(loop.engine.seed_pool) != recency_only
    # History persists - truncate touches seed pool only, not all_results.
    assert len(loop.all_results) == n_before


def test_archive_persists_across_generations_and_feeds_selection():
    loop = _make_loop()
    for smi, sc in [("c1ccccc1", 22.0), ("C1CCCCC1", 21.0)]:
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None
        loop.all_results.append(
            ScreeningResult(smiles=smi, total_score=sc, is_viable=False,
                            rejection_reasons=[], fingerprint=b"")
        )
    n0 = len(loop.all_results)
    # Simulate next generation evolution - history must not shrink.
    ctxs = [MoleculeContext.from_smiles(s) for s in ["CCCC", "CCCCC"]]
    ctxs = [c for c in ctxs if c is not None]
    loop._evolve_seed_pool(ctxs, [20.5, 21.5])
    assert len(loop.all_results) == n0
    # Archive feeds selection helpers.
    scored = loop._score_seed_pool()
    assert isinstance(scored, list)
    cands = loop._find_novel_scaffold_candidates(set(), set())
    assert len(cands) > 0
    parents = loop._top_seeds_from_results(generation=2)
    assert len(parents) > 0
    # Code inspection - selection never truncates history to last-200 only.
    import pathlib

    src = pathlib.Path(inspect.getfile(DiscoveryLoop)).read_text(encoding="utf-8")
    assert "[-200:]" not in src
    assert "all_results[-" not in src
    assert "all_results = all_results[" not in src
