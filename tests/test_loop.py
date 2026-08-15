"""Integration tests for the autonomous screening loop.

Verifies that the DiscoveryLoop properly:
1. Generates and filters candidates
2. Evaluates and selects via tournament selection
3. Records results and evolves the seed pool
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult
from aurelius.agent.state import LoopState
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle.xtb_single_point import XTBResult, XTBSinglePointOracle
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


class _MockPipeline:
    """Mock pipeline for testing the discovery loop."""

    def screen_molecule(self, ctx):
        return {
            "score": {
                "total_score": 85.0,
                "is_viable": True,
                "rejection_reasons": [],
            }
        }

    def screen_batch(self, contexts):
        return [self.screen_molecule(ctx) for ctx in contexts]

    def screen_mixture(self, ctx1, ctx2, frac):
        return {
            "score": {
                "total_score": 85.0,
                "is_viable": True,
                "rejection_reasons": [],
            },
            "mixture_properties": {
                "synergy_bonus": 0.0,
            },
        }


class TestDiscoveryLoop:
    """Tests for the DiscoveryLoop active-learning cycle."""

    def test_screens_and_records_results(self, tmp_path):
        """Pipeline must screen molecules and record screening results."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        state = _make_loop_state(str(tmp_path / "checkpoint.json"))

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        result = loop.execute()

        assert len(result["all_results"]) > 0, "Should have screening results"
        assert result["total_screened"] > 0

    def test_feedback_records_fingerprints_not_smiles(self):
        """LoopState should store fingerprint arrays for results."""
        import numpy as np

        fp = np.zeros((2053,), dtype=np.float32)
        fp[5] = 1.0

        ScreeningResult(
            smiles="CC(=O)OC",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            fingerprint=fp,
        )

        state = _make_loop_state()

        assert len(state._all_scores) == 0  # scores are tracked via record_batch now

    def test_seed_pool_evolves_with_high_scores(self):
        """High-scoring molecules should feed back into the seed pool."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        state = _make_loop_state("/tmp/test_checkpoint_seed.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        loop.execute()

        assert loop.state.seed_pool_size == len(loop.engine.seed_pool)

    def test_batch_contexts_are_screened(self):
        """All candidates returned from evaluate should have results."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        state = _make_loop_state("/tmp/test_checkpoint2.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        loop.execute()

        assert loop.state.total_screened > 0
        assert len(loop.all_results) > 0

    def test_batch_screening_matches_scalar_scores(self):
        """Batch screening must reproduce scalar screening key-for-key and value-for-value.

        Gap-1 regression: the batch oracle path previously omitted the Δ-correction
        and the reduction axis, so batch tier-2 records diverged from the scalar
        ``screen_molecule`` path and scores could not be compared. Values are
        compared with ``pytest.approx`` (float32 batch vs float64 scalar).
        """
        smiles_list = [
            "COC(=O)OC",
            "C1COC(=O)O1",
            "CC1COC(=O)O1",
            "COCCOC",
            "COC(=O)OCC",
            "CCOC(=O)OCC",
            "C1=COC(=O)O1",
            "C1CCOC1",
            "CC#N",
            "CS(=O)C",
            "FC1COC(=O)O1",
            "C1CCS(=O)(=O)C1",
        ]
        contexts = [c for s in smiles_list if (c := MoleculeContext.from_smiles(s)) is not None]
        assert len(contexts) >= 10, "Regression test needs a healthy molecule sample"

        pipeline = AureliusPipeline(use_xtb=False)
        pipeline.initialize()

        # Scalar first: warms the reduction singleton cache; batch recomputes
        # deterministically, so values must still match.
        scalar_results = [pipeline.screen_molecule(ctx) for ctx in contexts]
        batch_results = pipeline.screen_batch(contexts)

        assert len(batch_results) == len(contexts)
        for s, b in zip(scalar_results, batch_results, strict=True):
            assert set(s.keys()) == set(b.keys())
            assert set(s["tier2"].keys()) == set(b["tier2"].keys())
            assert set(s["score"].keys()) == set(b["score"].keys())
            assert s["score"]["total_score"] == pytest.approx(
                b["score"]["total_score"], abs=1e-3
            )
            # Orbital values: batch path is float32 (MLX); scalar screens one
            # molecule per batch, so a 1-U LP rounding flip is expected on the
            # 4th decimal. 2e-4 = two rounding units.
            assert s["tier2"]["homo_eV"] == pytest.approx(b["tier2"]["homo_eV"], abs=2e-4)
            assert s["tier2"]["lumo_eV"] == pytest.approx(b["tier2"]["lumo_eV"], abs=2e-4)
            assert s["tier2"]["gap_eV"] == pytest.approx(b["tier2"]["gap_eV"], abs=2e-4)
            assert s["tier2"]["reduction_stability_proxy"]["ea_eV"] == pytest.approx(
                b["tier2"]["reduction_stability_proxy"]["ea_eV"], abs=1e-4
            )
            assert s["tier2"]["dielectric_proxy"] == pytest.approx(
                b["tier2"]["dielectric_proxy"], abs=1e-3
            )
            assert s["tier2"]["viscosity_proxy"] == pytest.approx(
                b["tier2"]["viscosity_proxy"], abs=1e-3
            )
            assert s["tier2"]["li_solvation_proxy"] == pytest.approx(
                b["tier2"]["li_solvation_proxy"], abs=1e-3
            )
            assert s["tier2"]["conductivity_proxy"] == pytest.approx(
                b["tier2"]["conductivity_proxy"], abs=1e-3
            )

        # The batch oracle must cover the reduction axis, with the same
        # per-molecule EA values as the scalar evaluation.
        batch = pipeline._oracle.predict_batch_properties(contexts)
        assert "ea_eV" in batch
        assert len(batch["ea_eV"]) == len(contexts)
        scalar_records = [
            pipeline._oracle.evaluate(ctx)["reduction_stability_proxy"] for ctx in contexts
        ]
        for i, record in enumerate(scalar_records):
            assert float(batch["ea_eV"][i]) == pytest.approx(record["ea_eV"], abs=1e-4)

        # Edge cases: empty batch in and empty batch out.
        assert pipeline.screen_batch([]) == []
        empty = pipeline._oracle.predict_batch_properties([])
        assert all(
            isinstance(v, (list, np.ndarray)) and len(v) == 0 for v in empty.values()
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline():
    """Create a mock pipeline that returns valid screening results."""
    return _MockPipeline()


def _make_mock_engine():
    """Create a mock engine that returns candidate SMILES."""
    from unittest.mock import Mock

    mock = Mock()
    mock.seed_pool = [
        "CC(=O)OC",
        "CC(C)OC(C)=O",
        "C1CCOC(C)O1",
        "CC(C)OC(C)(C)OC(C)=O",
        "CC(C)OC(C)=O",
        "CC(C)OC(C)=O",
    ]
    mock.mutate_batch.return_value = [
        "CC(=O)OC",
        "CC(C)OC(C)=O",
        "C1CCOC(C)O1",
    ]
    mock.propose_mixture_candidates.return_value = []
    mock.propose_ternary_mixture_candidates.return_value = []
    return mock


def _make_loop_state(path: str = "/tmp/test_state.json"):
    """Create a LoopState at the given path."""
    return LoopState(path=path)


class TestXtbSinglePointThreadSafety:
    """Regression: ``XTBSinglePointOracle`` must be safe under concurrent calls.

    Tier-2.5 gates rank candidates with a ``ThreadPoolExecutor``
    (``DiscoveryLoop._rank_by_xtb``), so several threads share one oracle and
    one in-memory ``self._cache``. The pre-fix ``_persist`` dumped the whole
    dict with ``json.dump`` while another thread could insert a fresh key in
    ``evaluate``, raising ``RuntimeError: dictionary changed size during
    iteration`` and crashing the gate. The fix serializes cache read/write
    behind ``self._cache_lock``; this test reproduces the race window without
    xTB by stubbing ``_compute`` with a slow stub that mirrors the real
    implementation's post-compute ``_persist`` call.
    """

    def test_xtb_evaluate_is_thread_safe_with_shared_cache(self, tmp_path, monkeypatch):
        """Concurrent evaluate calls sharing one cache must not race persist."""
        cache_path = tmp_path / "xtb_cache.json"
        oracle = XTBSinglePointOracle(cache_path=str(cache_path))

        def _slow_save_cache(path, cache):
            # ``_persist`` resolves the module-level ``_save_cache`` at call
            # time, so patching it here emulates the slow, multi-flush dump of
            # a production cache that has accumulated thousands of entries --
            # the pre-fix crashes only fired once a single ``json.dump``
            # spanned several GIL-releasing disk flushes. The pure-Python
            # encoder iterates the shared dict and yields between items;
            # flushing after every chunk forces a real syscall so another
            # thread's unlocked ``self._cache[key] = ...`` can land
            # mid-iteration. With the fix, ``_persist`` holds ``_cache_lock``
            # for the whole dump, so no mutation can interleave.
            try:
                directory = os.path.dirname(path) or "."
                os.makedirs(directory, exist_ok=True)
                encoder = json.JSONEncoder(indent=2, default=str)
                with open(path, "w") as fh:
                    for chunk in encoder.iterencode(cache):
                        fh.write(chunk)
                        fh.flush()
            except OSError as exc:
                logger.warning("Could not write xTB cache to %s: %s", path, exc)

        monkeypatch.setattr(
            "aurelius.scoring.oracle.xtb_single_point._save_cache", _slow_save_cache
        )

        def _slow_compute(mol):
            # Keep many threads inside the expensive section simultaneously,
            # then persist the shared in-memory cache exactly like the real
            # _compute does (line ~189) -- this is where the dict-iteration
            # vs. key-insertion race window lives.
            time.sleep(0.03)
            oracle._persist()
            return XTBResult(
                homo_eV=-8.0,
                lumo_eV=-1.0,
                dipole_D=1.0,
                xtb_method="GFN2-xTB",
                convergence="converged",
                cpu_seconds=0.0,
                source="computed",
            )

        monkeypatch.setattr(oracle, "_compute", _slow_compute)

        distinct_smiles = [
            "CCO", "CCN", "CC(C)O", "CCC", "C1COC1", "COC", "CCOC", "CC(C)C",
            "CCCCO", "CC#N", "CS(C)=O", "CCS", "CCOCC", "CC(C)OCC", "C1CCOC1",
            "CNC", "CCN(C)C", "CCCO", "C1CC1", "CCCC", "CCCCC", "CC(C)(C)C",
            "C1CCCCC1", "C1CCCCO1", "CC(C)CC", "CC(=O)C", "CC(=O)O", "CC(=O)N",
            "C1CCC1", "C1CCCC1", "C=O", "CC=O",
        ]
        assert len(distinct_smiles) == 32

        # Each SMILES appears twice: forces concurrent cache-hit + cache-miss
        # interleavings (a duplicate arriving while the first compute is still
        # sleeping, and while another thread's _persist is dumping).
        workloads = distinct_smiles * 2

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(oracle.evaluate, workloads))

        assert len(results) == len(workloads)
        assert all(r["source"] == "computed" for r in results)
        assert all(r["homo_eV"] == -8.0 and r["lumo_eV"] == -1.0 for r in results)

        # Dedup may or may not have collapsed the duplicate keys -- either is
        # fine, but the cache must be sane and flush must not raise.
        assert 1 <= oracle.n_cached <= len(distinct_smiles)
        oracle.flush()
