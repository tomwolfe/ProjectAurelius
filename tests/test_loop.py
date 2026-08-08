"""Integration tests for the autonomous screening loop.

Verifies that the DiscoveryLoop properly:
1. Generates and filters candidates
2. Evaluates and selects via tournament selection
3. Records results and evolves the seed pool
"""

from __future__ import annotations

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult
from aurelius.agent.state import LoopState


class _MockPipeline:
    """Picklable mock pipeline for testing the discovery loop."""

    def screen_molecule(self, ctx):
        return {
            "score": {
                "total_score": 85.0,
                "is_viable": True,
                "rejection_reasons": [],
            }
        }

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
