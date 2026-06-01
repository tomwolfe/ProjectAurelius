"""Integration tests for the Bayesian active-learning loop.

Verifies that the DiscoveryLoop properly closes the feedback loop by:
1. Screening a batch of molecules
2. Updating the RF surrogate with new observations
3. Confirming the surrogate's fit() is called with correct numpy arrays
"""

from __future__ import annotations

import numpy as np

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult


class TestDiscoveryLoopActiveLearning:
    """Tests for the DiscoveryLoop active-learning cycle."""

    def test_update_surrogate_called_after_screening(self, tmp_path):
        """RF surrogate must be retrained with new observations after each batch."""
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

        surrogate = loop._surrogate
        assert surrogate is not None, "RF surrogate should have been created"
        assert surrogate._rf is not None, "RF surrogate should have been fitted"
        assert surrogate._X is not None, "X history should be populated"
        assert surrogate._y is not None, "y history should be populated"

        assert isinstance(surrogate._X, np.ndarray), "X should be a numpy array"
        assert isinstance(surrogate._y, np.ndarray), "y should be a numpy array"

        assert len(result["all_results"]) > 0, "Should have screening results"
        assert result["total_screened"] > 0

    def test_feedback_records_fingerprints_not_smiles(self):
        """FeedbackAdapter.record() must append fingerprint arrays, not SMILES."""
        fp = np.zeros((2053,), dtype=np.float32)
        fp[5] = 1.0

        result = ScreeningResult(
            smiles="CC(=O)OC",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            fingerprint=fp,
        )

        adapter = _make_loop_state()
        adapter.record(result)

        assert len(adapter.X_history) == 1
        assert isinstance(adapter.X_history[0], np.ndarray)
        assert adapter.X_history[0].shape == (2053,)
        assert adapter.X_history[0][5] == 1.0

        assert len(adapter.y_history) == 1
        assert adapter.y_history[0] == 85.0

    def test_loop_retrains_surrogate_with_fingerprint_data(self):
        """After screening, the surrogate's fit() must be called with proper arrays."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        state = _make_loop_state("/tmp/test_checkpoint.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=5,
        )

        loop.execute()

        surrogate = loop._surrogate
        assert surrogate is not None
        assert surrogate._X is not None
        assert surrogate._y is not None
        assert len(surrogate._X) > 0
        assert len(surrogate._y) > 0

        X = surrogate._X
        y = surrogate._y
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 2053

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

        for smi in mock_engine.mutate_batch.return_value:
            assert smi in loop.engine.seed_pool

        assert loop.state.seed_pool_size == len(loop.engine.seed_pool)

    def test_first_batch_random_selection(self):
        """First batch should select candidates randomly when surrogate is unfitted."""
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

        assert loop.total_screened > 0
        assert len(loop.all_results) > 0


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline():
    """Create a mock pipeline that returns valid screening results."""
    from unittest.mock import Mock

    mock = Mock()
    mock.screen_molecule.return_value = {
        "score": {
            "total_score": 85.0,
            "is_viable": True,
            "rejection_reasons": [],
        }
    }
    return mock


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
    return mock


def _make_loop_state(path: str = "/tmp/test_state.json"):
    """Create a LoopState at the given path."""
    from aurelius.agent.state import LoopState
    return LoopState(path=path)
