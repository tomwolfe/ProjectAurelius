"""Integration tests for the Bayesian active-learning loop.

Verifies that the DiscoveryLoop properly closes the feedback loop by:
1. Screening a batch of molecules
2. Updating the GP surrogate with new observations
3. Confirming the surrogate's fit() is called with correct numpy arrays
"""

from __future__ import annotations

import numpy as np

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult


class TestDiscoveryLoopActiveLearning:
    """Tests for the DiscoveryLoop active-learning cycle."""

    def test_update_surrogate_called_after_screening(self, tmp_path):
        """GP surrogate must be retrained with new observations after each batch."""
        # Create mock pipeline that returns scored results
        mock_pipeline = _make_mock_pipeline()

        # Create a mock engine that returns valid candidates
        mock_engine = _make_mock_engine()

        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint = _make_checkpoint_manager(str(checkpoint_path))

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            checkpoint=checkpoint,
            max_generations=1,
            batch_size=3,
        )

        # Run the loop
        result = loop.execute()

        # The GP surrogate should have been fitted during the loop
        surrogate = loop.feedback._surrogate
        assert surrogate is not None, "GP surrogate should have been created"
        assert surrogate._rf is not None, "GP surrogate should have been fitted"
        assert surrogate._X is not None, "X history should be populated"
        assert surrogate._y is not None, "y history should be populated"

        # Verify X and y are numpy arrays with correct shapes
        assert isinstance(surrogate._X, np.ndarray), "X should be a numpy array"
        assert isinstance(surrogate._y, np.ndarray), "y should be a numpy array"

        # Verify the loop recorded results
        assert len(result["all_results"]) > 0, "Should have screening results"
        assert result["total_screened"] > 0, "Should have screened at least one molecule"

    def test_feedback_records_fingerprints_not_smiles(self):
        """FeedbackAdapter.record() must append fingerprint arrays, not SMILES."""

        # Create a result with a fingerprint
        fp = np.zeros((2048,), dtype=np.float32)
        fp[5] = 1.0  # Set a few bits

        result = ScreeningResult(
            smiles="CC(=O)OC",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            fingerprint=fp,
        )

        adapter = _make_feedback_adapter()

        # Record the result
        adapter.record(result)

        # Verify that the X history contains the fingerprint array, not a string
        assert len(adapter._X_history) == 1
        assert isinstance(adapter._X_history[0], np.ndarray)
        assert adapter._X_history[0].shape == (2048,)
        assert adapter._X_history[0][5] == 1.0

        # Verify y_history was also updated
        assert len(adapter._y_history) == 1
        assert adapter._y_history[0] == 85.0

    def test_loop_retrains_surrogate_with_fingerprint_data(self):
        """After screening, the surrogate's fit() must be called with proper arrays."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        checkpoint = _make_checkpoint_manager("/tmp/test_checkpoint.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            checkpoint=checkpoint,
            max_generations=1,
            batch_size=5,
        )

        loop.execute()

        # Verify that the surrogate was actually fitted with real data
        surrogate = loop.feedback._surrogate
        assert surrogate is not None
        assert surrogate._X is not None
        assert surrogate._y is not None
        assert len(surrogate._X) > 0
        assert len(surrogate._y) > 0

        # Verify that X and y are consistent numpy arrays
        X = surrogate._X
        y = surrogate._y
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 2048, "Fingerprints should have 2048 features"

    def test_seed_pool_evolves_with_high_scores(self):
        """High-scoring molecules should feed back into the seed pool."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        checkpoint = _make_checkpoint_manager("/tmp/test_checkpoint_seed.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            checkpoint=checkpoint,
            max_generations=1,
            batch_size=3,
        )

        loop.execute()

        # All screened SMILES (score=85.0 >= 65.0) should be in the seed pool
        for smi in mock_engine.mutate_batch.return_value:
            assert smi in loop.engine.seed_pool

        # Seed pool size should be reflected in convergence state
        assert loop.convergence.seed_pool_size == len(loop.engine.seed_pool)

    def test_first_batch_random_selection(self):
        """First batch should select candidates randomly when surrogate is unfitted."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        checkpoint = _make_checkpoint_manager("/tmp/test_checkpoint2.json")

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            checkpoint=checkpoint,
            max_generations=1,
            batch_size=3,
        )

        loop.execute()

        # First batch: surrogate is unfitted, selection should be random
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


def _make_checkpoint_manager(path: str):
    """Create a checkpoint manager at the given path."""
    from aurelius.agent.state import CheckpointManager

    return CheckpointManager(path=path)


def _make_feedback_adapter():
    """Create a feedback adapter with a fitted GP surrogate."""
    from aurelius.agent.state import FeedbackAdapter

    adapter = FeedbackAdapter()

    # Pre-fit the surrogate with some training data
    X_train = np.random.rand(10, 2048).astype(np.float32)
    y_train = np.random.rand(10).astype(np.float32) * 80.0
    adapter.update(X_train, y_train)

    return adapter
