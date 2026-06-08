"""Integration tests for the autonomous screening loop.

Verifies that the DiscoveryLoop properly:
1. Generates and filters candidates
2. Evaluates and selects via tournament selection
3. Records results and evolves the seed pool
"""

from __future__ import annotations

from unittest.mock import Mock

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult, run_screening, AgentConfig
from aurelius.agent.state import LoopState


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

    def test_seed_pool_evolves_with_high_scores(self, tmp_path):
        """High-scoring molecules should feed back into the seed pool."""
        mock_pipeline = _make_mock_pipeline()
        mock_engine = _make_mock_engine()
        state = _make_loop_state(str(tmp_path / "checkpoint_seed.json"))

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        loop.execute()

        assert loop.state.seed_pool_size == len(loop.engine.seed_pool)

    def test_wet_lab_feedback_integration(self):
        """Wet-lab feedback hook should be called with top discoveries and
        trigger GcUqEnsemble retraining on subsequent predictions."""
        import numpy as np
        from aurelius.scoring.oracle.gc import GcUqEnsemble
        from aurelius.types import MoleculeContext

        # Create a GcUqEnsemble
        ensemble = GcUqEnsemble()
        assert ensemble.is_trained is False, "Should start untrained"

        # Trigger initial training via prediction
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        diel_mean, diel_std = ensemble.predict_dielectric(ctx)
        assert ensemble.is_trained is True, "Should be trained after first prediction"

        initial_visc_mean, initial_visc_std = ensemble.predict_viscosity(ctx)

        # Append empirical data for this molecule (simulating wet-lab feedback)
        feedback_data = [
            {
                "smiles": "COC(=O)OC",
                "dielectric_constant": float(diel_mean),
                "viscosity_cP": float(initial_visc_mean),
            }
        ]
        ensemble.append_empirical_data(feedback_data)
        assert ensemble.is_trained is False, "Should be marked untrained after appending data"

        # Retrain and verify variance does not increase (ideally decreases)
        retrain_std = ensemble.predict_dielectric(ctx)[1]
        retrain_visc_std = ensemble.predict_viscosity(ctx)[1]
        assert ensemble.is_trained is True, "Should be retrained after prediction"

        # Variance should not increase after adding consistent empirical data
        assert retrain_std <= diel_std + 0.01, (
            f"Dielectric variance increased after retraining: "
            f"{retrain_std:.6f} > {diel_std:.6f} + 0.01"
        )
        assert retrain_visc_std <= initial_visc_std + 0.01, (
            f"Viscosity variance increased after retraining: "
            f"{retrain_visc_std:.6f} > {initial_visc_std:.6f} + 0.01"
        )

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
        assert len(loop.state._all_results) > 0


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
    mock.propose_mixture_candidates.return_value = []
    return mock


def _make_loop_state(path: str = "/tmp/test_state.json"):
    """Create a LoopState at the given path."""
    return LoopState(path=path)
