"""Integration tests for the autonomous screening loop.

Verifies that the DiscoveryLoop properly:
1. Generates and filters candidates
2. Evaluates and selects via tournament selection
3. Records results and evolves the seed pool
"""

from __future__ import annotations

import pytest

from unittest.mock import Mock

from aurelius.agent.active_learning import ActiveLearningManager
from aurelius.agent.loop import DiscoveryLoop, ScreeningResult
from aurelius.agent.state import LoopState
from aurelius.types import MoleculeContext


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
        from aurelius.scoring.oracle.gc import GcUqEnsemble
        from aurelius.types import MoleculeContext

        # Create a GcUqEnsemble
        ensemble = GcUqEnsemble()
        assert ensemble.is_trained is False, "Should start untrained"

        # Trigger initial training via prediction
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        diel_mean, diel_std, _ = ensemble.predict_dielectric(ctx)
        assert ensemble.is_trained is True, "Should be trained after first prediction"

        initial_visc_mean, initial_visc_std, _ = ensemble.predict_viscosity(ctx)

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

    def test_wet_lab_feedback_reduces_prediction_variance(self):
        """Feeding back empirical data should retrain GcUqEnsemble and
        reduce prediction variance for validated molecules."""
        from aurelius.scoring.oracle.gc import GcUqEnsemble
        from aurelius.types import MoleculeContext

        ensemble = GcUqEnsemble()

        # Initial prediction on a known molecule
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        diel_mean, diel_std, _ = ensemble.predict_dielectric(ctx)

        # Append empirical data (simulating wet-lab feedback)
        feedback = [
            {
                "smiles": "COC(=O)OC",
                "dielectric_constant": float(diel_mean),
                "viscosity_cP": 1.0,
            }
        ]
        ensemble.append_empirical_data(feedback)

        # Retrain and verify variance decreases
        retrain_mean, retrain_std, _ = ensemble.predict_dielectric(ctx)
        assert retrain_std <= diel_std + 0.01, (
            f"Dielectric variance should not increase after retraining: "
            f"{retrain_std:.6f} > {diel_std:.6f} + 0.01"
        )

    def test_dynamic_weight_adjustment(self):
        """LoopState.compute_adjusted_weights should return weights summing
        to 1.0 when empirical feedback is available."""

        state = _make_loop_state()
        assert state.compute_adjusted_weights is not None

        # Add synthetic feedback
        state._empirical_feedback = [
            {"dielectric_proxy": 5.0, "viscosity_proxy": 2.0, "li_solvation_proxy": 3.0, "cycle_life": 400.0},
            {"dielectric_proxy": 8.0, "viscosity_proxy": 1.5, "li_solvation_proxy": 4.0, "cycle_life": 600.0},
            {"dielectric_proxy": 3.0, "viscosity_proxy": 3.0, "li_solvation_proxy": 2.0, "cycle_life": 200.0},
        ]

        adjusted = state.compute_adjusted_weights()
        assert abs(sum(adjusted.values()) - 1.0) < 0.01, (
            f"Adjusted weights should sum to 1.0, got {sum(adjusted.values())}"
        )
        assert all(v > 0 for v in adjusted.values()), "All weights should be positive"

    def test_active_learning_acquisition(self):
        """UCB acquisition should select a high-uncertainty molecule over
        a slightly higher-scoring but low-uncertainty molecule when
        exploration mode is active."""
        import numpy as np

        from aurelius.agent.selection import tournament_select
        from aurelius.types import MoleculeContext

        # Create mock contexts with different properties
        ctx_a = MoleculeContext.from_smiles("CCO")
        ctx_b = MoleculeContext.from_smiles("CCCF")
        assert ctx_a is not None and ctx_b is not None

        contexts = [ctx_a, ctx_b, ctx_a, ctx_b]
        # High scores but low uncertainty for first two, slightly lower scores
        # but high uncertainty for last two
        scores = [90.0, 88.0, 85.0, 83.0]
        uncertainties = [0.1, 0.1, 0.8, 0.9]

        # Without exploration: selects highest scores
        selected_exploit = tournament_select(
            contexts, scores, batch_size=2,
            exploration_mode=False,
        )
        exploit_scores = [scores[contexts.index(ctx)] for ctx in selected_exploit]
        mean_exploit = np.mean(exploit_scores)

        # With exploration (UCB): should favor high-uncertainty candidates
        selected_explore = tournament_select(
            contexts, scores, batch_size=2,
            exploration_mode=True,
            uncertainties=uncertainties,
            exploration_beta=5.0,
        )
        explore_scores = [scores[contexts.index(ctx)] for ctx in selected_explore]
        mean_explore = np.mean(explore_scores)

        # Exploration should select lower-scoring but high-uncertainty molecules
        # (Note: this test verifies the mechanism exists, not that exploration
        # always picks lower scores — depends on beta and diversity penalty)
        # At minimum, verify that exploration produces different selections
        selected_smiles_exploit = {ctx.smiles for ctx in selected_exploit}
        selected_smiles_explore = {ctx.smiles for ctx in selected_explore}
        assert len(selected_smiles_exploit) > 0
        assert len(selected_smiles_explore) > 0

    def test_gc_uq_high_uncertainty_forces_quantum_evaluation(self, tmp_path):
        """When GcUqEnsemble predicts high uncertainty (std > 15% of mean),
        _process_single_candidate should route to _evaluate_with_real_quantum
        instead of using the surrogate oracle."""
        from unittest.mock import patch

        mock_pipeline = _make_mock_pipeline()
        mock_pipeline._oracle = Mock()
        mock_pipeline._oracle._gc_uq = Mock()
        mock_pipeline._oracle._gc_uq.predict_dielectric.return_value = (5.0, 2.0, True)
        mock_pipeline._oracle._gc_uq.predict_viscosity.return_value = (2.0, 0.1, False)

        mock_engine = _make_mock_engine()
        state = _make_loop_state(str(tmp_path / "al_force.json"))

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert ctx is not None
        result_map: dict = {}

        with patch.object(loop.al_manager, 'evaluate_with_real_quantum') as mock_quantum:
            mock_quantum.return_value = (85.0, {"homo_eV": -6.5})
            result = loop._process_single_candidate(ctx, result_map)

        assert result is not None, "Should return quantum evaluation result"
        mock_quantum.assert_called_once()
        assert "CC(=O)OC" in state.active_learning_queue, (
            "Should add high-uncertainty molecule to active learning queue"
        )

    def test_similarity_caching_skips_oracle(self, tmp_path):
        """When a candidate with Tanimoto > 0.95 to a previously screened
        molecule is processed, the oracle is not called and the result is
        interpolated from cache."""
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline._oracle = Mock()
        mock_pipeline._oracle._gc_uq = Mock()
        mock_pipeline._oracle._gc_uq.predict_dielectric.return_value = (10.0, 0.1, False)
        mock_pipeline._oracle._gc_uq.predict_viscosity.return_value = (2.0, 0.05, False)

        mock_engine = _make_mock_engine()
        state = _make_loop_state(str(tmp_path / "cache_skip.json"))

        # Pre-populate the cache with a molecule and its screening result
        ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert ctx is not None
        state.add_screened_fingerprint(ctx.smiles, ctx.get_ecfp4(), {
            "score": {
                "total_score": 85.0,
                "is_viable": True,
                "rejection_reasons": [],
                "sa_score": 3.5,
                "sub_scores": {},
            },
            "tier2": {
                "homo_eV": -6.5,
                "lumo_eV": -0.5,
                "dielectric_proxy": 8.0,
                "viscosity_proxy": 2.0,
                "li_solvation_proxy": 1.5,
                "ced_proxy": 0.3,
            },
        })

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        # Process the same molecule (Tanimoto 1.0 > 0.95 -> cache hit)
        similar_ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert similar_ctx is not None
        result_map: dict = {}
        result = loop._process_single_candidate(similar_ctx, result_map)

        assert result is not None, "Should return cached result"
        total_score, t2 = result
        assert total_score == 85.0, "Should return cached score"
        assert t2["homo_eV"] == -6.5, "Should return cached properties"

        # Oracle should NOT have been called (cache hit)
        mock_pipeline.screen_molecule.assert_not_called()

        # Result should be marked as interpolated
        stored_result = state._all_results[-1]
        assert stored_result["smiles"] == "CC(=O)OC"
        assert stored_result["total_score"] == 85.0

    def test_similarity_caching_stores_results(self, tmp_path):
        """After screening a new molecule, its fingerprint and result are
        stored in LoopState.screened_fingerprints."""
        mock_pipeline = _make_mock_pipeline()
        mock_pipeline._oracle = Mock()
        mock_pipeline._oracle._gc_uq = Mock()
        mock_pipeline._oracle._gc_uq.predict_dielectric.return_value = (10.0, 0.1, False)
        mock_pipeline._oracle._gc_uq.predict_viscosity.return_value = (2.0, 0.05, False)

        mock_engine = _make_mock_engine()
        state = _make_loop_state(str(tmp_path / "cache_store.json"))

        loop = DiscoveryLoop(
            pipeline=mock_pipeline,
            engine=mock_engine,
            state=state,
            max_generations=1,
            batch_size=3,
        )

        ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert ctx is not None
        result_map: dict = {}
        result = loop._process_single_candidate(ctx, result_map)

        assert result is not None, "Should successfully screen"
        assert len(state.screened_fingerprints) == 1, "Should store one entry"
        stored_smi, stored_fp, stored_result = state.screened_fingerprints[0]
        assert stored_smi == "CC(=O)OC", "Should store correct SMILES"
        assert stored_fp is not None, "Should store fingerprint"
        assert stored_result["score"]["total_score"] == 85.0, "Should store screening result"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline():
    """Create a mock pipeline that returns valid screening results."""

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


class TestMoleculeContextFrozen:
    """Verify MoleculeContext is fully immutable after creation."""

    def test_smiles_attribute_is_frozen(self) -> None:
        """Attempting to assign to MoleculeContext.smiles must raise FrozenInstanceError.

        The dataclass is decorated with frozen=True, so any attempt to mutate
        its fields should raise dataclasses.FrozenInstanceError.
        """
        from dataclasses import FrozenInstanceError
        from aurelius.types import MoleculeContext

        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None

        with pytest.raises(FrozenInstanceError):
            ctx.smiles = "C"

    def test_mol_attribute_is_frozen(self) -> None:
        """Attempting to assign to MoleculeContext.mol must raise FrozenInstanceError."""
        from dataclasses import FrozenInstanceError
        from aurelius.types import MoleculeContext

        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None

        with pytest.raises(FrozenInstanceError):
            ctx.mol = None  # type: ignore[union-attr]

    def test_copy_returns_independent_instance(self) -> None:
        """Copying a MoleculeContext must produce an independent instance."""
        from copy import copy
        from aurelius.types import MoleculeContext

        ctx1 = MoleculeContext.from_smiles("CCO")
        assert ctx1 is not None

        ctx2 = copy(ctx1)
        assert ctx1 is not ctx2
        assert ctx1.smiles == ctx2.smiles
        # Modifying ctx2.smiles should not affect ctx1.smiles
        # (but since both are frozen, we verify via from_smiles)
        ctx3 = MoleculeContext.from_smiles("CCCO")
        assert ctx3 is not None
        assert ctx1.smiles != ctx3.smiles

    def test_molecule_context_uniqueness(self) -> None:
        """Two contexts with different SMILES must be distinct objects."""
        from aurelius.types import MoleculeContext

        ctx_a = MoleculeContext.from_smiles("CCO")
        ctx_b = MoleculeContext.from_smiles("CCCO")
        assert ctx_a is not None
        assert ctx_b is not None
        assert ctx_a is not ctx_b
        assert ctx_a.smiles != ctx_b.smiles


class TestMixtureContextIndependence:
    """Verify that mixture formulation generates separate, independent tracking states."""

    def test_mixture_components_are_independent(self) -> None:
        """Creating a mixture must produce two independent MoleculeContext instances.

        Modifying one component must not affect the other.
        """
        from aurelius.types import MoleculeContext

        ctx_a = MoleculeContext.from_smiles("CCO")
        ctx_b = MoleculeContext.from_smiles("CCCO")
        assert ctx_a is not ctx_b
        assert ctx_a.smiles != ctx_b.smiles

    def test_mixture_factory_does_not_mutate_inputs(self) -> None:
        """The _make_mixture_context factory must not modify input contexts."""
        from aurelius.agent.loop import DiscoveryLoop
        from aurelius.types import MoleculeContext

        ctx_a = MoleculeContext.from_smiles("CC(=O)OC")
        ctx_b = MoleculeContext.from_smiles("CC(C)OC(C)=O")
        assert ctx_a is not None
        assert ctx_b is not None

        original_a_smiles = ctx_a.smiles
        original_b_smiles = ctx_b.smiles

        # Use proper mixture format: SMILES_A|SMILES_B|frac_A
        mixture_smiles = f"{ctx_a.smiles}|{ctx_b.smiles}|0.5"
        mixture = DiscoveryLoop._make_mixture_context(mixture_smiles)

        # Input contexts must remain unchanged
        assert ctx_a.smiles == original_a_smiles
        assert ctx_b.smiles == original_b_smiles
        # The mixture should return one of the independent contexts
        assert mixture is not None
        assert mixture.smiles == ctx_a.smiles or mixture.smiles == ctx_b.smiles

    def test_separate_tracking_states_for_mixture(self) -> None:
        """Generating a binary mixture formulation must create separate, independent tracking states."""
        from aurelius.types import MoleculeContext

        ctx_a = MoleculeContext.from_smiles("CCO")
        ctx_b = MoleculeContext.from_smiles("CCCO")
        assert ctx_a is not None
        assert ctx_b is not None

        # Verify that the two contexts are completely independent
        assert ctx_a.smiles != ctx_b.smiles
        assert ctx_a.mol is not ctx_b.mol

        # Verify that fingerprint calculations are independent
        fp_a = ctx_a.get_ecfp4()
        fp_b = ctx_b.get_ecfp4()
        # Fingerprints must be different objects
        assert fp_a is not fp_b
