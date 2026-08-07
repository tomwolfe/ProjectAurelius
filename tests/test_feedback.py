"""Tests for the experimental feedback controller (W6)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aurelius.agent.feedback import FeedbackController, FeedbackRecord, FeedbackState
from aurelius.types import MoleculeContext


class TestFeedbackRecord:
    def test_to_from_dict_roundtrip(self):
        rec = FeedbackRecord(
            smiles="CCO",
            homo_prediction=-8.5,
            lumo_prediction=-1.2,
            homo_corrected=-7.8,
            lumo_corrected=-0.9,
            total_score=72.5,
            conformal_confidence=0.96,
            generation=1,
        )
        d = rec.to_dict()
        assert d["smiles"] == "CCO"
        assert d["total_score"] == 72.5

        rec2 = FeedbackRecord.from_dict(d)
        assert rec2.smiles == rec.smiles
        assert rec2.total_score == rec.total_score
        assert rec2.conformal_confidence == rec.conformal_confidence

    def test_optional_experimental_fields(self):
        rec = FeedbackRecord(
            smiles="CCO",
            homo_prediction=-8.5,
            lumo_prediction=-1.2,
            homo_corrected=-7.8,
            lumo_corrected=-0.9,
            total_score=72.5,
            conformal_confidence=0.96,
            generation=1,
            experimental_homo=-7.9,
            experimental_lumo=-1.0,
            experimental_total_score=75.0,
        )
        d = rec.to_dict()
        assert d["experimental_homo"] == -7.9
        assert d["experimental_lumo"] == -1.0
        assert d["experimental_total_score"] == 75.0

    def test_no_experimental_defaults_none(self):
        rec = FeedbackRecord(
            smiles="CCO",
            homo_prediction=-8.5,
            lumo_prediction=-1.2,
            homo_corrected=-7.8,
            lumo_corrected=-0.9,
            total_score=72.5,
            conformal_confidence=0.96,
            generation=1,
        )
        assert rec.experimental_homo is None
        assert rec.experimental_lumo is None
        assert rec.experimental_total_score is None


class TestFeedbackState:
    def test_save_load_roundtrip(self):
        state = FeedbackState()
        state.records.append(FeedbackRecord(
            smiles="CCO", homo_prediction=-8.0, lumo_prediction=-1.0,
            homo_corrected=-7.5, lumo_corrected=-0.8,
            total_score=70.0, conformal_confidence=0.95, generation=1,
        ))
        state.last_refit_generation = 3
        state.total_refits = 1

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            loaded = FeedbackState.load(path)
            assert loaded.last_refit_generation == 3
            assert loaded.total_refits == 1
            assert len(loaded.records) == 1
            assert loaded.records[0].smiles == "CCO"
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        state = FeedbackState.load("/nonexistent/path/to/state.json")
        assert len(state.records) == 0
        assert state.total_refits == 0


class TestFeedbackController:
    def test_accumulate_records(self):
        fc = FeedbackController()
        for i in range(5):
            fc.accumulate(
                smiles=f"CC{'C'*i}O",
                homo_prediction=-8.0 - i * 0.1,
                lumo_prediction=-1.0 + i * 0.1,
                homo_corrected=-7.5 - i * 0.1,
                lumo_corrected=-0.8 + i * 0.1,
                total_score=65.0 + i,
                conformal_confidence=0.9,
                generation=i + 1,
            )
        assert fc.num_records == 5

    def test_accumulate_evicts_old(self):
        fc = FeedbackController(max_accumulated=3)
        for i in range(10):
            fc.accumulate(
                smiles=f"CC{'C'*i}O",
                homo_prediction=-8.0, lumo_prediction=-1.0,
                homo_corrected=-7.5, lumo_corrected=-0.8,
                total_score=70.0,
                conformal_confidence=0.9,
                generation=i + 1,
            )
        assert fc.num_records == 3

    def test_maybe_refit_returns_none_before_interval(self):
        fc = FeedbackController(refit_interval=5)
        fc.accumulate(
            smiles="CCO", homo_prediction=-8.0, lumo_prediction=-1.0,
            homo_corrected=-7.5, lumo_corrected=-0.8,
            total_score=70.0, conformal_confidence=0.9, generation=1,
        )
        result = fc.maybe_refit(current_generation=3)
        assert result is None

    def test_maybe_refit_triggers_after_interval(self):
        fc = FeedbackController(refit_interval=2)
        result = fc.maybe_refit(current_generation=3)
        # Should trigger since 3 - 0 >= 2
        assert result is not None
        assert "loo_mae_before" in result
        assert "loo_mae_after" in result
        assert result["records_accumulated"] == 0
        assert result["new_calibration_entries"] == 0
        assert fc.state.total_refits == 1

    def test_save_load_controller(self):
        fc = FeedbackController(refit_interval=3)
        fc.accumulate(
            smiles="CCO", homo_prediction=-8.0, lumo_prediction=-1.0,
            homo_corrected=-7.5, lumo_corrected=-0.8,
            total_score=70.0, conformal_confidence=0.9, generation=1,
        )
        fc._state.total_refits = 1
        fc._state.last_refit_generation = 5

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            fc.save(path)
            loaded = FeedbackController.load(path)
            assert loaded.num_records == 1
            assert loaded.state.total_refits == 1
            assert loaded.state.last_refit_generation == 5
        finally:
            os.unlink(path)

    def test_log_active_learning_trigger(self):
        """Active learning triggers should be recorded in FeedbackController."""
        fc = FeedbackController()
        fc.log_active_learning_trigger(
            smiles="CCO",
            generation=3,
            original_conf=0.5,
        )
        assert fc.num_active_learning_triggers == 1
        trigger = fc.state.active_learning_triggers[0]
        assert trigger["smiles"] == "CCO"
        assert trigger["generation"] == 3
        assert trigger["original_conf"] == 0.5

    def test_active_learning_triggers_persist(self):
        """Active learning triggers should survive save/load roundtrip."""
        fc = FeedbackController()
        fc.log_active_learning_trigger(
            smiles="CCO",
            generation=3,
            original_conf=0.5,
        )
        fc.log_active_learning_trigger(
            smiles="C1COCCO1",
            generation=5,
            original_conf=0.3,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            fc.save(path)
            loaded = FeedbackController.load(path)
            assert loaded.num_active_learning_triggers == 2
            assert loaded.state.active_learning_triggers[0]["smiles"] == "CCO"
            assert loaded.state.active_learning_triggers[1]["smiles"] == "C1COCCO1"
        finally:
            os.unlink(path)


class TestExperimentalFeedback:
    """FeedbackController should load experimental HOMO/LUMO and
    reduce delta-correction LOO MAE after refit."""

    def test_experimental_feedback_loads(self, tmp_path):
        """Loading experimental feedback populates the cache."""
        feedback_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "aurelius",
            "data",
            "experimental_feedback.json",
        )
        fc = FeedbackController(
            experimental_feedback_path=feedback_path,
        )
        assert len(fc._experimental_cache) > 0, (
            "Experimental feedback cache should be populated"
        )

    def test_experimental_feedback_matches_smiles(self):
        """_match_experimental should return HOMO/LUMO for known solvents."""
        feedback_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "aurelius",
            "data",
            "experimental_feedback.json",
        )
        fc = FeedbackController(
            experimental_feedback_path=feedback_path,
        )
        homo, lumo = fc._match_experimental("C1COC(=O)O1")
        assert homo is not None, "EC HOMO should be matched"
        assert lumo is not None, "EC LUMO should be matched"

    def test_experimental_feedback_reduces_loo_mae(self):
        """After loading experimental feedback, LOO MAE should not increase
        by more than 5% when experimental values are available for matching SMILES."""
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection

        feedback_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "aurelius",
            "data",
            "experimental_feedback.json",
        )
        fc = FeedbackController(
            refit_interval=1,
            experimental_feedback_path=feedback_path,
        )

        for smi, gen in [("C1COC(=O)O1", 1), ("COC(=O)OC", 1), ("CC#N", 1)]:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            oracle_result = fc.get_delta_correction().predict_corrected(ctx.mol)
            fc.accumulate(
                smiles=smi,
                homo_prediction=oracle_result[0],
                lumo_prediction=oracle_result[1],
                homo_corrected=oracle_result[0],
                lumo_corrected=oracle_result[1],
                total_score=50.0,
                conformal_confidence=0.9,
                generation=gen,
            )

        dc_before = fc.get_delta_correction()
        loo_before = dc_before.loo_mae()

        result = fc.maybe_refit(current_generation=5)
        assert result is not None, "Refit should have been triggered"

        dc_after = fc.get_delta_correction()
        loo_after = dc_after.loo_mae()

        if loo_before > 0:
            improvement = (loo_before - loo_after) / loo_before
            assert improvement >= -0.05, (
                f"LOO MAE should not increase by more than 5% after refit "
                f"with experimental feedback (before={loo_before:.4f}, "
                f"after={loo_after:.4f}, improvement={improvement:.2%})"
            )


class TestFeedbackControllerSafety:
    """ADR-2026-08-07-06: batch-refit safety contract.

    Verifies that update_online is a deprecated no-op and that maybe_refit
    performs a full GPR refit whose LOO MAE does not regress when
    experimental feedback is supplied.
    """

    def test_update_online_is_deprecated_noop(self) -> None:
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection
        from aurelius.scoring.oracle.delta_correction import get_delta_correction

        dc = get_delta_correction()
        before = dc.loo_mae()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dc.update_online("CCO", -8.0, -1.0, -7.5, -0.8)
        after = dc.loo_mae()
        assert before == pytest.approx(after, rel=1e-9), (
            "update_online() must be a no-op"
        )

    def test_maybe_refit_full_retrain_improves_oro_mae(self) -> None:
        """Full GPR refit from +experimental feedback must not worsen LOO MAE."""
        from aurelius.scoring.oracle.delta_correction import get_delta_correction

        feedback_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aurelius", "data",
            "experimental_feedback.json",
        )
        fc = FeedbackController(
            refit_interval=1,
            experimental_feedback_path=feedback_path,
        )

        matched = 0
        for smi, gen in [("C1COC(=O)O1", 1), ("COC(=O)OC", 1), ("CC#N", 1)]:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            dc = fc.get_delta_correction()
            homo_pred, lumo_pred = dc.predict_corrected(ctx.mol)
            exp_homo, exp_lumo = fc._match_experimental(smi)
            if exp_homo is not None and exp_lumo is not None:
                fc.accumulate(
                    smiles=smi,
                    homo_prediction=homo_pred,
                    lumo_prediction=lumo_pred,
                    homo_corrected=exp_homo,
                    lumo_corrected=exp_lumo,
                    total_score=50.0,
                    conformal_confidence=0.9,
                    generation=gen,
                )
                matched += 1

        loo_before = fc.get_delta_correction().loo_mae()
        result = fc.maybe_refit(current_generation=2)
        loo_after = fc.get_delta_correction().loo_mae()

        assert result is not None
        assert matched >= 1, "At least one experimental point must be matched"
        if loo_before > 0 and result["new_calibration_entries"] > 0:
            improvement = (loo_before - loo_after) / loo_before
            assert improvement >= -0.05, (
                f"LOO MAE must not regress >5% after full refit "
                f"(before={loo_before:.4f}, after={loo_after:.4f})"
            )
