"""Tests for the experimental feedback controller (W6)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aurelius.agent.feedback import FeedbackController, FeedbackRecord, FeedbackState


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
