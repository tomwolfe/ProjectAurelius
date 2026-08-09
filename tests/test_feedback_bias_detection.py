"""Tests for systematic physical-model bias detection (ADR-2026-08-07-09)."""

from __future__ import annotations

from aurelius.agent.feedback import FeedbackController


def _make_controller_with_dielectric_records(
    n: int,
    offset: float,
    noise: bool = False,
) -> FeedbackController:
    """Build a controller with n synthetic dielectric records.

    Args:
        n: Number of records to create.
        offset: Systematic offset added to all experimental values.
        noise: If True, add random noise instead of a fixed offset.
    """
    import random as py_random

    py_random.seed(123)
    fc = FeedbackController()
    for i in range(n):
        pred = 5.0 + i * 0.1
        exp_val = pred + py_random.uniform(-1.0, 1.0) if noise else pred + offset
        fc.accumulate(
            smiles=f"CC{'C' * i}O",
            homo_prediction=-8.0,
            lumo_prediction=-1.0,
            homo_corrected=-7.5,
            lumo_corrected=-0.8,
            total_score=70.0,
            conformal_confidence=0.9,
            generation=1,
            predicted_dielectric=pred,
            experimental_dielectric=exp_val,
        )
    return fc


def _make_controller_with_viscosity_records(n: int, offset: float) -> FeedbackController:
    """Build a controller with n synthetic viscosity records."""
    fc = FeedbackController()
    for i in range(n):
        pred = 2.0 + i * 0.1
        exp_val = pred + offset
        fc.accumulate(
            smiles=f"CC{'C' * i}O",
            homo_prediction=-8.0,
            lumo_prediction=-1.0,
            homo_corrected=-7.5,
            lumo_corrected=-0.8,
            total_score=70.0,
            conformal_confidence=0.9,
            generation=1,
            predicted_viscosity=pred,
            experimental_viscosity=exp_val,
        )
    return fc


class TestSystematicBiasDetection:
    """detect_systematic_bias() must flag systematic model deviation."""

    def test_dielectric_systematic_offset_detected(self):
        """Systematic dielectric overprediction should be flagged."""
        fc = _make_controller_with_dielectric_records(n=15, offset=3.0)
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is True
        assert bias["dielectric"]["n_records"] == 15
        assert bias["dielectric"]["direction"] == "underpredicted"
        assert bias["dielectric"]["magnitude"] > 2.0

    def test_viscosity_systematic_offset_detected(self):
        """Systematic viscosity overprediction should be flagged."""
        fc = _make_controller_with_viscosity_records(n=12, offset=1.0)
        bias = fc.detect_systematic_bias()
        assert bias["viscosity"]["bias_detected"] is True
        assert bias["viscosity"]["n_records"] == 12
        assert bias["viscosity"]["direction"] == "underpredicted"
        assert bias["viscosity"]["magnitude"] > 0.5

    def test_random_noise_not_flagged(self):
        """Random noise within tolerance should not be flagged."""
        fc = _make_controller_with_dielectric_records(n=20, offset=0.0, noise=True)
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is False

    def test_small_systematic_offset_not_flagged(self):
        """A small offset below threshold should not trigger."""
        fc = _make_controller_with_dielectric_records(n=15, offset=0.5)
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is False

    def test_fewer_than_10_records_not_flagged(self):
        """Below minimum record count, never flag regardless of offset."""
        fc = _make_controller_with_dielectric_records(n=8, offset=5.0)
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is False
        assert bias["dielectric"]["n_records"] == 8

    def test_no_records_returns_zero(self):
        """Empty controller should return no records and no bias."""
        fc = FeedbackController()
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is False
        assert bias["viscosity"]["bias_detected"] is False
        assert bias["dielectric"]["n_records"] == 0
        assert bias["viscosity"]["n_records"] == 0

    def test_both_properties_flagged_independently(self):
        """Dielectric and viscosity can each have independent bias."""
        fc = FeedbackController()
        for i in range(12):
            fc.accumulate(
                smiles=f"CC{'C' * i}O",
                homo_prediction=-8.0,
                lumo_prediction=-1.0,
                homo_corrected=-7.5,
                lumo_corrected=-0.8,
                total_score=70.0,
                conformal_confidence=0.9,
                generation=1,
                predicted_dielectric=5.0,
                experimental_dielectric=9.0,  # +4.0 offset → flagged
                predicted_viscosity=2.0,
                experimental_viscosity=2.2,  # +0.2 offset → not flagged
            )
        bias = fc.detect_systematic_bias()
        assert bias["dielectric"]["bias_detected"] is True
        assert bias["viscosity"]["bias_detected"] is False
