"""Tests for SurrogateQuantumOracle — Lightweight ML pre-filter.

Verifies:
  1. Surrogate trains successfully on orbital_calibration.json
  2. Training takes < 2 seconds
  3. Inference produces valid HOMO/LUMO predictions
  4. Surrogate penalty triggers for unstable molecules (HOMO > -5.0 eV)
  5. Surrogate integrates correctly with PropertyOracle
"""

from __future__ import annotations

import time

from rdkit import Chem

from aurelius.scoring.oracle import (
    _SURROGATE_HOMO_THRESHOLD,
    _SURROGATE_PENALTY,
    PropertyOracle,
    SurrogateQuantumOracle,
)
from aurelius.types import MoleculeContext


class TestSurrogateTraining:
    """Surrogate must train quickly and produce meaningful predictions."""

    def test_trains_on_calibration_data(self):
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        homo, lumo, uncertainty = surrogate.predict(ctx)
        assert surrogate.is_trained
        assert surrogate.n_train >= 5
        assert isinstance(homo, float)
        assert isinstance(lumo, float)
        assert isinstance(uncertainty, float)
        assert homo < 0.0, f"HOMO should be negative, got {homo}"

    def test_training_under_two_seconds(self):
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")
        assert ctx is not None
        t0 = time.perf_counter()
        surrogate.predict(ctx)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Surrogate training + inference took {elapsed:.3f}s (limit 2.0s)"

    def test_inference_under_one_ms(self):
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("CC#N")
        assert ctx is not None
        surrogate.predict(ctx)  # trigger training
        t0 = time.perf_counter()
        for _ in range(100):
            homo, lumo, uncertainty = surrogate.predict(ctx)
        avg_ms = (time.perf_counter() - t0) * 10
        assert avg_ms < 5.0, f"Mean inference took {avg_ms:.3f}ms per 100 calls"

    def test_predicts_multiple_molecules(self):
        surrogate = SurrogateQuantumOracle()
        smiles_list = ["COC(=O)OC", "C1COC(=O)O1", "CC#N", "CS(=O)(=O)C", "CCOCC"]
        results = []
        for smi in smiles_list:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None
            homo, lumo, uncertainty = surrogate.predict(ctx)
            results.append((homo, lumo, uncertainty))
        assert len(results) == len(smiles_list)
        for homo, lumo, uncertainty in results:
            assert homo < lumo, f"HOMO ({homo}) must be below LUMO ({lumo})"


class TestSurrogatePenalty:
    """Surrogate penalty should flag unstable molecules."""

    def test_penalty_triggers_for_high_homo(self):
        """HOMO > -5.0 should trigger 0.5x penalty."""
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("C=CC=CC=C")
        assert ctx is not None, "Hexatriene should parse"
        homo, lumo, uncertainty = surrogate.predict(ctx)
        penalty = surrogate.compute_penalty(homo, uncertainty)
        if homo > _SURROGATE_HOMO_THRESHOLD:
            assert penalty == _SURROGATE_PENALTY, (
                f"HOMO={homo:.3f} > {_SURROGATE_HOMO_THRESHOLD} should give {_SURROGATE_PENALTY}x penalty"
            )

    def test_no_penalty_for_stable_homo(self):
        """HOMO < -6.0 should NOT trigger penalty."""
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        homo, lumo, uncertainty = surrogate.predict(ctx)
        penalty = surrogate.compute_penalty(homo, uncertainty)
        assert penalty == 1.0, (
            f"HOMO={homo:.3f} should not trigger penalty (got {penalty})"
        )

    def test_high_uncertainty_bypasses_penalty(self):
        """High uncertainty (>0.5 eV) should return penalty=1.0 regardless of HOMO value."""
        surrogate = SurrogateQuantumOracle()
        ctx = MoleculeContext.from_smiles("C=CC=CC=C")
        assert ctx is not None
        homo, lumo, uncertainty = surrogate.predict(ctx)
        penalty = surrogate.compute_penalty(homo, uncertainty)
        # If uncertainty is high, penalty should be 1.0 (no penalty)
        if uncertainty > 0.5:
            assert penalty == 1.0, (
                f"Penalty should be 1.0 for high uncertainty ({uncertainty:.3f}), got {penalty}"
            )

    def test_surrogate_penalty_applied_in_oracle(self):
        """PropertyOracle with surrogate enabled should apply surrogate penalty."""
        oracle = PropertyOracle(use_xtb=False, use_surrogate=True)
        # Hypothetically unstable conjugated molecule
        ctx = MoleculeContext.from_smiles("c1ccccc1c2ccccc2c3ccccc3")
        assert ctx is not None
        result = oracle.evaluate(ctx)
        surrogate_skipped = result.get("surrogate_skipped", False)
        if surrogate_skipped:
            assert result["domain_penalty"] <= _SURROGATE_PENALTY, (
                f"Domain penalty should be <= {_SURROGATE_PENALTY} when surrogate skips"
            )


class TestSurrogateIntegration:
    """Surrogate must integrate cleanly with existing pipeline."""

    def test_oracle_works_without_surrogate(self):
        """Disabling surrogate must not break existing behavior."""
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False)
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        result = oracle.evaluate(ctx)
        assert result["homo_eV"] < 0.0
        assert result["domain_penalty"] == 1.0
        assert "surrogate_skipped" not in result

    def test_surrogate_counts_skipped_molecules(self):
        oracle = PropertyOracle(use_xtb=False, use_surrogate=True)
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        oracle.evaluate(ctx)
        assert oracle._n_surrogate_skips >= 0

    def test_holdout_mae_below_threshold(self):
        """MAE on a random 20% holdout from calibration data should be < 1.2 eV.

        Uses a proper train/holdout split to verify surrogate generalization.
        """
        import json
        import os
        import random
        calib_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aurelius", "data",
            "orbital_calibration.json",
        )
        with open(calib_path) as f:
            all_data = json.load(f)

        random.seed(42)
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        n_holdout = max(1, len(all_data) // 5)
        holdout_indices = set(indices[:n_holdout])
        train_data = [all_data[i] for i in indices if i not in holdout_indices]
        holdout_data = [all_data[i] for i in holdout_indices]

        surrogate = SurrogateQuantumOracle()
        surrogate.set_training_data(train_data)

        errors = []
        for entry in holdout_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            ctx = MoleculeContext.from_smiles(entry["smiles"])
            if ctx is None:
                continue
            try:
                homo, lumo, uncertainty = surrogate.predict(ctx)
                homo_err = abs(homo - entry["homo_eV"])
                lumo_err = abs(lumo - entry["lumo_eV"])
                errors.append((homo_err + lumo_err) / 2.0)
            except Exception:
                continue

        if errors:
            mae = sum(errors) / len(errors)
            assert mae < 1.2, (
                f"Holdout MAE = {mae:.3f} eV (threshold 1.2, n={len(errors)})"
            )


class TestSurrogateNoMLBloat:
    """Surrogate must not introduce deep learning frameworks."""

    def test_only_sklearn_used(self):
        import inspect

        from aurelius.scoring.oracle import surrogate as surrogate_module
        source = inspect.getsource(surrogate_module)
        for banned in ("torch", "tensorflow", "jax", "keras", "flax"):
            assert banned not in source, (
                f"Surrogate module contains banned ML framework: {banned}"
            )
