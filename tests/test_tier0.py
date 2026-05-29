"""Tests for Tier 0 activation energy prediction using real descriptors."""

from __future__ import annotations

import pytest

from aurelius.screening.tier0.predictor import Tier0ActivationPredictor


class TestTier0ActivationPredictor:
    """Tests for the descriptor-based Tier 0 activation energy predictor."""

    def test_predictor_returns_valid_energies(self):
        """Verify that the predictor returns physically bounded activation energies.

        All four activation energies must be in the range [0.0, 1.0] eV,
        reflecting realistic electrochemical window bounds.
        """
        predictor = Tier0ActivationPredictor()
        result = predictor.predict(
            smiles="CC(=O)OC1=CC(=O)O1",
        )
        assert isinstance(result, dict)
        for key in ("ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"):
            assert key in result
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} outside [0.0, 1.0] eV"

    def test_predict_with_descriptors_only(self):
        """Verify prediction using pre-computed descriptors dict."""
        predictor = Tier0ActivationPredictor()
        descriptors = {
            "mol_weight": 100.0,
            "num_h_donors": 1,
            "num_h_acceptors": 3,
            "num_rotatable_bonds": 2,
            "logp": 1.5,
            "tpsa": 50.0,
        }
        result = predictor.predict(descriptors=descriptors)
        assert isinstance(result, dict)
        for key in ("ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"):
            assert key in result
            val = result[key]
            assert 0.0 <= val <= 1.0

    def test_predict_empty_smiles_returns_defaults(self):
        """Verify that empty SMILES returns default predictions."""
        predictor = Tier0ActivationPredictor()
        result = predictor.predict(smiles="")
        assert isinstance(result, dict)
        for key in ("ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"):
            assert key in result

    def test_predict_invalid_smiles_raises(self):
        """Verify that invalid SMILES raises RuntimeError."""
        predictor = Tier0ActivationPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(smiles="invalid-smiles-string!!!")

    def test_predict_no_input_raises(self):
        """Verify that calling predict() with no input raises RuntimeError."""
        predictor = Tier0ActivationPredictor()
        with pytest.raises(RuntimeError, match="RDKit"):
            predictor.predict(smiles="invalid!")
