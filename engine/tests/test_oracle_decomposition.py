"""Tests for oracle decomposition — verifying that private methods are called correctly."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aurelius.scoring.oracle.oracle import PropertyOracle
from aurelius.types import MoleculeContext


@pytest.fixture
def oracle() -> PropertyOracle:
    return PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)


class TestOracleDecomposition:
    """Verify that PropertyOracle's evaluate() delegates to private methods."""

    def test_evaluate_calls_private_methods_in_order(self, oracle: PropertyOracle) -> None:
        """evaluate() must call _run_surrogate, _compute_quantum, _compute_uq_penalty,
        _compute_gc_properties, _build_domain, _apply_sei_penalty, and _assemble_result."""
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None

        with patch.object(oracle, "_run_surrogate", return_value=(1.0, -99.0, 99.0, False)) as mock_surrogate:
            with patch.object(oracle, "_compute_quantum", return_value=(-6.5, -0.5, 6.0, "tom", "tom_high", 0.0, 0.0, 0.0)) as mock_quantum:
                with patch.object(oracle, "_compute_uq_penalty", return_value=(1.0, 0.0, 0.0)) as mock_uq:
                    with patch.object(oracle, "_compute_gc_properties", return_value={}) as mock_gc:
                        with patch.object(oracle, "_build_domain", return_value=(0.9, "ok", True)) as mock_domain:
                            with patch.object(oracle, "_apply_sei_penalty", return_value=(0.9, "ok", True)) as mock_sei:
                                with patch.object(oracle, "_assemble_result", return_value={"homo_eV": -6.5, "lumo_eV": -0.5}) as mock_assemble:
                                    result = oracle.evaluate(ctx)

        # Verify all private methods were called
        mock_surrogate.assert_called_once_with(ctx)
        mock_quantum.assert_called_once()
        mock_uq.assert_called_once()
        mock_gc.assert_called_once()
        mock_domain.assert_called_once()
        mock_sei.assert_called_once()
        mock_assemble.assert_called_once()

    def test_mocking_run_surrogate_skips_quantum(self, oracle: PropertyOracle) -> None:
        """When _run_surrogate returns skip_quantum=True, _compute_quantum should return surrogate values."""
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None

        with patch.object(oracle, "_run_surrogate", return_value=(1.0, -6.0, -0.5, True)) as mock_surrogate:
            with patch.object(oracle, "_compute_quantum", return_value=(-6.0, -0.5, 5.5, "surrogate", "surrogate", 0.0, 0.0, 0.0)) as mock_quantum:
                with patch.object(oracle, "_compute_uq_penalty", return_value=(1.0, 0.0, 0.0)) as mock_uq:
                    with patch.object(oracle, "_compute_gc_properties", return_value={}) as mock_gc:
                        with patch.object(oracle, "_build_domain", return_value=(0.9, "ok", True)) as mock_domain:
                            with patch.object(oracle, "_apply_sei_penalty", return_value=(0.9, "ok", True)) as mock_sei:
                                with patch.object(oracle, "_assemble_result", return_value={"homo_eV": -6.0, "lumo_eV": -0.5}) as mock_assemble:
                                    result = oracle.evaluate(ctx)

        # Verify skip_quantum was passed to _compute_quantum
        call_args = mock_quantum.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "skip_quantum" in kwargs
        assert kwargs["skip_quantum"] is True

    def test_evaluate_returns_valid_dict(self, oracle: PropertyOracle) -> None:
        """evaluate() must return a dict with all expected keys."""
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        result = oracle.evaluate(ctx)

        assert isinstance(result, dict)
        assert "homo_eV" in result
        assert "lumo_eV" in result
        assert "gap_eV" in result
        assert "domain_applicable" in result
        assert "domain_reason" in result
        assert "domain_penalty" in result
        assert "quantum_method" in result
        assert "quantum_confidence" in result

    def test_evaluate_caches_result(self, oracle: PropertyOracle) -> None:
        """Calling evaluate twice with the same SMILES must return identical results (cache hit)."""
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        r1 = oracle.evaluate(ctx)
        r2 = oracle.evaluate(ctx)
        assert r1 == r2
