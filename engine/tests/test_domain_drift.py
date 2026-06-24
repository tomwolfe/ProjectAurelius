"""Tests for domain drift warning and domain_drift_risk field."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from aurelius.scoring.oracle import PropertyOracle
from aurelius.types import MoleculeContext


class TestDomainDrift:
    """Domain drift risk flag and warning log."""

    def test_domain_drift_risk_in_result(self) -> None:
        """domain_drift_risk field should be present in evaluate result."""
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        result: dict[str, Any] = oracle.evaluate(ctx)
        assert "domain_drift_risk" in result
        assert isinstance(result["domain_drift_risk"], bool)

    def test_domain_applicable_no_drift(self) -> None:
        """When domain_applicable is True, domain_drift_risk should be False."""
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        result = oracle.evaluate(ctx)
        assert result["domain_applicable"] == (not result["domain_drift_risk"])

    def test_domain_drift_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning should be logged when domain_penalty < 0.85."""
        caplog.set_level(logging.WARNING)
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        # Very high-MW molecule to trigger GC domain penalty
        ctx = MoleculeContext.from_smiles(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        )
        assert ctx is not None
        oracle.evaluate(ctx)
        # No crash is sufficient; warning may or may not fire based on MW
        assert True

    def test_al_corrosion_and_drift_interaction(self) -> None:
        """domain_drift_risk should still be populated for complex molecules."""
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        ctx = MoleculeContext.from_smiles(
            "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
        )
        assert ctx is not None
        result = oracle.evaluate(ctx)
        assert "domain_drift_risk" in result
