"""Tests for Tier 1 Electrolyte Viability Filter.

Updated for strict MoleculeContext API.
"""

from __future__ import annotations

import pytest

from aurelius.screening.tier1 import Filter
from aurelius.types import MoleculeContext


class TestFilter:
    def setup_method(self):
        self.filter = Filter()

    def _ctx(self, smiles: str) -> MoleculeContext:
        ctx = MoleculeContext.from_smiles(smiles)
        assert ctx is not None
        return ctx

    def test_screen_valid_molecule(self):
        result = self.filter.screen(self._ctx("CC(=O)OC1=CC(=O)O1"))
        assert isinstance(result, dict)
        assert "is_viable" in result
        assert "electrolyte_violations" in result
        assert "inference_time_ms" in result

    def test_deterministic_output_same_smiles(self):
        smiles = "CC(=O)OC1=CC(=O)O1"
        results = [self.filter.screen(self._ctx(smiles)) for _ in range(5)]
        assert all(r["is_viable"] == results[0]["is_viable"] for r in results)
        assert all(r["electrolyte_violations"] == results[0]["electrolyte_violations"] for r in results)

    def test_different_smiles_may_differ(self):
        smiles_list = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = [self.filter.screen(self._ctx(s)) for s in smiles_list]
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_screen_batch(self):
        smiles_list = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        contexts = [self._ctx(s) for s in smiles_list]
        results = self.filter.screen_batch(contexts)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_invalid_smiles_raises(self):
        with pytest.raises(TypeError):
            self.filter.screen("not_a_valid_smiles")

    def test_screen_smiles_handles_invalid(self):
        result = self.filter.screen_smiles("C1=CC=CC=CC=C1=Z")
        assert result["is_viable"] is False
        assert "Invalid" in result["electrolyte_violations"][0]

    def test_is_viable_smiles_static(self):
        result = Filter.is_viable_smiles("CCO")
        assert isinstance(result, bool)

    def test_ethanol_fails_hbd(self):
        """Ethanol (CCO) has 1 H-bond donor (OH) — must fail electrolyte filter."""
        result = self.filter.screen(self._ctx("CCO"))
        assert result["is_viable"] is False
        assert any("H-bond donors present" in v for v in result["electrolyte_violations"])

    def test_dmc_passes(self):
        """Dimethyl carbonate (COC(=O)OC) has MW=90, HBD=0, rot=2, HBA=3 — must pass."""
        result = self.filter.screen(self._ctx("COC(=O)OC"))
        assert result["is_viable"] is True
        assert len(result["electrolyte_violations"]) == 0

    def test_large_molecule_not_viable(self):
        result = self.filter.screen(
            self._ctx("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        )
        assert result["is_viable"] is False
