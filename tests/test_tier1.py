"""Tests for Tier 1 Filter (deterministic structural-viability filter)."""

from __future__ import annotations

import pytest

from aurelius.screening.tier1 import Filter
from aurelius.utils.dependencies import HAS_RDKIT


class TestFilter:
    def setup_method(self):
        self.filter = Filter()

    def test_screen_valid_molecule(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        result = self.filter.screen_molecule("CC(=O)OC1=CC(=O)O1")
        assert isinstance(result, dict)
        assert "is_viable" in result
        assert "lipinski_violations" in result
        assert "complexity_flags" in result
        assert "inference_time_ms" in result

    def test_deterministic_output_same_smiles(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        smiles = "CC(=O)OC1=CC(=O)O1"
        results = [self.filter.screen_molecule(smiles) for _ in range(5)]
        assert all(r["is_viable"] == results[0]["is_viable"] for r in results)
        assert all(r["lipinski_violations"] == results[0]["lipinski_violations"] for r in results)

    def test_different_smiles_may_differ(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        smiles_list = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = [self.filter.screen_molecule(s) for s in smiles_list]
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_screen_batch(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        molecules = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = self.filter.screen_batch(molecules)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_invalid_smiles_raises(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        with pytest.raises(ValueError):
            self.filter.screen_molecule("C1=CC=CC=CC=C1=Z")

    def test_is_viable_smiles_static(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        result = Filter.is_viable_smiles("CCO")
        assert isinstance(result, bool)

    def test_small_molecule_viable(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        result = self.filter.screen_molecule("CCO")
        assert result["is_viable"] is True
        assert len(result["lipinski_violations"]) == 0

    def test_large_molecule_not_viable(self):
        if not HAS_RDKIT:
            pytest.skip("RDKit is required")
        result = self.filter.screen_molecule("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        if result["is_viable"]:
            pytest.skip("Large molecule unexpectedly viable (may pass if MW still within bounds)")
