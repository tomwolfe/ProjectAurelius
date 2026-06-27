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

    def test_screen_does_not_call_get_ecfp4(self):
        """ECFP4 fingerprint must not be computed during Tier 1 screening.

        Tier 1 only checks MW, HBD, rotatable bonds, HBA count, and LogP.
        Accessing the fingerprint during screening would waste computation
        on molecules that will be rejected anyway.
        """
        import unittest.mock

        with unittest.mock.patch(
            "aurelius.types.MoleculeContext.get_ecfp4",
        ) as mock_ecfp4:
            ctx = self._ctx("COC(=O)OC")
            result = self.filter.screen(ctx)
            assert result["is_viable"] is True
            mock_ecfp4.assert_not_called()


class TestStructuralPreCheck:
    """Tests for the lightweight ``is_structurally_viable`` pre-check.

    This pure-Python gate catches obviously invalid molecules (hypervalent
    species, unmatched syntax) *before* they reach RDKit's C++ layer,
    eliminating the need for heavy stderr redirection in mutation hot loops.
    """

    # ------------------------------------------------------------------
    # Valid molecules — must pass
    # ------------------------------------------------------------------

    def test_valid_ethanol(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("CCO")

    def test_valid_dimethyl_carbonate(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("COC(=O)OC")

    def test_valid_ethylene_carbonate(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C1COC(=O)O1")

    def test_valid_cyclohexane(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C1CCCCC1")

    def test_valid_benzene(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("c1ccccc1")

    def test_valid_carbon_tetrafluoride(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C(F)(F)(F)F")

    def test_valid_ethene(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C=C")

    def test_valid_butadiene(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C=CC=C")

    def test_valid_neopentane(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C(C)(C)(C)C")

    def test_valid_acetonitrile(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("CC#N")

    def test_valid_dimethyl_sulfone(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("CS(=O)(=O)C")

    def test_valid_tetramethylammonium(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("[N+](C)(C)(C)C")

    def test_valid_dot_disconnected(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("C.C")

    def test_valid_aromatic_oxygen(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("c1ccoc1")

    def test_valid_aromatic_nitrogen(self):
        from aurelius.screening.structural import is_structurally_viable
        assert is_structurally_viable("c1ccncc1")

    # ------------------------------------------------------------------
    # Invalid molecules — must be rejected
    # ------------------------------------------------------------------

    def test_invalid_unmatched_parens(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C(C")

    def test_invalid_unmatched_closing_parens(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C)C")

    def test_invalid_unmatched_brackets(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("[C")

    def test_invalid_unmatched_ring_digits(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C1CC")

    def test_invalid_empty_string(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("")

    def test_invalid_none(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable(None)  # type: ignore[arg-type]

    def test_invalid_pentavalent_carbon(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C(F)(F)(F)(F)(F)")

    def test_invalid_hexavalent_carbon(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C(F)(F)(F)(F)(F)(F)")

    def test_invalid_trivalent_oxygen(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("O(F)(F)(F)")

    def test_invalid_tetravalent_nitrogen_neutral(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("N(C)(C)(C)(C)")

    def test_invalid_divalent_fluorine(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("F(C)(C)")

    def test_invalid_carbon_double_plus_three_single(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C(=O)(F)(F)(F)")

    def test_invalid_pentavalent_carbon_from_branches(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("C(C)(C)(C)(C)(C)")

    def test_invalid_heptavalent_sulfur(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("S(F)(F)(F)(F)(F)(F)(F)")

    def test_invalid_divalent_chlorine(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("Cl(C)(C)")

    def test_invalid_divalent_bromine(self):
        from aurelius.screening.structural import is_structurally_viable
        assert not is_structurally_viable("Br(C)(C)")

    # ------------------------------------------------------------------
    # Integration tests — pre-check runs before RDKit
    # ------------------------------------------------------------------

    def test_molecule_context_rejects_invalid_pre_rdkit(self):
        """MoleculeContext.from_smiles must return None for invalid SMILES
        without reaching RDKit's C++ layer."""
        ctx = MoleculeContext.from_smiles("C(F)(F)(F)(F)(F)")
        assert ctx is None, (
            "Pentavalent carbon should be rejected by structural pre-check"
        )

    def test_molecule_context_accepts_valid(self):
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None

    def test_filter_screen_smiles_rejects_hypervalent(self):
        """Tier 1 filter must reject hypervalent molecules via the pre-check."""
        result = Filter().screen_smiles("C(F)(F)(F)(F)(F)")
        assert result["is_viable"] is False
        assert any("Invalid" in v or "pre-check" in v for v in result["electrolyte_violations"])

    def test_mutation_engine_rejects_invalid_seeds(self, monkeypatch):
        """MutationEngine must handle invalid SMILES gracefully."""
        from aurelius.agent.mutation import MutationEngine
        engine = MutationEngine(seed_smiles=["CCO"])  # valid seed
        ctx = engine._get_ctx("C(F)(F)(F)(F)(F)")
        assert ctx is None, "Engine must reject pentavalent carbon"
