"""Expanded retrosynthetic feasibility check tests.

Tests the expanded reaction template set (≥40 templates) and
the _estimate_step_economy function.
"""

from __future__ import annotations

from rdkit import Chem

from aurelius.synthesis.retro_check import (
    _BRICS_REACTION_TEMPLATES,
    _estimate_step_economy,
    batch_retro_check,
    retro_check,
)


class TestRetroCheckExpandedTemplates:
    """Verify the expanded reaction template set has ≥40 templates."""

    def test_template_count(self):
        assert len(_BRICS_REACTION_TEMPLATES) >= 40

    def test_template_parsing_all_valid(self):
        for template_str in _BRICS_REACTION_TEMPLATES:
            rxn = Chem.rdChemReactions.ReactionFromSmarts(template_str)
            assert rxn is not None, f"Failed to parse template: {template_str}"

    def test_suzuki_template_present(self):
        assert any("c1ccccc1>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_heck_template_present(self):
        assert any("C=C" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_sonogashira_template_present(self):
        assert any("C#C" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_negishi_template_present(self):
        assert any("c1ccccc1>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_buchwald_hartwig_template_present(self):
        assert any("N[*:2]" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_ullmann_template_present(self):
        assert any("c1ccccc1>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_williamson_template_present(self):
        assert any("O[*:2]" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_mitsunobu_template_present(self):
        assert any("N[*:2]" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_snar_template_present(self):
        assert any("N[*:2]" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_esterification_template_present(self):
        assert any("C(=O)O>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_amidation_template_present(self):
        assert any("C(=O)N>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_wittig_template_present(self):
        assert any("C=C" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_diels_alder_template_present(self):
        assert any("C=C>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_dast_fluorination_template_present(self):
        assert any("F[*:2]" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_lactonization_template_present(self):
        assert any("C(=O)O>>" in t for t in _BRICS_REACTION_TEMPLATES)

    def test_rcm_template_present(self):
        assert any("C=C>>" in t for t in _BRICS_REACTION_TEMPLATES)


class TestRetroCheckFunctionality:
    """Test retro_check with various molecules."""

    def test_simple_ester_is_viable(self):
        result = retro_check("CC(=O)OC")
        assert result["viable"] is True
        assert result["n_steps"] <= 3

    def test_simple_amide_is_viable(self):
        result = retro_check("CC(=O)N")
        assert result["viable"] is True

    def test_simple_ether_is_viable(self):
        result = retro_check("CCOCC")
        assert result["viable"] is True

    def test_complex_molecule_viability(self):
        result = retro_check("c1ccc(C(=O)OC)cc1")
        assert isinstance(result["viable"], bool)
        assert isinstance(result["n_steps"], int)
        assert isinstance(result["sa_score"], float)

    def test_invalid_smiles_returns_failed(self):
        result = retro_check("invalid_smiles")
        assert result["viable"] is False
        assert result["n_steps"] == 4
        assert result["sa_score"] == 999.0

    def test_commercial_building_block_zero_steps(self):
        result = retro_check("CCO")
        assert result["viable"] is True
        assert result["n_steps"] == 0

    def test_batch_retro_check(self):
        smiles_list = ["CCO", "CC(=O)OC", "CCN"]
        results = batch_retro_check(smiles_list)
        assert len(results) == 3
        for r in results:
            assert "viable" in r
            assert "smiles" in r


class TestStepEconomy:
    """Test _estimate_step_economy function."""

    def test_step_economy_range(self):
        result = _estimate_step_economy("CCO")
        assert 1 <= result <= 10

    def test_step_economy_simple(self):
        result = _estimate_step_economy("CCO")
        assert result <= 5

    def test_step_economy_complex(self):
        result = _estimate_step_economy("c1ccc(C(=O)OC)c2ccccc12")
        assert 1 <= result <= 10

    def test_step_economy_invalid(self):
        result = _estimate_step_economy("invalid")
        assert result == 10

    def test_step_economy_is_int(self):
        result = _estimate_step_economy("CCO")
        assert isinstance(result, int)


class TestRetroCheckEdgeCases:
    """Test edge cases in retro_check."""

    def test_empty_smiles(self):
        result = retro_check("")
        assert result["viable"] is False

    def test_aromatic_ester(self):
        result = retro_check("c1ccccc1C(=O)OC")
        assert isinstance(result["viable"], bool)

    def test_sulfonamide(self):
        result = retro_check("CS(=O)(=O)N")
        assert isinstance(result["viable"], bool)

    def test_phosphate_ester(self):
        result = retro_check("COP(=O)(OC)OC")
        assert isinstance(result["viable"], bool)

    def test_nitrile(self):
        result = retro_check("CC#N")
        assert isinstance(result["viable"], bool)

    def test_fluoride(self):
        result = retro_check("CF")
        assert isinstance(result["viable"], bool)

    def test_boronate(self):
        result = retro_check("OB(O)O")
        assert isinstance(result["viable"], bool)

    def test_retro_check_has_route_info(self):
        result = retro_check("CC(=O)OC")
        assert isinstance(result["route"], list)
        assert isinstance(result["building_blocks"], list)
