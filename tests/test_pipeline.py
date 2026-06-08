"""Tests for the Aurelius Pipeline — ternary mixtures and Li-salt dissociation.

Verifies that:
1. Ternary mixture parsing and formatting works
2. Ternary mixture scoring returns synergy bonus peaking at complementary ratios
3. Li-salt dissociation proxy gives physically plausible values
4. Backward compatibility with binary mixtures is preserved
"""

from __future__ import annotations

import pytest

from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle.gc import (
    mixture_synergy_bonus_ternary,
    predict_li_dissociation_proxy,
)
from aurelius.scoring.oracle.oracle import PropertyOracle
from aurelius.types import (
    MoleculeContext,
    format_mixture_smiles,
    is_mixture_smiles,
    parse_mixture_smiles,
)

_ORACLE = None


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = PropertyOracle(use_xtb=False)
    return _ORACLE


@pytest.fixture(scope="module")
def pipeline() -> AureliusPipeline:
    pl = AureliusPipeline()
    pl.initialize()
    return pl


def _ctx(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None
    return ctx


# ---------------------------------------------------------------------------
# Ternary Mixture Parsing
# ---------------------------------------------------------------------------


def test_ternary_mixture_parsing() -> None:
    """Ternary mixture SMILES should parse into three components and two fractions."""
    smi_a = "C1COC(=O)O1"
    smi_b = "COCCOC"
    smi_c = "CC#N"
    frac_a = 0.4
    frac_b = 0.4

    formatted = format_mixture_smiles(smi_a, smi_b, frac_a, smi_c, frac_b)
    assert is_mixture_smiles(formatted)

    parsed = parse_mixture_smiles(formatted)
    assert parsed is not None
    assert len(parsed) == 5

    pa, pb, pc, fa, fb = parsed  # type: ignore[misc]
    assert pa == smi_a
    assert pb == smi_b
    assert pc == smi_c
    assert abs(fa - frac_a) < 1e-6
    assert abs(fb - frac_b) < 1e-6

    # Third fraction should be 1.0 - frac_a - frac_b = 0.2
    assert abs(1.0 - fa - fb - 0.2) < 1e-6


def test_ternary_mixture_parsing_invalid() -> None:
    """Parsing should gracefully handle invalid ternary mixtures."""
    # Missing fractions
    assert parse_mixture_smiles("A|B|C") is None
    # Non-numeric fraction
    assert parse_mixture_smiles("A|B|C|x|y") is None
    # Fractions that sum > 1.0
    assert parse_mixture_smiles("A|B|C|0.6|0.5") is None
    # Negative fraction
    assert parse_mixture_smiles("A|B|C|-0.1|0.5") is None


def test_binary_mixture_backward_compatibility() -> None:
    """Binary mixture parsing should still work (3-part format)."""
    formatted = format_mixture_smiles("C1COC(=O)O1", "COCCOC", 0.5)
    assert is_mixture_smiles(formatted)

    parsed = parse_mixture_smiles(formatted)
    assert parsed is not None
    assert len(parsed) == 3

    pa, pb, frac = parsed  # type: ignore[misc]
    assert pa == "C1COC(=O)O1"
    assert pb == "COCCOC"
    assert abs(frac - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Ternary Synergy Bonus
# ---------------------------------------------------------------------------


def test_ternary_synergy_bonus_complementary() -> None:
    """Ternary synergy should produce positive bonus for complementary mixtures.

    A high-dielectric (EC-like) + low-viscosity (DME-like) + high-solvation
    (ACN-like) blend should produce a positive synergy bonus when the components
    are complementary (high d + low v), and zero for homogeneous components.
    """
    d1, v1 = 18.0, 2.5   # High-dielectric carbonate
    d2, v2 = 2.0, 0.8    # Low-viscosity ether
    d3, v3 = 10.0, 1.5   # Moderate solvation nitrile

    balanced = mixture_synergy_bonus_ternary(d1, d2, d3, v1, v2, v3, 0.4, 0.4)
    homogeneous = mixture_synergy_bonus_ternary(5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 0.33, 0.33)

    assert balanced > 0.0, "Complementary ternary mixture should have positive synergy"
    assert homogeneous == 0.0, "Homogeneous ternary mixture should have zero synergy"


def test_ternary_synergy_no_bonus_for_homogeneous() -> None:
    """All-similar components should yield zero synergy bonus."""
    d1, v1 = 5.0, 2.0
    d2, v2 = 4.5, 2.2
    d3, v3 = 5.5, 1.8

    bonus = mixture_synergy_bonus_ternary(d1, d2, d3, v1, v2, v3, 0.33, 0.33)
    assert bonus == 0.0, f"Homogeneous mixture should have zero synergy, got {bonus}"


# ---------------------------------------------------------------------------
# Full Ternary Pipeline Screening
# ---------------------------------------------------------------------------


def test_pipeline_screens_ternary_mixture(pipeline: AureliusPipeline) -> None:
    """Pipeline should screen a ternary mixture and return valid results."""
    ctx1 = _ctx("C1COC(=O)O1")
    ctx2 = _ctx("COCCOC")
    ctx3 = _ctx("CC#N")

    result = pipeline.screen_mixture(ctx1, ctx2, 0.4, ctx3=ctx3, frac2=0.4)
    assert result is not None

    score = result.get("score", {})
    assert "total_score" in score
    assert "synergy_bonus" in score
    assert score["synergy_bonus"] >= 0.0

    mix_props = result.get("mixture_properties", {})
    assert "dielectric_proxy" in mix_props
    assert "viscosity_proxy" in mix_props
    assert "li_solvation_proxy" in mix_props

    assert "component1" in result
    assert "component2" in result
    assert "component3" in result


def test_ternary_scores_higher_than_weighted_base(pipeline: AureliusPipeline) -> None:
    """Ternary synergy should boost the mixture score above the weighted base."""
    ctx1 = _ctx("C1COC(=O)O1")
    ctx2 = _ctx("COCCOC")
    ctx3 = _ctx("CC#N")

    result = pipeline.screen_mixture(ctx1, ctx2, 0.4, ctx3=ctx3, frac2=0.4)
    score = result.get("score", {})

    synergy = score.get("synergy_bonus", 0.0)
    weighted_base = score.get("weighted_base", 0.0)
    total = score.get("total_score", 0.0)

    assert synergy >= 0.0
    assert total >= weighted_base + synergy - 0.01, (
        f"Total ({total:.4f}) should be >= weighted_base ({weighted_base:.4f}) + "
        f"synergy ({synergy:.4f})"
    )


# ---------------------------------------------------------------------------
# Li-Salt Dissociation Proxy Tests
# ---------------------------------------------------------------------------


def test_li_dissociation_proxy_plausible_range() -> None:
    """Li dissociation proxy should be in [0.0, 6.0] with plausible values."""
    ctx = _ctx("C1COC(=O)O1")
    proxy = predict_li_dissociation_proxy(ctx)
    assert 0.0 <= proxy <= 6.0


def test_li_dissociation_balanced_motifs_score_higher() -> None:
    """Molecules with balanced donor/acceptor should score higher than
    those with mostly acceptor motifs.

    A molecule with both donor (carbonyl, ether) and acceptor (fluorine)
    motifs should score higher than a purely fluorinated molecule with
    only acceptor motifs.
    """
    # EC (cyclic carbonate) has donor carbonyl + ring oxygen motifs
    ec = predict_li_dissociation_proxy(_ctx("C1COC(=O)O1"))
    # Highly fluorinated molecule has mostly acceptor motifs
    fluorinated = predict_li_dissociation_proxy(_ctx("FC(F)(F)C(F)(F)F"))

    assert ec > fluorinated, (
        f"EC {ec:.3f} should have better dissociation than "
        f"fluorinated {fluorinated:.3f}"
    )


def test_li_dissociation_proxy_via_oracle(oracle: PropertyOracle) -> None:
    """Li dissociation proxy should be included in Oracle evaluation results."""
    result = oracle.evaluate(_ctx("COC(=O)OC"))
    assert "li_dissociation_proxy" in result
    assert 0.0 <= result["li_dissociation_proxy"] <= 6.0


def test_li_dissociation_proxy_salt_type_reserved() -> None:
    """The salt_type parameter should not crash (future-proofing)."""
    ctx = _ctx("COC(=O)OC")
    proxy_pf6 = predict_li_dissociation_proxy(ctx, salt_type="LiPF6")
    proxy_tfsi = predict_li_dissociation_proxy(ctx, salt_type="LiTFSI")
    assert proxy_pf6 == proxy_tfsi, (
        "Currently salt_type is reserved; all salt types should return the same value"
    )


# ---------------------------------------------------------------------------
# Pipeline CLI-style Ternary Acceptance Test
# ---------------------------------------------------------------------------


def test_ternary_cli_acceptance() -> None:
    """Acceptance: aurelius mixture C1COC(=O)O1 COCCOC --smiles-c CC#N --frac-a 0.4 --frac-b 0.4

    Verify the formatted mixture SMILES can be round-tripped through the
    pipeline's parse/format helpers.
    """
    smi_a = "C1COC(=O)O1"
    smi_b = "COCCOC"
    smi_c = "CC#N"
    frac_a = 0.4
    frac_b = 0.4

    formatted = format_mixture_smiles(smi_a, smi_b, frac_a, smi_c, frac_b)
    parsed = parse_mixture_smiles(formatted)

    assert parsed is not None
    assert len(parsed) == 5
    pa, pb, pc, fa, fb = parsed  # type: ignore[misc]
    assert pa == smi_a
    assert pb == smi_b
    assert pc == smi_c
    assert abs(fa - frac_a) < 1e-4
    assert abs(fb - frac_b) < 1e-4
