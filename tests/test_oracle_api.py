"""Tests for the standalone oracle API (Feature 3: v11.0).

Verifies that:
1. predict_properties returns full property dict matching pipeline.screen_molecule()
2. get_domain_applicability returns a (penalty, reason) tuple
3. predict_mixture supports binary and ternary components with ideal mixing
4. Validation gates reject malformed inputs (anti-gaming)
"""

from __future__ import annotations

import pytest

from aurelius.oracle_api import (
    get_domain_applicability,
    predict_mixture,
    predict_properties,
    reset_pipeline,
)
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext


@pytest.fixture(autouse=True)
def _fresh_pipeline() -> None:
    reset_pipeline()
    yield
    reset_pipeline()


def test_predict_properties_returns_all_keys() -> None:
    props = predict_properties("C1COC(=O)O1")
    for key in (
        "homo_eV",
        "lumo_eV",
        "gap_eV",
        "dielectric_proxy",
        "viscosity_proxy",
        "li_solvation_proxy",
        "domain_applicable",
        "domain_penalty",
        "total_score",
        "is_viable",
        "smiles",
    ):
        assert key in props, f"Missing key {key}"


def test_predict_properties_matches_pipeline() -> None:
    """API results must equal pipeline.screen_molecule() for identical input."""
    pipeline = AureliusPipeline()
    pipeline.initialize()
    ctx = MoleculeContext.from_smiles("C1COC(=O)O1")
    assert ctx is not None

    direct = pipeline.screen_molecule(ctx)
    api = predict_properties("C1COC(=O)O1")

    for key in ("homo_eV", "lumo_eV", "dielectric_proxy", "viscosity_proxy",
                "li_solvation_proxy", "domain_penalty"):
        assert api[key] == direct["tier2"][key], f"Mismatch on {key}"
    assert api["total_score"] == direct["score"]["total_score"]
    assert api["is_viable"] == direct["score"]["is_viable"]


def test_predict_properties_invalid_smiles_raises() -> None:
    with pytest.raises(ValueError):
        predict_properties("not_a_valid_smiles!!")


def test_get_domain_applicability_shape() -> None:
    penalty, reason = get_domain_applicability("COC(=O)OC")
    assert 0.70 <= penalty <= 1.0
    assert isinstance(reason, str) and len(reason) > 0


def test_predict_mixture_binary_ideal_mixing() -> None:
    """Binary dielectric must equal the volume-fraction weighted average."""
    mix = predict_mixture(["C1COC(=O)O1", "COCCOC"], [0.5, 0.5])
    d1 = mix["component_results"][0]["dielectric_proxy"]
    d2 = mix["component_results"][1]["dielectric_proxy"]
    expected = 0.5 * d1 + 0.5 * d2
    assert mix["mixture_properties"]["dielectric_proxy"] == pytest.approx(expected, abs=1e-3)


def test_predict_mixture_ternary_ideal_mixing() -> None:
    """Ternary dielectric/Li+ solvation must match the fraction-weighted mean."""
    mix = predict_mixture(
        ["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C"],
        [0.5, 0.3, 0.2],
    )
    fracs = [0.5, 0.3, 0.2]
    expected_d = sum(
        r["dielectric_proxy"] * f
        for r, f in zip(mix["component_results"], fracs, strict=False)
    )
    expected_ls = sum(
        r["li_solvation_proxy"] * f
        for r, f in zip(mix["component_results"], fracs, strict=False)
    )
    assert mix["mixture_properties"]["dielectric_proxy"] == pytest.approx(expected_d, abs=1e-3)
    assert mix["mixture_properties"]["li_solvation_proxy"] == pytest.approx(expected_ls, abs=1e-3)


def test_predict_mixture_matches_pipeline_screen_mixture() -> None:
    """For a binary mixture the API must agree with pipeline.screen_mixture()."""
    pipeline = AureliusPipeline()
    pipeline.initialize()
    ctx_ec = MoleculeContext.from_smiles("C1COC(=O)O1")
    ctx_dme = MoleculeContext.from_smiles("COCCOC")
    assert ctx_ec is not None and ctx_dme is not None

    direct = pipeline.screen_mixture(ctx_ec, ctx_dme, 0.5)
    api = predict_mixture(["C1COC(=O)O1", "COCCOC"], [0.5, 0.5])

    assert api["mixture_properties"]["dielectric_proxy"] == pytest.approx(
        direct["mixture_properties"]["dielectric_proxy"], abs=1e-3
    )
    assert api["mixture_properties"]["viscosity_proxy"] == pytest.approx(
        direct["mixture_properties"]["viscosity_proxy"], abs=1e-3
    )
    assert api["score"]["total_score"] == pytest.approx(
        direct["score"]["total_score"], abs=1e-3
    )


def test_predict_mixture_validates_inputs() -> None:
    with pytest.raises(ValueError):
        predict_mixture(["C1COC(=O)O1"], [1.0])
    with pytest.raises(ValueError):
        predict_mixture(["C1COC(=O)O1", "COCCOC"], [0.5])
    with pytest.raises(ValueError):
        predict_mixture(["C1COC(=O)O1", "COCCOC"], [0.8, 0.8])
    with pytest.raises(ValueError):
        predict_mixture(["C1COC(=O)O1", "invalid_smiles!!"], [0.5, 0.5])


def test_predict_mixture_evaluates_every_component() -> None:
    """Each component must be individually screened (anti-gaming gate)."""
    mix = predict_mixture(["C1COC(=O)O1", "COCCOC"], [0.3, 0.7])
    assert len(mix["component_results"]) == 2
    for res in mix["component_results"]:
        assert "total_score" in res
        assert "homo_eV" in res
