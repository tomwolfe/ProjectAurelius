"""Tests for XAI Lite — fragment-level rejection reasons.

Verifies that the _build_rejection_reasons method includes the top
contributing fragments in rejection reason strings.
"""

from __future__ import annotations

import pytest

from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext


@pytest.fixture(scope="module")
def pipeline() -> AureliusPipeline:
    pl = AureliusPipeline()
    pl.initialize()
    return pl


def _ctx(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"Failed to parse SMILES: {smiles}"
    return ctx


def test_ether_fragments_in_rejection(pipeline: AureliusPipeline) -> None:
    """A rejected molecule with ether groups should include 'ether'
    among the top contributors in the rejection reason.
    """
    result = pipeline.screen_molecule(_ctx("CCOCC"))
    score = result.get("score", {})
    reasons = score.get("rejection_reasons", [])
    joined = " ".join(reasons).lower()
    assert not score.get("is_viable", True), "Molecule should be rejected"
    assert "ether" in joined, (
        f"Expected rejection reason to mention 'ether', got: {reasons}"
    )


def test_dielectric_fragments_in_rejection(pipeline: AureliusPipeline) -> None:
    """A rejected molecule with low dielectric should include
    'dielectric' and at least one fragment name in the reason.
    """
    result = pipeline.screen_molecule(_ctx("CCOCC"))
    score = result.get("score", {})
    reasons = score.get("rejection_reasons", [])
    joined = " ".join(reasons).lower()
    assert not score.get("is_viable", True), "Molecule should be rejected"
    assert "dielectric" in joined, (
        f"Expected rejection reason to mention 'dielectric', got: {reasons}"
    )
    assert "ether" in joined, (
        f"Expected rejection reason to mention fragment 'ether', got: {reasons}"
    )


def test_rejection_format_includes_contributors(pipeline: AureliusPipeline) -> None:
    """Rejection reasons with fragment insights should include
    'Top contributors:' in their text.
    """
    result = pipeline.screen_molecule(_ctx("CCOCC"))
    score = result.get("score", {})
    reasons = score.get("rejection_reasons", [])
    joined = " ".join(reasons)
    assert "Top contributors:" in joined, (
        f"Expected 'Top contributors:' in rejection, got: {reasons}"
    )


def test_viable_molecule_has_no_rejection(pipeline: AureliusPipeline) -> None:
    """A viable molecule (ethylene carbonate) should have no rejection reasons."""
    result = pipeline.screen_molecule(_ctx("C1COC(=O)O1"))
    score = result.get("score", {})
    reasons = score.get("rejection_reasons", [])
    assert score.get("is_viable", False), "EC should be viable"
    assert len(reasons) == 0, (
        f"Viable molecule should have no rejection reasons, got: {reasons}"
    )
