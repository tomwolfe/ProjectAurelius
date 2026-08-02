"""End-to-end smoke tests for the Aurelius pipeline.

These tests verify that the core scoring workflows produce
valid, non-zero results end-to-end. They are lightweight enough
to run in CI on every commit.
"""

from __future__ import annotations

from aurelius.pipeline import AureliusPipeline


def test_screen_ec():
    """A basic electrolyte SMILES should produce a positive score."""
    p = AureliusPipeline()
    p.initialize()
    r = p.screen_smiles("C1COC(=O)O1")
    assert r["score"]["total_score"] > 0


def test_screen_li_binding_contributes():
    """The li_binding_energy_kcal objective should contribute to the score."""
    p = AureliusPipeline()
    p.initialize()
    r = p.screen_smiles("C1COC(=O)O1")
    sub_scores = r["score"]["sub_scores"]
    assert "li_binding_reward" in sub_scores
    assert sub_scores["li_binding_reward"] > 0


def test_pipeline_screen_molecule_returns_valid_score():
    """Regression (Step 2): screen_molecule must return a valid score for EC."""
    p = AureliusPipeline()
    p.initialize()
    r = p.screen_smiles("C1COC(=O)O1")
    assert r.get("tier2") is not None
    assert "score" in r
    assert r["score"]["total_score"] >= 0
    assert r["score"]["total_score"] > 0
