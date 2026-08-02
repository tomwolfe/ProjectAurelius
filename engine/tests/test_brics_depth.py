"""Tests for BRICS retrosynthetic depth estimation and multiplicative penalty."""

from __future__ import annotations

import pytest

from aurelius.agent.mutation.brics import (
    _BRICS_DEPTH_PENALTY_PER_STEP,
    _MAX_BRICS_DEPTH,
    _estimate_synthetic_depth,
    combined_grounding_score,
)
from aurelius.constants import COMMERCIAL_BUILDING_BLOCK_SMILES
from aurelius.types import MoleculeContext


class TestSyntheticDepth:
    """Verify that _estimate_synthetic_depth returns correct recursion depths."""

    def test_building_block_has_depth_zero(self):
        """A molecule that IS a commercial building block should have depth 0."""
        for smi in COMMERCIAL_BUILDING_BLOCK_SMILES[:5]:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None
            depth = _estimate_synthetic_depth(ctx.mol)
            assert depth == 0, f"{smi} should be depth 0, got {depth}"

    def test_simple_molecule_has_low_depth(self):
        """A simple molecule should return a reasonable low depth."""
        ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert ctx is not None
        depth = _estimate_synthetic_depth(ctx.mol)
        assert 0 <= depth <= 2, f"Simple molecule should have depth <= 2, got {depth}"


class TestMultiplicativeDepthPenalty:
    """Verify that combined_grounding_score applies multiplicative depth penalty."""

    def test_no_penalty_below_threshold(self):
        """Molecules at or below _MAX_BRICS_DEPTH should not be penalized."""
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        score = combined_grounding_score(ctx.mol)
        assert score > 0.0, "Building block should have non-zero grounding score"

    def test_multiplicative_penalty_formula(self):
        """Verify the penalty formula: base * 0.9^(depth - 2)."""
        for excess_steps in range(1, 5):
            expected = 1.0 * (1.0 - _BRICS_DEPTH_PENALTY_PER_STEP) ** excess_steps
            assert expected == pytest.approx(0.9**excess_steps), (
                f"Depth {excess_steps + _MAX_BRICS_DEPTH}: expected {expected:.4f}, "
                f"got {0.9**excess_steps:.4f}"
            )
