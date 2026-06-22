"""Tests for active learning UCB selection strategy."""

from __future__ import annotations

from aurelius.agent.selection import select_for_active_learning
from aurelius.types import MoleculeContext


class TestSelectForActiveLearning:
    """Tests for select_for_active_learning UCB acquisition."""

    def test_ucb_prioritizes_high_uncertainty(self):
        """A low-score high-uncertainty candidate should be selected over
        a high-score low-uncertainty candidate when beta is high."""
        smiles = ["CCO", "CCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [90.0, 50.0]
        uncertainties = [0.01, 0.99]

        selected = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=1, beta=100.0,
        )

        assert len(selected) == 1
        assert selected[0].smiles == "CCCO"

    def test_normalization_bounds(self):
        """Min-max normalised uncertainties must be in [0, 1] for all cases."""
        from aurelius.agent.selection import _normalize_uncertainties

        cases: list[list[float]] = [
            [0.1, 0.5, 0.9],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 100.0],
            [0.001, 0.002, 0.003],
        ]
        for unc in cases:
            norm = _normalize_uncertainties(unc)
            assert len(norm) == len(unc)
            for v in norm:
                assert 0.0 <= v <= 1.0, f"Value {v} out of bounds for input {unc}"

    def test_batch_size_respected(self):
        """Output list length must match batch_size (or total if less)."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [10.0, 20.0, 30.0]
        uncertainties = [0.1, 0.2, 0.3]

        selected_1 = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=2,
        )
        assert len(selected_1) == 2

        selected_2 = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=5,
        )
        assert len(selected_2) == 3

    def test_empty_input_returns_empty(self):
        """Empty context or score lists should return empty list."""
        result = select_for_active_learning([], [], [], batch_size=5)
        assert result == []

    def test_deduplicates_by_smiles(self):
        """Duplicate SMILES should appear only once, keeping highest UCB."""
        smiles = ["CCO", "CCO", "CCCO", "CCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [80.0, 70.0, 60.0, 50.0]
        uncertainties = [0.1, 0.9, 0.2, 0.8]

        selected = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=4, beta=1.0,
        )
        selected_smiles = [c.smiles for c in selected]
        assert selected_smiles.count("CCO") == 1
        assert selected_smiles.count("CCCO") == 1

    def test_beta_zero_falls_back_to_raw_score(self):
        """With beta=0, selection should be identical to raw score ranking."""
        smiles = ["CCO", "CCCO", "CCCCO", "CCCCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [10.0, 50.0, 80.0, 90.0]
        uncertainties = [0.9, 0.8, 0.1, 0.05]

        selected = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=2, beta=0.0,
        )

        assert selected[0].smiles == "CCCCCCO"
        assert selected[1].smiles == "CCCCO"

    def test_ucb_score_monotonic_with_uncertainty(self):
        """For equal raw scores, higher uncertainty yields higher UCB rank."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [50.0, 50.0, 50.0]
        uncertainties = [0.1, 0.5, 0.9]

        selected = select_for_active_learning(
            contexts, scores, uncertainties, batch_size=3, beta=1.0,
        )

        assert selected[0].smiles == "CCCCO"
        assert selected[2].smiles == "CCO"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_context(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"Could not parse SMILES: {smiles}"
    return ctx
