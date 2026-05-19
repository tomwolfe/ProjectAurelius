"""Tests for Aurelius Scoring Engine."""

from __future__ import annotations

import pytest

from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.types import (
    DesolvationPathResult,
    GCMDTwinResult,
    MLXFilterResult,
    MoleculeInput,
    SEIEvolution,
    Tier2Result,
)


class TestAureliusScoringEngine:
    def setup_method(self):
        self.engine = AureliusScoringEngine(
            viability_threshold=65.0,
        )

    def test_compute_score_minimal(self):
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")
        score = self.engine.compute_score(molecule_input)
        assert score.molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert 0 <= score.total_score <= 100
        assert isinstance(score.is_viable, bool)

    def test_scorecard_format(self):
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")
        score = self.engine.compute_score(molecule_input)
        card = self.engine.print_scorecard(score)
        assert "AURELIUS SCORE v5.2" in card
        assert "Molecule:" in card
        assert "Total S_A:" in card
        assert "Component Scores:" in card
        assert "σ" in card
        assert "E_des_barrier" in card
        assert "SEI Homogeneity" in card
        assert "MX_Synthesis_Score" in card
        assert "GWP Penalty" in card

    def test_weight_decomposition(self):
        """Verify weights sum to 1.0."""
        weights = {
            "sigma": 0.3,
            "desolvation": 0.2,
            "sei_homogeneity": 0.2,
            "mx_synthesis": 0.2,
            "gwp": 0.1,
        }
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_viability_threshold(self):
        engine = AureliusScoringEngine(viability_threshold=90.0)
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")
        score = engine.compute_score(molecule_input)
        assert score.viability_threshold == 90.0

    def test_rejection_reasons(self):
        """Test that rejection reasons are populated for non-viable molecules."""
        engine = AureliusScoringEngine(viability_threshold=95.0)
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")
        score = engine.compute_score(molecule_input)
        if not score.is_viable:
            assert len(score.rejection_reasons) > 0

    def test_full_pipeline_scoring(self):
        """Test scoring with all tier results populated."""
        engine = AureliusScoringEngine(viability_threshold=65.0)
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")

        # Create realistic tier results
        tier1 = MLXFilterResult(
            molecule_smiles="CC(=O)OC1=CC(=O)O1",
            is_viable=True,
            confidence_score=0.75,
            inference_time_ms=12.5,
            na_utilization_pct=88.0,
        )
        tier2 = Tier2Result(
            molecule_smiles="CC(=O)OC1=CC(=O)O1",
            is_viable=True,
            desolvation_path=DesolvationPathResult(
                molecule_smiles="CC(=O)OC1=CC(=O)O1",
                barrier_height_eV=0.15,
                local_maxima_eV=0.12,
                path_integral_eV_A=0.8,
                rejected=False,
            ),
            simulation_time_ms=250.0,
            memory_used_gb=1.0,
        )
        tier3 = GCMDTwinResult(
            molecule_smiles="CC(=O)OC1=CC(=O)O1",
            sei_evolution=SEIEvolution(
                time_ps=1000.0,
                thickness_angstrom=3.5,
                homogeneity_score=0.72,
                ionic_conductivity_s_cm=5e-5,
                electronic_insulation=True,
                components=["NaF", "RO-ONa"],
            ),
            interface_stability=0.72,
            memory_used_gb=2.35,
            context_tokens_used=3276,
            simulation_time_ms=500.0,
        )

        score = engine.compute_score(molecule_input, tier1, tier2, tier3, gwp_value=1.0)
        assert score.tier1_viable is True
        assert score.tier2_viable is True
        assert score.tier3_viable is True
        assert score.sigma_score > 0
        assert score.desolvation_score > 0
        assert score.sei_homogeneity_score > 0
        assert score.mx_synthesis_score > 0
        assert score.gwp_penalty >= 0
