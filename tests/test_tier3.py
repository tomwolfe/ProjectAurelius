"""Tests for GCMDigitalTwin (Tier 3)."""

from __future__ import annotations

from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig


class TestGCMDigitalTwin:
    def setup_method(self):
        self.twin = GCMDigitalTwin(
            gcmtwin_config=GCMDTConfig(max_simulation_steps=5000)
        )

    def test_simulate_sei_evolution(self):
        result = self.twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1",
            "ec:dmc",
            "NaPF6",
        )
        assert result.molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert result.sei_evolution.thickness_angstrom > 0
        assert 0 <= result.sei_evolution.homogeneity_score <= 1
        assert result.sei_evolution.components
        assert result.context_tokens_used == 5000  # max_simulation_steps

    def test_simulation_stats(self):
        stats = self.twin.get_simulation_stats()
        assert stats["max_simulation_steps"] == 5000
        assert stats["record_interval"] == 50
        assert stats["transport_limit_thickness_angstrom"] == 15.0
        assert stats["use_mass_transport_limitation"] is True

    def test_kmc_deterministic(self):
        """kMC simulation must produce deterministic results for same inputs."""
        smiles = "CC(=O)OC1=CC(=O)O1"
        results = [
            self.twin.simulate_sei_evolution(smiles, "ec:dmc", "NaPF6")
            for _ in range(3)
        ]
        thicknesses = [r.sei_evolution.thickness_angstrom for r in results]
        assert all(t == thicknesses[0] for t in thicknesses)
        homogeneities = [r.sei_evolution.homogeneity_score for r in results]
        assert all(h == homogeneities[0] for h in homogeneities)

    def test_voltage_dependent_growth(self):
        """Higher voltage should produce thicker SEI (faster reaction rates)."""
        result_low = self.twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6", voltage_cutoff=0.01
        )
        result_high = self.twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6", voltage_cutoff=0.1
        )
        # Higher voltage -> faster kinetics -> thicker SEI
        assert result_high.sei_evolution.thickness_angstrom >= result_low.sei_evolution.thickness_angstrom

    def test_sei_thickness_physically_plausible(self):
        """SEI thickness should be in realistic range (1-50 Angstroms)."""
        result = self.twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6"
        )
        thickness = result.sei_evolution.thickness_angstrom
        assert 1.0 <= thickness <= 50.0
