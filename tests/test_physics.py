"""Tests for physics validation (Arrhenius behavior and energy conservation)."""

from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

try:
    from aurelius.screening.tier2_mattersim import MatterSimMPEngine
except ImportError:
    MatterSimMPEngine = None  # type: ignore[misc, assignment]

from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin


class TestArrheniusBehavior:
    """Tests verifying Arrhenius equation behavior in Tier 3 kMC."""

    def test_increasing_temperature_increases_rate(self):
        """Verify that increasing temperature increases reaction rates.

        The Arrhenius equation k = A * exp(-Ea/(kB*T)) predicts that
        reaction rates increase exponentially with temperature.
        """
        twin = GCMDigitalTwin()

        # Run simulations at different temperatures
        result_250k = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6", voltage_cutoff=0.05, temperature_k=250.0
        )
        result_298k = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6", voltage_cutoff=0.05, temperature_k=298.15
        )
        result_350k = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6", voltage_cutoff=0.05, temperature_k=350.0
        )

        # Higher temperature -> faster kinetics -> thicker SEI
        thickness_250 = result_250k.sei_evolution.thickness_angstrom
        thickness_298 = result_298k.sei_evolution.thickness_angstrom
        thickness_350 = result_350k.sei_evolution.thickness_angstrom

        assert thickness_250 <= thickness_298 <= thickness_350, (
            f"SEI thickness should increase with temperature: "
            f"{thickness_250:.2f} <= {thickness_298:.2f} <= {thickness_350:.2f}"
        )

    def test_arrhenius_rate_formula(self):
        """Verify the Arrhenius rate formula produces physically correct behavior."""
        twin = GCMDigitalTwin()

        temperature = 298.15
        concentration = 1.0
        overpotential = 0.05

        # Compute rate at different activation energies
        k_low_ea = twin._arrhenius_rate(
            activation_energy_eV=0.50,
            temperature_k=temperature,
            concentration=concentration,
            pre_exponential_base=5.0,
            overpotential_V=overpotential,
        )
        k_high_ea = twin._arrhenius_rate(
            activation_energy_eV=1.20,
            temperature_k=temperature,
            concentration=concentration,
            pre_exponential_base=5.0,
            overpotential_V=overpotential,
        )

        # Lower activation energy -> higher rate
        assert k_low_ea > k_high_ea, f"Lower Ea should give higher rate: {k_low_ea} > {k_high_ea}"

        # Both rates should be positive
        assert k_low_ea > 0
        assert k_high_ea > 0

    def test_concentration_dependent_pre_exponential(self):
        """Verify that pre-exponential factor decreases with lower concentration.

        As SEI grows, solvent concentration at the interface decreases,
        reducing the reaction rate through mass transport limitation.
        """
        twin = GCMDigitalTwin()

        temperature = 298.15
        overpotential = 0.05

        k_full = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=temperature,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=overpotential,
        )
        k_half = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=temperature,
            concentration=0.3,
            pre_exponential_base=5.0,
            overpotential_V=overpotential,
        )
        k_low = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=temperature,
            concentration=0.05,
            pre_exponential_base=5.0,
            overpotential_V=overpotential,
        )

        # Rate should decrease as concentration drops
        assert k_full > k_half > k_low, (
            f"Rate should decrease with concentration: {k_full:.4f} > {k_half:.4f} > {k_low:.4f}"
        )


class TestPhysicsConservation:
    """Tests verifying physics engine reads coordinate tensor properly."""

    def test_energy_changes_on_coordinate_perturbation(self):
        """Verify that perturbing coordinates changes the computed energy.

        This test ensures the physics engine is actually reading the
        coordinate tensor and not just computing a static value based
        on atomic numbers alone.
        """
        if not HAS_TORCH or MatterSimMPEngine is None:
            pytest.skip("PyTorch or MatterSimMPEngine not available")

        # Build a simple water-like cluster (H2O-like: O at center, H on sides)
        atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Oxygen at origin
                [0.96, 0.0, 0.0],  # H1 at 0.96 A
                [-0.96, 0.0, 0.0],  # H2 at -0.96 A
            ],
            dtype=torch.float32,
        )

        engine = MatterSimMPEngine()
        baseline_energy = engine(atomic_numbers, coordinates)

        # Add small Gaussian noise to coordinates
        noise = torch.randn_like(coordinates) * 0.01
        perturbed = coordinates + noise

        energy_perturbed = engine(atomic_numbers, perturbed)

        # The energy must change when coordinates change
        assert abs(energy_perturbed - baseline_energy) > 1e-6, (
            f"Energy should change with coordinate perturbation: "
            f"baseline={baseline_energy.item():.6f}, perturbed={energy_perturbed.item():.6f}"
        )
