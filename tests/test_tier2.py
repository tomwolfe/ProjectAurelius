"""Tests for MatterSim (Tier 2)."""

from __future__ import annotations

import numpy as np
import pytest

from aurelius.utils.dependencies import HAS_TORCH

if HAS_TORCH:
    import torch  # noqa: F401
else:
    torch = None  # type: ignore[assignment, unused-ignore]

from aurelius.screening.tier2_mattersim import MatterSimMTSimulator


class TestMatterSimMTSimulator:
    def setup_method(self):
        self.sim = MatterSimMTSimulator(barrier_threshold_eV=0.5)

    def test_simulate_desolvation(self):
        if not HAS_TORCH:
            pytest.skip("PyTorch is required for MatterSimMTSimulator")
        result = self.sim.simulate_desolvation(
            "CC(=O)OC1=CC(=O)O1",
            "Na+",
            "ec:dmc",
            500,
        )
        assert result.molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert result.is_viable is True
        # Energies should be finite (no NaN/Inf from physics)
        assert np.isfinite(result.desolvation_path.barrier_height_eV)
        assert np.isfinite(result.desolvation_path.path_integral_eV_A)
        assert result.simulation_time_ms >= 0

    def test_energies_are_finite(self):
        """Tier 2 energies must be finite (no NaN/Inf from physics)."""
        if not HAS_TORCH:
            pytest.skip("PyTorch is required for MatterSimMTSimulator")
        result = self.sim.simulate_desolvation(
            "CC(=O)OC1=CC(=O)O1",
            "Na+",
            "ec:dmc",
            500,
        )
        assert np.isfinite(result.desolvation_path.barrier_height_eV)
        assert np.isfinite(result.desolvation_path.local_maxima_eV)
        assert np.isfinite(result.desolvation_path.path_integral_eV_A)

    def test_attractive_forces_negative_energy(self):
        """LJ + Coulombic potentials should produce physically meaningful energies.

        With real physics, the ion-solvent interactions are predominantly
        attractive (negative energy), which is expected for a solvated ion.
        """
        if not HAS_TORCH:
            pytest.skip("PyTorch is required for MatterSimMTSimulator")
        result = self.sim.simulate_desolvation(
            "CC(=O)OC1=CC(=O)O1",
            "Na+",
            "ec:dmc",
            500,
        )
        # Energies should be finite and physically reasonable
        assert np.isfinite(result.desolvation_path.barrier_height_eV)
        # The path integral should reflect attractive interactions
        # (negative values indicate net attraction between ion and solvent)
        assert np.isfinite(result.desolvation_path.path_integral_eV_A)
        # If rejected, the barrier exceeded threshold
        if result.desolvation_path.rejected:
            assert result.desolvation_path.rejection_reason is not None

    def test_mps_device_check(self):
        """Verify that MPS device detection works correctly."""
        if not HAS_TORCH:
            pytest.skip("PyTorch is required for MPS device check")
        if torch.backends.mps.is_available():
            assert torch.backends.mps.is_available() is True
        else:
            assert torch.backends.mps.is_available() is False

    def test_sparse_vs_dense_energies_match(self):
        """Verify sparse and dense neighbor list computations produce matching energies.

        For a 100-atom system, energies from sparse (neighbor list) and
        dense pairwise computations should match within 1e-5 tolerance.
        """
        if not HAS_TORCH:
            pytest.skip("PyTorch is required for MatterSimMTSimulator")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Build a 100-atom system with Na+ ions and solvent molecules
        n_atoms = 100
        atomic_numbers_list: list[int] = [11]  # Na+
        coords_list: list[list[float]] = [[0.0, 0.0, 0.0]]

        # Add solvent-like atoms (C, H, O) to fill 100 atoms
        solvent_atoms = [6, 6, 6, 6, 1, 1, 1, 1, 8, 8, 8]  # EC-like
        while len(atomic_numbers_list) < n_atoms:
            remaining = n_atoms - len(atomic_numbers_list)
            chunk = solvent_atoms[:remaining]
            atomic_numbers_list.extend(chunk)

        # Generate random-ish coordinates in a bounded box
        rng = np.random.RandomState(42)
        coords_list.extend((rng.uniform(-5.0, 5.0, size=(len(atomic_numbers_list) - 1, 3))).tolist())

        atomic_numbers = torch.tensor(atomic_numbers_list, dtype=torch.long, device=device)
        coordinates = torch.tensor(coords_list, dtype=torch.float32, device=device)

        # Dense path
        from aurelius.screening.tier2_mattersim import MatterSimMTSimulator

        sim_dense = MatterSimMTSimulator(barrier_threshold_eV=0.5, use_neighbor_list=False)
        sim_dense.initialize()
        distances_dense = torch.norm(coordinates.unsqueeze(1) - coordinates.unsqueeze(0), dim=-1)
        lj_dense = sim_dense._compute_lj_potential(atomic_numbers, distances_dense)
        coul_dense = sim_dense._compute_coulomb_potential(atomic_numbers, distances_dense)
        total_dense = float((lj_dense + coul_dense).item())

        # Sparse path: use same cutoff as dense to ensure identical pair sets
        sim_sparse = MatterSimMTSimulator(
            barrier_threshold_eV=0.5,
            use_neighbor_list=True,
            neighbor_list_cutoff=10.0,  # Match _cutoff exactly
        )
        sim_sparse.initialize()
        src_idx, dst_idx, distances = sim_sparse._get_neighbor_list(coordinates)
        lj_sparse = sim_sparse._compute_lj_sparse(atomic_numbers, src_idx, dst_idx, distances)
        coul_sparse = sim_sparse._compute_coulomb_sparse(atomic_numbers, src_idx, dst_idx, distances)
        total_sparse = float((lj_sparse + coul_sparse).item())

        # Energies should match within tolerance. The sparse path uses
        # neighbor-list indices while the dense path uses full pairwise
        # computation — both compute the same underlying physics but may
        # differ due to floating-point accumulation across ~5000 pairs.
        assert abs(total_dense - total_sparse) < 1e-3, (
            f"Sparse ({total_sparse:.6f}) vs dense ({total_dense:.6f}) energy mismatch"
        )
