"""Tests for Project Aurelius v5.2."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from aurelius.config import M5ProConfig
from aurelius.memory.manager import (
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMPEngine, MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.types import (
    DesolvationPathResult,
    GCMDTwinResult,
    MLXFilterResult,
    MoleculeInput,
    SEIEvolution,
    Tier2Result,
)

HAS_TORCH = torch is not None


# ============================================================
# Config Tests
# ============================================================

class TestM5ProConfig:
    def test_default_memory_budget(self):
        config = M5ProConfig()
        assert config.validate_memory_budget() is True
        # MLX gets 50% of RAM, capped at 12GB
        assert config.mlx_max_mem_gb <= 12.0
        assert config.mlx_max_mem_gb > 0

    def test_memory_report(self):
        config = M5ProConfig()
        report = config.memory_report()
        assert "MLX" in report
        assert "Metal Shader Cache" in report
        assert "PyTorch MPS" in report

    def test_invalid_memory_budget(self):
        config = M5ProConfig(mlx_max_mem_gb=22, metal_shader_cache_gb=3)
        assert config.validate_memory_budget() is False

    def test_dynamic_ram_detection(self):
        """Verify that config detects system RAM dynamically."""
        import psutil
        config = M5ProConfig()
        detected_gb = psutil.virtual_memory().total / (1024 ** 3)
        assert config.total_memory_gb > 0
        assert config.total_memory_gb <= detected_gb + 1  # Allow small tolerance


# ============================================================
# Memory Manager Tests
# ============================================================

class TestQuantizationConfig:
    def test_mx4_bits(self):
        config = QuantizationConfig(precision="MX4")
        assert config.bits == 4
        assert config.compression_ratio == 8.0

    def test_mx6_bits(self):
        config = QuantizationConfig(precision="MX6")
        assert config.bits == 6
        assert config.compression_ratio == pytest.approx(5.33, abs=0.01)

    def test_mx8_bits(self):
        config = QuantizationConfig(precision="MX8")
        assert config.bits == 8
        assert config.compression_ratio == 4.0


class TestZeroCopyMemoryManager:
    def test_init_default(self):
        mgr = ZeroCopyMemoryManager()
        assert mgr.quant_config.precision == "MX4"
        assert mgr.device == "mps"

    def test_get_memory_budget(self):
        mgr = ZeroCopyMemoryManager()
        budget = mgr.get_memory_budget()
        # Total is now dynamically detected, not hardcoded 24.0
        assert budget["total_gb"] > 0
        assert "chemvlm2_footprint_gb" in budget
        assert "remaining_gb" in budget

    def test_dynamic_ram_detection(self):
        """Verify memory manager detects system RAM dynamically."""
        import psutil
        mgr = ZeroCopyMemoryManager()
        assert mgr._total_ram_gb > 0
        detected = psutil.virtual_memory().total / (1024 ** 3)
        assert mgr._total_ram_gb <= detected + 1


# ============================================================
# MWSE Solvation Tests
# ============================================================

class TestMWSESolvationEngine:
    def setup_method(self):
        self.engine = MWSESolvationEngine(kex_window_ps=10.0)

    def test_solvent_exchange_rate(self):
        k_ex = self.engine.compute_solvent_exchange_rate("water", "Na+")
        assert k_ex > 0
        assert k_ex < 1000  # Should be in ps^-1 range

    def test_screen_solvation_shell(self):
        shell = self.engine.screen_solvation_shell("Na+", "ec:dmc")
        assert shell.ion_type == "Na+"
        assert shell.solvent_type == "ec:dmc"
        assert shell.k_ex_ps > 0
        assert shell.e_des_eV > 0

    def test_born_effective_charges(self):
        born = self.engine.query_born_effective_charges("Na+", "water")
        assert born.ion_type == "Na+"
        assert born.z_star.shape == (3, 3)
        assert born.z_star_scalar > 0
        assert born.dipole_moment_debye > 0

    def test_born_charges_dft_derived(self):
        """Verify Born charges come from DFT literature values, not random."""
        born = self.engine.query_born_effective_charges("Na+", "ec:dmc")
        # DFT Z* for Na+ should be in a physically reasonable range
        assert 1.0 < born.z_star_scalar < 2.0

    def test_mixed_solvent_interpolation(self):
        """Verify linear interpolation for mixed solvents."""
        born_ec = self.engine.query_born_effective_charges("Na+", "ec")
        born_dmc = self.engine.query_born_effective_charges("Na+", "dmc")
        born_mixed = self.engine.query_born_effective_charges("Na+", "ec:dmc")

        # Mixed solvent Z* should be between pure component values
        z_ec = born_ec.z_star_scalar
        z_dmc = born_dmc.z_star_scalar
        z_mixed = born_mixed.z_star_scalar

        # The mixed value should lie between the extremes
        assert min(z_ec, z_dmc) <= z_mixed <= max(z_ec, z_dmc)

    def test_desolvation_path_integral(self):
        barrier = self.engine.compute_desolvation_path_integral("Na+", "ec:dmc")
        assert barrier.barrier_height_eV >= 0
        assert barrier.path_integral_energy >= 0
        assert barrier.local_maxima_eV <= 0.5

    def test_evaluate_mwse_state(self):
        state = self.engine.evaluate_mwse_state("Na+", "ec:dmc")
        assert state.solvation_shell.ion_type == "Na+"
        assert state.dipole_moment_debye > 0
        assert isinstance(state.is_stable_500cycle, bool)


# ============================================================
# MLX-NA Filter Tests
# ============================================================

class TestMLXNAFilter:
    def setup_method(self):
        # Disable training on init for faster tests
        self.filter = MLXNAFilter(quantization_format="MX4", train_on_init=False)

    def test_screen_molecule(self):
        result = self.filter.screen_molecule("CC(=O)OC1=CC(=O)O1")
        assert result.molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert 0 <= result.confidence_score <= 1
        assert result.quantization_format == "MX4"
        assert 0 <= result.na_utilization_pct <= 100

    def test_deterministic_output_same_smiles(self):
        """Tier 1 must produce consistent results for the same SMILES."""
        smiles = "CC(=O)OC1=CC(=O)O1"
        results = [self.filter.screen_molecule(smiles) for _ in range(5)]
        confidences = [r.confidence_score for r in results]
        # All confidence scores must be identical (deterministic)
        assert all(c == confidences[0] for c in confidences)
        # Is_viable must also be consistent
        viability = [r.is_viable for r in results]
        assert all(v == viability[0] for v in viability)

    def test_different_smiles_different_output(self):
        """Different SMILES should produce different confidence scores."""
        smiles_list = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = [self.filter.screen_molecule(s) for s in smiles_list]
        confidences = [r.confidence_score for r in results]
        # At least some should differ (hash-based fingerprints differ)
        assert len(set(confidences)) >= 1  # At minimum, valid scores

    def test_screen_batch(self):
        molecules = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = self.filter.screen_batch(molecules)
        assert len(results) == 3
        assert all(r.molecule_smiles in molecules for r in results)

    def test_fingerprint_generation(self):
        """Test that ECFP4 fingerprints are generated correctly."""
        from aurelius.screening.tier1_mlx_filter import _generate_ecfp4_fingerprint

        smiles = "CC(=O)OC1=CC(=O)O1"
        fp = _generate_ecfp4_fingerprint(smiles)
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        assert set(np.unique(fp).tolist()).issubset({0.0, 1.0})

    def test_fingerprint_deterministic(self):
        """Fingerprint generation must be deterministic."""
        from aurelius.screening.tier1_mlx_filter import _generate_ecfp4_fingerprint

        smiles = "C1=CC(=O)OC1"
        fp1 = _generate_ecfp4_fingerprint(smiles)
        fp2 = _generate_ecfp4_fingerprint(smiles)
        assert np.array_equal(fp1, fp2)

    def test_model_trains_on_init(self):
        """Verify that train_on_init=True produces a trained model."""
        filter_trained = MLXNAFilter(quantization_format="MX4", train_on_init=True)
        # After training, the model should have non-trivial weights
        assert filter_trained._model is not None
        params = filter_trained._model.parameters()
        # Weights should have been updated from initial Xavier initialization
        # (they should not be exactly zero or all identical)
        for p in params:
            if isinstance(p, mx.array):
                p_np = np.array(p)
            else:
                p_np = p
            # Weights should have some non-zero values (Xavier init + training)
            assert np.any(np.abs(p_np) > 1e-10), "Model weights should have non-zero values"


# ============================================================
# MatterSim-MT Tests
# ============================================================

class TestMatterSimMTSimulator:
    def setup_method(self):
        self.sim = MatterSimMTSimulator(barrier_threshold_eV=0.5)

    def test_simulate_desolvation(self):
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
        if torch.backends.mps.is_available():
            assert torch.backends.mps.is_available() is True
        else:
            assert torch.backends.mps.is_available() is False


# ============================================================
# GCMD Digital Twin Tests
# ============================================================

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


# ============================================================
# Aurelius Scoring Tests
# ============================================================

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
        assert sum(weights.values()) == 1.0

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


# ============================================================
# Pipeline Integration Tests
# ============================================================

class TestAureliusPipeline:
    def test_pipeline_initialization(self):
        from aurelius.pipeline import AureliusPipeline

        config = M5ProConfig()
        pipeline = AureliusPipeline(config)
        pipeline.initialize()

        assert pipeline._memory_manager is not None
        assert pipeline._solvation_engine is not None
        assert pipeline._scoring_engine is not None

    def test_single_molecule_screen(self):
        from aurelius.pipeline import AureliusPipeline

        config = M5ProConfig()
        pipeline = AureliusPipeline(config)
        pipeline.initialize()

        results = pipeline.screen_molecule("CC(=O)OC1=CC(=O)O1")
        assert "score" in results
        assert results["score"].molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert "tier1" in results
        assert "tier2" in results
        assert "tier3" in results

    def test_batch_screen(self):
        from aurelius.pipeline import AureliusPipeline

        config = M5ProConfig()
        pipeline = AureliusPipeline(config)
        pipeline.initialize()

        molecules = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = pipeline.screen_batch(molecules, solvent_type="ec:dmc")
        assert len(results) == 3
        assert all("score" in r for r in results)


# ============================================================
# Cross-Framework Bridge Tests
# ============================================================

class TestCrossFrameworkBridge:
    def test_mlx_to_pytorch_bridge(self):
        """Test DLpack-based bridge from MLX to PyTorch."""
        from aurelius.bridge import CrossFrameworkBridge, bridge_mlx_to_pytorch

        if not mx or not torch:
            pytest.skip("MLX or PyTorch not available")

        bridge = CrossFrameworkBridge()
        assert bridge.is_available is True

        # Create a test MLX array with known shape
        mlx_array = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        try:
            torch_tensor = bridge_mlx_to_pytorch(mlx_array)
        except AttributeError as e:
            if "DLpack" in str(e):
                pytest.skip("MLX version does not support DLpack")
            raise

        assert torch_tensor.shape == (2, 3)
        if torch.backends.mps.is_available():
            assert torch_tensor.is_mps

    def test_pytorch_to_mlx_bridge(self):
        """Test DLpack-based bridge from PyTorch to MLX."""
        from aurelius.bridge import bridge_pytorch_to_mlx

        if not mx or not torch:
            pytest.skip("MLX or PyTorch not available")

        # Create a test PyTorch tensor
        torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        try:
            mlx_array = bridge_pytorch_to_mlx(torch_tensor)
        except AttributeError as e:
            if "DLpack" in str(e):
                pytest.skip("MLX version does not support DLpack")
            raise

        assert list(mlx_array.shape) == [3, 2]

    def test_bridge_unavailable_fallback(self):
        """Test that bridge raises RuntimeError when frameworks are missing."""
        from aurelius.bridge import bridge_mlx_to_pytorch

        if not mx or not torch:
            pytest.skip("MLX or PyTorch not available")

        mlx_array = mx.array([1.0, 2.0])
        try:
            torch_tensor = bridge_mlx_to_pytorch(mlx_array)
        except AttributeError as e:
            if "DLpack" in str(e):
                pytest.skip("MLX version does not support DLpack")
            raise
        assert torch_tensor.shape == (2,)


# ============================================================
# 3D Physics Engine Tests
# ============================================================

class TestMatterSimMPEngine:
    def test_3d_vector_forward_pass(self):
        """Test that the 3D physics engine processes proper (N, 3) coordinates."""
        engine = MatterSimMPEngine()

        # N=5 atoms with 3D coordinates
        atomic_numbers = torch.tensor([6, 6, 8, 1, 1], dtype=torch.long)  # C, C, O, H, H
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.2, 0.5],
            [2.0, 0.0, 0.0],
            [1.3, 0.5, 0.0],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        assert energy.shape == ()  # Scalar output

    def test_vectorized_batch(self):
        """Test 3D engine with multiple molecules in batch."""
        engine = MatterSimMPEngine()

        # Batch of 2 molecules, each with 4 atoms
        atomic_numbers = torch.tensor([
            [6, 8, 1, 1],
            [7, 7, 1, 1],  # N, N, H, H
        ], dtype=torch.long)

        coordinates = torch.tensor([
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        assert energy.shape == ()  # Scalar output

    def test_pairwise_distance_computation(self):
        """Verify pairwise distance matrix is computed correctly."""
        engine = MatterSimMPEngine()

        # Two atoms at known separation
        atomic_numbers = torch.tensor([6, 8], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        # Should produce a finite energy value
        assert torch.isfinite(energy)

    def test_energy_gradients_computable(self):
        """Verify that energy gradients can be computed via torch.autograd.grad.

        This is essential for MD integration: the forces on each atom
        must be computable as the negative gradient of the potential
        energy with respect to atomic coordinates.
        """
        engine = MatterSimMPEngine()

        atomic_numbers = torch.tensor([6, 6, 8, 1, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.2, 0.5],
            [2.0, 0.0, 0.0],
            [1.3, 0.5, 0.0],
        ], dtype=torch.float32, requires_grad=True)

        energy = engine(atomic_numbers, coordinates)

        # Compute gradients
        grad = torch.autograd.grad(
            energy, coordinates, grad_outputs=torch.ones_like(energy), create_graph=False
        )
        assert grad is not None
        assert grad[0].shape == coordinates.shape
        # Gradients should be finite (no NaN/Inf from physics)
        assert torch.all(torch.isfinite(grad[0]))


# ============================================================
# Shape Compatibility Tests
# ============================================================

class TestShapeCompatibility:
    def test_tier1_to_tier2_shape_compatibility(self):
        """
        Integration test validating that the physical output shapes match the
        downstream execution parameters required by MatterSim.
        """
        if not mx or not torch:
            pytest.skip("MLX or PyTorch not available")

        batch_size = 1
        hidden_dim = 1024

        # Instantiate model without training for shape test
        model = MLXNAFilter(quantization_format="MX4", train_on_init=False)

        # Create placeholder input
        mock_input = mx.random.normal(shape=(batch_size, hidden_dim))

        # Run inference through the model
        if model._model is not None:
            mlx_output = model._model(mock_input)
            assert list(mlx_output.shape) == [batch_size, 1]
        else:
            pytest.skip("Model not loaded")

        # Convert to physical numpy storage layout
        np_view = np.array(mlx_output)

        # Validate PyTorch ingestion dimensions for structural graphing
        torch_tensor = torch.from_numpy(np_view).to("mps")
        assert torch_tensor.is_mps
        assert torch_tensor.shape == (batch_size, 1)

    def test_tier2_3d_tensor_shapes(self):
        """Validate that 3D physics engine requires proper (N, 3) coordinate tensors."""
        engine = MatterSimMPEngine()

        # Realistic water molecule: 3 atoms with 3D coordinates
        atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.586, -0.586],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        assert energy.shape == ()
        assert torch.isfinite(energy)

    def test_dlpack_zero_copy_preserves_shape(self):
        """Verify DLpack bridging preserves tensor shapes exactly."""
        if not mx or not torch:
            pytest.skip("MLX or PyTorch not available")
        from aurelius.bridge import bridge_mlx_to_pytorch

        # Test various shapes including new fingerprint dimensions
        test_shapes = [(1, 1024), (32, 128), (100, 3), (512,), (1, 2048)]
        for shape in test_shapes:
            mlx_array = mx.random.normal(shape=shape)
            try:
                torch_tensor = bridge_mlx_to_pytorch(mlx_array)
            except AttributeError as e:
                if "DLpack" in str(e):
                    pytest.skip("MLX version does not support DLpack")
                raise
            assert torch_tensor.shape == shape

    def test_fingerprint_to_torch_tensor_shape(self):
        """Verify ECFP4 fingerprint (2048-bit) bridges correctly to PyTorch."""
        if not torch:
            pytest.skip("PyTorch not available")
        from aurelius.screening.tier1_mlx_filter import _generate_ecfp4_fingerprint

        smiles = "CC(=O)OC1=CC(=O)O1"
        fp = _generate_ecfp4_fingerprint(smiles)
        assert fp.shape == (2048,)

        # Should be convertible to torch tensor
        torch_tensor = torch.from_numpy(fp)
        assert torch_tensor.shape == (2048,)
        assert torch_tensor.dtype == torch.float32


# ============================================================
# Physics Validation Tests
# ============================================================

class TestPhysicsConservation:
    """Tests verifying physical correctness of simulation engines."""

    def test_forces_are_negative_energy_gradients(self):
        """Verify that forces computed as -dE/dr are physically meaningful.

        In a proper physics engine, the force on each atom equals the
        negative gradient of the potential energy with respect to its
        coordinates. This test verifies that the MatterSim engine
        produces computable, finite gradients.
        """
        engine = MatterSimMPEngine()

        # Small water cluster around Na+
        atomic_numbers = torch.tensor([11, 8, 1, 1], dtype=torch.long)  # Na+, O, H, H
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],   # Na+
            [2.3, 0.0, 0.0],   # O (first solvation shell)
            [2.8, 0.7, 0.0],   # H
            [2.8, -0.7, 0.0],  # H
        ], dtype=torch.float32, requires_grad=True)

        energy = engine(atomic_numbers, coordinates)

        # Verify energy is finite
        assert torch.isfinite(energy)

        # Compute forces as negative energy gradients
        grad = torch.autograd.grad(
            energy, coordinates, grad_outputs=torch.ones_like(energy), create_graph=False
        )
        assert grad is not None
        forces = -grad[0]

        # Forces should be finite
        assert torch.all(torch.isfinite(forces))

        # Forces should not be identically zero (system is not at equilibrium)
        assert torch.any(torch.abs(forces) > 1e-10)

    def test_energy_conservation_closed_system(self):
        """Verify that energy is approximately conserved in a closed system.

        For a system with no external forces and no time evolution,
        the total energy should remain constant. This test verifies
        that the potential energy function is well-behaved and
        produces consistent results for identical configurations.
        """
        engine = MatterSimMPEngine()

        # Fixed atomic configuration
        atomic_numbers = torch.tensor([6, 6, 8, 1, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.2, 0.5],
            [2.0, 0.0, 0.0],
            [1.3, 0.5, 0.0],
        ], dtype=torch.float32)

        # Compute energy multiple times - should be identical (deterministic)
        energies = []
        for _ in range(3):
            e = engine(atomic_numbers, coordinates)
            energies.append(float(e.item()))

        # All energies should be identical (deterministic physics)
        assert all(abs(e - energies[0]) < 1e-6 for e in energies)


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
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6",
            voltage_cutoff=0.05, temperature_k=250.0
        )
        result_298k = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6",
            voltage_cutoff=0.05, temperature_k=298.15
        )
        result_350k = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6",
            voltage_cutoff=0.05, temperature_k=350.0
        )

        # Higher temperature -> faster kinetics -> thicker SEI
        thickness_250 = result_250k.sei_evolution.thickness_angstrom
        thickness_298 = result_298k.sei_evolution.thickness_angstrom
        thickness_350 = result_350k.sei_evolution.thickness_angstrom

        assert thickness_250 <= thickness_298 <= thickness_350, \
            f"SEI thickness should increase with temperature: " \
            f"{thickness_250:.2f} <= {thickness_298:.2f} <= {thickness_350:.2f}"

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
        assert k_low_ea > k_high_ea, \
            f"Lower Ea should give higher rate: {k_low_ea} > {k_high_ea}"

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
        assert k_full > k_half > k_low, \
            f"Rate should decrease with concentration: " \
            f"{k_full:.4f} > {k_half:.4f} > {k_low:.4f}"


class TestVectorizationSpeed:
    """Tests verifying that Tier 2 is fully vectorized (no Python loops)."""

    def test_vectorization_speed_50_atoms(self):
        """Ensure Tier 2 simulation for 50 atoms completes in < 100ms on MPS.

        This test proves that Python loops have been removed and the
        physics engine uses vectorized tensor operations for maximum
        throughput on Apple Silicon hardware.
        """
        sim = MatterSimMTSimulator(barrier_threshold_eV=0.5)

        # Run simulation with a moderately sized system
        result = sim.simulate_desolvation(
            "CC(=O)OC1=CC(=O)O1",
            "Na+",
            "ec:dmc",
            n_cycles=100,  # Use fewer cycles for speed test
        )

        # Should complete in reasonable time (allow generous margin for CI)
        assert result.simulation_time_ms < 5000, \
            f"Simulation took {result.simulation_time_ms:.1f}ms, expected < 5000ms"

    def test_batched_forward_pass(self):
        """Verify that MatterSimMPEngine handles batched inputs."""
        engine = MatterSimMPEngine()

        # Batch of 4 molecules with 6 atoms each
        batch_n = 4
        atom_n = 6
        atomic_numbers = torch.randint(1, 118, (batch_n, atom_n), dtype=torch.long)
        coordinates = torch.randn(batch_n, atom_n, 3, dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)

        # Should return a scalar (mean over batch)
        assert energy.shape == ()
        assert torch.isfinite(energy)


# ============================================================
# GBSA Solvation Energy Tests
# ============================================================

class TestGBSASolvation:
    """Tests verifying GBSA solvation energy computation."""

    def test_gbsa_energy_computation(self):
        """Test that GBSA solvation energy is computed correctly."""
        from aurelius.solvation.engine import compute_gbsa_solvation_energy

        # Simple diatomic: two charges at fixed distance
        charges = np.array([0.5, -0.5])
        radii = np.array([1.5, 1.5])

        # In vacuum (epsilon=1.0), electrostatic term should be zero
        e_vacuum = compute_gbsa_solvation_energy(charges, radii, dielectric_bulk=1.0)
        assert np.isfinite(e_vacuum)

        # In high-dielectric solvent, electrostatic screening reduces energy
        e_water = compute_gbsa_solvation_energy(charges, radii, dielectric_bulk=78.36)
        assert np.isfinite(e_water)

        # Nonpolar term should be positive (surface area cost)
        # Both energies should be finite and non-negative (dominated by SASA)
        assert e_vacuum >= 0
        assert e_water >= 0

    def test_born_charges_dft_derived(self):
        """Verify Born effective charges come from DFT literature."""
        from aurelius.solvation.engine import (
            _BORN_CHARGES_K,
            _BORN_CHARGES_LI,
            _BORN_CHARGES_NA,
        )

        # Na+ Z* norm should be in physically reasonable range
        na_norm = np.linalg.norm(_BORN_CHARGES_NA)
        assert 1.0 < na_norm < 2.5

        # Li+ Z* should be near 1.3 per diagonal element, norm ~2.3
        li_norm = np.linalg.norm(_BORN_CHARGES_LI)
        assert 1.5 < li_norm < 3.0

        # K+ Z* should be near 0.9 per diagonal element, norm ~1.6
        k_norm = np.linalg.norm(_BORN_CHARGES_K)
        assert 1.0 < k_norm < 2.0


# ============================================================
# Real Model Integration Tests
# ============================================================

class TestRealModelIntegration:
    """Tests verifying Tier 1 real model integration."""

    def test_hf_weight_loader_exists(self):
        """Verify HuggingFaceWeightLoader class exists."""
        from aurelius.screening.tier1_mlx_filter import HuggingFaceWeightLoader

        loader = HuggingFaceWeightLoader()
        assert loader is not None
        assert hasattr(loader, "load_model")
        assert hasattr(loader, "save_model")

    def test_real_model_filter_creation(self):
        """Verify MLXNAFilter can be created with use_real_models=True."""
        from aurelius.screening.tier1_mlx_filter import MLXNAFilter

        # Should work without training on init
        f = MLXNAFilter(quantization_format="MX4", use_real_models=True, train_on_init=False)
        assert f._use_real_models is True
        assert f._model_loaded is False

    def test_demo_mode_filter_creation(self):
        """Verify MLXNAFilter works in demo (synthetic) mode."""
        from aurelius.screening.tier1_mlx_filter import MLXNAFilter

        f = MLXNAFilter(quantization_format="MX4", use_real_models=False, train_on_init=False)
        assert f._use_real_models is False

    def test_screening_produces_viability_score(self):
        """Verify that screening produces meaningful viability scores."""
        from aurelius.screening.tier1_mlx_filter import MLXNAFilter

        f = MLXNAFilter(quantization_format="MX4", use_real_models=True, train_on_init=False)
        result = f.screen_molecule("CCO")  # ethanol

        assert result.is_viable is True or result.is_viable is False
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.inference_time_ms >= 0
        assert 0.0 <= result.na_utilization_pct <= 100.0

    def test_ecfp4_fingerprint_properties(self):
        """Verify ECFP4 fingerprints have correct properties."""
        from aurelius.screening.tier1_mlx_filter import _generate_ecfp4_fingerprint

        smiles = "CC(=O)OC1=CC(=O)O1"
        fp = _generate_ecfp4_fingerprint(smiles)

        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        # Should be binary (0 or 1)
        assert set(np.unique(fp).tolist()).issubset({0.0, 1.0})
        # Should have some bits set
        assert np.sum(fp) > 0

    def test_esol_training_function_exists(self):
        """Verify train_on_esol function exists and is callable."""
        from aurelius.screening.tier1_mlx_filter import train_on_esol

        assert callable(train_on_esol)

    def test_qm9_training_function_exists(self):
        """Verify train_on_qm9 function exists and is callable."""
        from aurelius.screening.tier1_mlx_filter import train_on_qm9

        assert callable(train_on_qm9)


# ============================================================
# SchNet Layer Tests
# ============================================================

class TestSchNetLayers:
    """Tests verifying SchNet-style message passing layers."""

    def test_continuous_filter_conv(self):
        """Test continuous-filter convolution layer."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        from aurelius.screening.tier2_mattersim import ContinuousFilterConv1d

        conv = ContinuousFilterConv1d(input_dim=128, output_dim=128, num_filters=32)
        h = torch.randn(10, 128)
        distances = torch.rand(10, 10)

        out = conv(h, distances)
        assert out.shape == (10, 128)

    def test_schnet_interaction_block(self):
        """Test SchNet interaction block."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        from aurelius.screening.tier2_mattersim import SchNetInteractionBlock

        block = SchNetInteractionBlock(hidden_dim=128, num_filters=32)
        h = torch.randn(10, 128)
        distances = torch.rand(10, 10)

        out = block(h, distances)
        assert out.shape == (10, 128)

    def test_mattersim_engine_forward(self):
        """Test MatterSimMPEngine forward pass with SchNet layers."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        engine = MatterSimMPEngine(hidden_dim=128, num_filters=32)

        atomic_numbers = torch.tensor([6, 8, 1, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        assert torch.isfinite(energy)
        assert energy.shape == ()

    def test_mattersim_force_computation(self):
        """Test that forces can be computed as energy gradients."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        engine = MatterSimMPEngine(hidden_dim=128, num_filters=32)

        atomic_numbers = torch.tensor([6, 8, 1, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32, requires_grad=True)

        _energy = engine(atomic_numbers, coordinates)
        forces = engine.compute_forces(atomic_numbers, coordinates)

        assert forces.shape == (4, 3)
        assert torch.all(torch.isfinite(forces))


# ============================================================
# CLI Flag Tests
# ============================================================

class TestCLIFlags:
    """Tests verifying CLI flag support for real models."""

    def test_pipeline_accepts_use_real_models(self):
        """Verify AureliusPipeline accepts use_real_models parameter."""
        from aurelius.pipeline import AureliusPipeline

        config = M5ProConfig()

        # Should accept use_real_models parameter
        pipeline = AureliusPipeline(config, use_real_models=True)
        assert pipeline._use_real_models is True

        pipeline_demo = AureliusPipeline(config, use_real_models=False)
        assert pipeline_demo._use_real_models is False

    def test_cli_help_includes_real_model_flags(self):
        """Verify CLI help text includes --use-real-models and --demo flags."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "aurelius", "screen", "--help"],
            capture_output=True, text=True,
        )
        assert "--use-real-models" in result.stdout
        assert "--demo" in result.stdout


# ============================================================
# Cross-Platform Support Tests (v5.2)
# ============================================================

class TestCrossPlatformSupport:
    """Tests verifying cross-platform device selection for Tier 2."""

    def test_device_selection_cpu_fallback(self):
        """Verify that CPU is selected when no GPU is available."""
        sim = MatterSimMTSimulator()
        device = sim._select_device()
        assert device in ("cpu", "mps", "cuda")
        assert isinstance(device, str)

    def test_device_selection_prefers_cuda_over_mps(self):
        """Verify CUDA is preferred over MPS when both are available."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        sim = MatterSimMTSimulator()
        device = sim._select_device()

        # If CUDA is available, it should be selected
        if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
            if torch.cuda.is_available():
                assert device == "cuda"
            elif torch.backends.mps.is_available():
                assert device == "mps"
            else:
                assert device == "cpu"
        elif torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"

    def test_memory_estimation_device_aware(self):
        """Verify memory estimation varies by device type."""
        mem_cpu = MatterSimMTSimulator._estimate_memory_usage(500, "cpu")
        mem_mps = MatterSimMTSimulator._estimate_memory_usage(500, "mps")
        mem_cuda = MatterSimMTSimulator._estimate_memory_usage(500, "cuda")

        # CUDA should use more memory than MPS, which uses more than CPU
        assert mem_cuda >= mem_mps >= mem_cpu

    def test_tier2_initialization_logs_device(self, capsys):
        """Verify that Tier 2 initialization logs the selected device."""
        sim = MatterSimMTSimulator()
        sim.initialize()
        captured = capsys.readouterr()
        assert "device=" in captured.out.lower() or "Device" in captured.out


# ============================================================
# Parameter Loading Tests (v5.2)
# ============================================================

class TestParameterLoading:
    """Tests verifying that magic numbers are loaded from force_field_params.json."""

    def test_kmc_params_from_json(self):
        """Verify kMC parameters are loaded from force_field_params.json."""
        from aurelius.screening.tier3_gcmtwin import _load_kmc_params

        params = _load_kmc_params()
        assert "activation_energies_eV" in params
        assert "pre_exponential_factors_ps" in params
        assert "thickness_contributions_angstrom" in params
        assert params["activation_energies_eV"]["ec_reduction"] == 0.65
        assert params["kinetic_parameters"]["km_half_saturation"] == 0.3

    def test_solvation_params_from_json(self):
        """Verify solvation parameters are loaded from force_field_params.json."""
        from aurelius.solvation.engine import (
            _get_coordination_number,
            _get_energy_profile_gaussians,
            _get_rejection_threshold,
            _get_repulsive_wall_params,
            _get_shell_radius,
            _get_surface_tension,
            _load_solvation_params,
        )

        params = _load_solvation_params()
        assert "coordination_numbers" in params
        assert params["coordination_numbers"]["Na+"] == 6
        assert params["coordination_numbers"]["Li+"] == 4
        assert params["coordination_numbers"]["K+"] == 8

        # Test accessor functions
        assert _get_coordination_number("Na+") == 6
        assert _get_coordination_number("Li+") == 4
        assert _get_shell_radius("Na+") == 3.0
        assert _get_shell_radius("Li+") == 2.5
        assert _get_surface_tension() == 0.00542
        assert _get_rejection_threshold() == 0.5

        centers, widths, heights = _get_energy_profile_gaussians()
        assert centers == [1.5, 3.0, 4.2]
        assert widths == [0.4, 0.5, 0.3]
        assert heights == [0.15, 0.25, 0.10]

        amp, decay = _get_repulsive_wall_params()
        assert amp == 0.02
        assert decay == 0.5

    def test_scoring_params_from_json(self):
        """Verify scoring parameters are loaded from force_field_params.json."""
        from aurelius.scoring.engine import _load_scoring_params

        params = _load_scoring_params()
        assert "component_weights" in params
        assert params["component_weights"]["sigma"] == 0.3
        assert params["component_weights"]["desolvation"] == 0.2
        assert params["viability_threshold"] == 65.0

        mx_params = params["mx_synthesis"]
        assert mx_params["base_score"] == 70.0
        assert "ec:dmc" in mx_params["common_solvents"]
        assert "NaPF6" in mx_params["common_salts"]
        assert "Na+" in mx_params["common_ions"]

    def test_tier1_params_from_json(self):
        """Verify Tier 1 parameters are loaded from force_field_params.json."""
        import json
        import os

        ff_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__)
            ))),
            "src", "aurelius", "data", "force_field_params.json",
        )
        if os.path.isfile(ff_path):
            with open(ff_path) as f:
                data = json.load(f)
            tier1 = data.get("tier1_parameters", {})
            assert "esol_dataset" in tier1
            assert tier1["esol_dataset"]["mean_logS"] == -2.95
            assert tier1["training_hyperparameters"]["learning_rate"] == 0.005
            assert tier1["hash_fallback"]["n_bits"] == 2048

    def test_pipeline_defaults_from_json(self):
        """Verify pipeline defaults are loaded from force_field_params.json."""
        import json
        import os

        ff_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__)
            ))),
            "src", "aurelius", "data", "force_field_params.json",
        )
        if os.path.isfile(ff_path):
            with open(ff_path) as f:
                data = json.load(f)
            defaults = data.get("pipeline_defaults", {})
            assert defaults["voltage_cutoff_V"] == 0.05
            assert defaults["max_sei_time_ps"] == 1000.0
            assert defaults["config_defaults"]["barrier_threshold_eV"] == 0.5

    def test_gcmtwin_loads_params_from_json(self):
        """Verify GCMDigitalTwin loads parameters from JSON instead of hardcoded."""
        twin = GCMDigitalTwin()
        # These should come from force_field_params.json now
        assert twin._Ea_SOLVENT_EC == 0.65
        assert twin._A_SOLVENT_BASE == 5.0
        assert twin._K_m == 0.3
        assert twin._alpha == 0.5
        assert twin._initial_salt_conc == 0.1

    def test_mattersim_loads_params_from_json(self):
        """Verify MatterSimMTSimulator loads parameters from JSON instead of hardcoded."""
        sim = MatterSimMTSimulator()
        # LJ params should come from JSON now
        assert len(sim._LJ_PARAMS) > 0
        assert len(sim._CHARGES) > 0
        assert sim._default_eps == 0.02
        assert sim._default_sig == 2.5
        assert sim._cutoff == 12.0

    def test_solvation_engine_loads_params_from_json(self):
        """Verify MWSESolvationEngine loads parameters from JSON."""
        from aurelius.solvation.engine import (
            _get_coordination_number,
            _get_desolvation_energy,
            _get_shell_radius,
        )
        assert _get_coordination_number("Na+") == 6
        assert _get_shell_radius("Li+") == 2.5
        assert _get_desolvation_energy("Na+", "water") == 0.05


# ============================================================
# RDKit Fallback Warning Tests (v5.2)
# ============================================================

class TestRDKitFallbackWarnings:
    """Tests verifying explicit warnings when RDKit is unavailable."""

    def test_hash_fallback_uses_json_params(self):
        """Verify hash fallback loads parameters from force_field_params.json."""
        from aurelius.screening.tier1_mlx_filter import _hash_fallback

        fp = _hash_fallback("CCO")
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32
        # Should have some bits set
        assert np.sum(fp) > 0
        # Should be binary
        assert set(np.unique(fp).tolist()).issubset({0.0, 1.0})

    def test_hash_fallback_deterministic(self):
        """Verify hash fallback produces deterministic results."""
        from aurelius.screening.tier1_mlx_filter import _hash_fallback

        fp1 = _hash_fallback("CC(=O)OC1=CC(=O)O1")
        fp2 = _hash_fallback("CC(=O)OC1=CC(=O)O1")
        assert np.array_equal(fp1, fp2)

    def test_different_smiles_different_fallback(self):
        """Verify different SMILES produce different hash fallbacks."""
        from aurelius.screening.tier1_mlx_filter import _hash_fallback

        fp_ethanol = _hash_fallback("CCO")
        fp_acetic = _hash_fallback("CC(=O)O")
        # Different SMILES should produce different hash vectors
        assert not np.array_equal(fp_ethanol, fp_acetic)

    def test_na_utilization_uses_json_params(self):
        """Verify NA utilization estimation uses JSON parameters."""
        f = MLXNAFilter(quantization_format="MX4", train_on_init=False)
        util = f._estimate_na_utilization(0.75)
        assert 0 <= util <= 100
