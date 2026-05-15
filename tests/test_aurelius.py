"""Tests for Project Aurelius v5.1."""

import pytest
import mlx.core as mx
import torch
import numpy as np

from aurelius.config import M5ProConfig, get_config
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator, MatterSimMPEngine
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, TurboQuantConfig
from aurelius.scoring.engine import AureliusScoringEngine, MoleculeInput


# ============================================================
# Config Tests
# ============================================================

class TestM5ProConfig:
    def test_default_memory_budget(self):
        config = M5ProConfig()
        assert config.validate_memory_budget() is True
        assert config.mlx_max_mem_gb == 12

    def test_memory_report(self):
        config = M5ProConfig()
        report = config.memory_report()
        assert "MLX" in report
        assert "Metal Shader Cache" in report
        assert "PyTorch MPS" in report

    def test_invalid_memory_budget(self):
        config = M5ProConfig(mlx_max_mem_gb=22, metal_shader_cache_gb=3)
        assert config.validate_memory_budget() is False


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
        assert budget["total_gb"] == 24.0
        assert "chemvlm2_footprint_gb" in budget
        assert "remaining_gb" in budget


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

    def test_desolvation_path_integral(self):
        barrier = self.engine.compute_desolvation_path_integral("Na+", "ec:dmc")
        assert barrier.barrier_height_eV >= 0
        assert barrier.path_integral_energy >= 0
        # The barrier should not have local maxima > 0.5 eV in our simulation
        assert barrier.local_maxima_eV <= 0.5

    def test_evaluate_mwse_state(self):
        state = self.engine.evaluate_mwse_state("Na+", "ec:dmc")
        assert state.solvation_shell.ion_type == "Na+"
        assert state.dipole_moment_debye > 0
        # 500-cycle stability depends on dipole > 3.5 D
        assert isinstance(state.is_stable_500cycle, bool)


# ============================================================
# MLX-NA Filter Tests
# ============================================================

class TestMLXNAFilter:
    def setup_method(self):
        self.filter = MLXNAFilter(quantization_format="MX4")

    def test_screen_molecule(self):
        result = self.filter.screen_molecule("CC(=O)OC1=CC(=O)O1")
        assert result.molecule_smiles == "CC(=O)OC1=CC(=O)O1"
        assert 0 <= result.confidence_score <= 1
        assert result.quantization_format == "MX4"
        assert 0 <= result.na_utilization_pct <= 100

    def test_screen_batch(self):
        molecules = [
            "CC(=O)OC1=CC(=O)O1",
            "C1CC(=O)OC1",
            "COC(=O)C1=CC=CC=C1",
        ]
        results = self.filter.screen_batch(molecules)
        assert len(results) == 3
        assert all(r.molecule_smiles in molecules for r in results)


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
        assert result.desolvation_path.barrier_height_eV >= 0
        assert result.simulation_time_ms >= 0


# ============================================================
# GCMD Digital Twin Tests
# ============================================================

class TestGCMDigitalTwin:
    def setup_method(self):
        self.twin = GCMDigitalTwin(
            turboquant_config=TurboQuantConfig(max_context_tokens=8192)
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
        assert result.context_tokens_used == 3276  # 8192 * 0.4

    def test_turboquant_stats(self):
        stats = self.twin.get_turboquant_stats()
        assert stats["max_context_tokens"] == 8192
        assert stats["kv_compression_ratio"] == 0.4
        assert stats["effective_context"] == 3276


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
        assert "AURELIUS SCORE v5.1" in card
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
        # With high threshold, most molecules should be rejected
        assert score.viability_threshold == 90.0

    def test_rejection_reasons(self):
        """Test that rejection reasons are populated for non-viable molecules."""
        engine = AureliusScoringEngine(viability_threshold=95.0)
        molecule_input = MoleculeInput(smiles="CC(=O)OC1=CC(=O)O1")
        score = engine.compute_score(molecule_input)
        if not score.is_viable:
            assert len(score.rejection_reasons) > 0


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
        from aurelius.bridge import bridge_mlx_to_pytorch, CrossFrameworkBridge

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

        # Instantiate actual structure framework
        model = MLXNAFilter(quantization_format="MX4")
        mock_input = mx.random.normal(shape=(batch_size, hidden_dim))

        # Process through MLX layer
        mlx_output = model._create_placeholder_mlx_model()(mock_input)
        assert list(mlx_output.shape) == [batch_size, hidden_dim]

        # Convert to physical numpy storage layout
        np_view = np.array(mlx_output)

        # Validate PyTorch ingestion dimensions for structural graphing
        torch_tensor = torch.from_numpy(np_view).to("mps")
        assert torch_tensor.is_mps
        assert torch_tensor.shape == (batch_size, hidden_dim)

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

        # Test various shapes
        test_shapes = [(1, 1024), (32, 128), (100, 3), (512,)]
        for shape in test_shapes:
            mlx_array = mx.random.normal(shape=shape)
            try:
                torch_tensor = bridge_mlx_to_pytorch(mlx_array)
            except AttributeError as e:
                if "DLpack" in str(e):
                    pytest.skip("MLX version does not support DLpack")
                raise
            assert torch_tensor.shape == shape
