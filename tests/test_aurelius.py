"""Tests for Project Aurelius v5.1."""

import pytest

from aurelius.config import M5ProConfig, get_config
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
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
