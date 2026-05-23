"""Tests for Project Aurelius v5.2."""

from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    HAS_MLX = False

try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None  # type: ignore[assignment, unused-ignore]

from aurelius.config import AureliusConfig
from aurelius.memory.manager import (
    QuantizationConfig,
    ZeroCopyMemoryManager,
)

# ============================================================
# Config Tests
# ============================================================


class TestAureliusConfig:
    def test_default_memory_budget(self):
        config = AureliusConfig()
        assert config.validate_memory_budget() is True
        # MLX gets 50% of RAM, capped at 12GB
        assert config.mlx_max_mem_gb <= 12.0
        assert config.mlx_max_mem_gb > 0

    def test_memory_report(self):
        config = AureliusConfig()
        report = config.memory_report()
        assert "MLX" in report
        assert "Metal Shader Cache" in report
        assert "PyTorch MPS" in report

    def test_invalid_memory_budget(self):
        config = AureliusConfig(mlx_max_mem_gb=22, metal_shader_cache_gb=3)
        assert config.validate_memory_budget() is False

    def test_dynamic_ram_detection(self):
        """Verify that config detects system RAM dynamically."""
        import psutil

        config = AureliusConfig()
        detected_gb = psutil.virtual_memory().total / (1024**3)
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
        detected = psutil.virtual_memory().total / (1024**3)
        assert mgr._total_ram_gb <= detected + 1
