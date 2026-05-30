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
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None  # type: ignore[assignment, unused-ignore]

from aurelius.config import AureliusConfig

# ============================================================
# Config Tests
# ============================================================


class TestAureliusConfig:
    def test_default_values(self):
        config = AureliusConfig()
        assert config.weight_sigma == 0.4
        assert config.weight_desolvation_barrier == 0.2
        assert config.chemvlm_quantization == "MX4"
        assert config.tier1_mlxfilter_enabled is True

    def test_custom_values(self):
        config = AureliusConfig(weight_sigma=0.5, weight_desolvation_barrier=0.3)
        assert config.weight_sigma == 0.5
        assert config.weight_desolvation_barrier == 0.3
