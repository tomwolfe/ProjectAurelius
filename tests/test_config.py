"""Tests for Project Aurelius v9.0 configuration."""

from __future__ import annotations

from aurelius.config import AureliusConfig


class TestAureliusConfig:
    def test_default_values(self):
        config = AureliusConfig()
        assert config.weight_sigma == 0.4
        assert config.weight_desolvation_barrier == 0.2

    def test_custom_values(self):
        config = AureliusConfig(weight_sigma=0.5, weight_desolvation_barrier=0.3)
        assert config.weight_sigma == 0.5
        assert config.weight_desolvation_barrier == 0.3
