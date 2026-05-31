"""Tests for Project Aurelius v9.0 configuration."""
from __future__ import annotations

from aurelius.config import AureliusConfig


class TestAureliusConfig:
    def test_default_instantiation(self):
        config = AureliusConfig()
        assert isinstance(config, AureliusConfig)
