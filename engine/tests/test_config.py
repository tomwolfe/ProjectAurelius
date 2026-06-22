"""Tests for Project Aurelius constants."""
from __future__ import annotations

from aurelius.constants import (
    FINGERPRINT_SIZE,
    HOMO_THRESHOLD,
    LUMO_TARGET,
    VIABILITY_THRESHOLD,
)


class TestConstants:
    def test_fingerprint_size(self):
        assert FINGERPRINT_SIZE == 2048

    def test_homo_threshold_is_negative(self):
        assert HOMO_THRESHOLD < 0

    def test_lumo_target(self):
        assert LUMO_TARGET == -1.0

    def test_viability_threshold(self):
        assert VIABILITY_THRESHOLD == 50.0
