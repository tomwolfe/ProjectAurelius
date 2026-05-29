"""Tests for Aurelius Scoring Engine.

REMOVED in v8.0: The scoring engine was replaced by the real ML-based Oracle.
These tests are now skipped.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "AureliusScoringEngine removed in v8.0; use PretrainedGNNOracle instead",
    allow_module_level=True,
)
