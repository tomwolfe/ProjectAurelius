"""Tests for MatterSim (Tier 2).

REMOVED in v8.0: MatterSim was replaced by the real ML-based Oracle.
These tests are now skipped.
"""

from __future__ import annotations

import pytest

pytest.skip("MatterSim removed in v8.0; use PretrainedGNNOracle instead", allow_module_level=True)
