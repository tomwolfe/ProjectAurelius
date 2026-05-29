"""Tests for physics validation (Arrhenius behavior and energy conservation).

REMOVED in v8.0: Fake physics engines were replaced by real ML oracles.
These tests are now skipped.
"""

from __future__ import annotations

import pytest

pytest.skip("Fake physics removed in v8.0; use PretrainedGNNOracle instead", allow_module_level=True)
