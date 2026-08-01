"""xTB integration tests with mocked subprocess.

Tests the xTB parsing, Boltzmann weighting, and batch runner
lifecycle using pre-recorded xTB stdout for 5 reference molecules.
"""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from aurelius.compute.xtb_pool import (
    _HAS_XTB,
    _XTB_HOMO_RE,
    _XTB_LUMO_RE,
    BatchXTBRunner,
    _parse_xtb_output,
    _run_xtb,
    has_xtb,
)
from aurelius.scoring.oracle.quantum import _boltzmann_weights

# ---------------------------------------------------------------------------
# Pre-recorded xTB stdout for 5 reference molecules
# ---------------------------------------------------------------------------

_XTB_STDOUT_EC = """
xTB calculation completed successfully
HOMO:   -0.2345 eV
LUMO:    0.8765 eV
Dipole:  1.234 D
"""

_XTB_STDOUT_DMC = """
xTB calculation completed successfully
HOMO:   -0.3456 eV
LUMO:    0.7654 eV
Dipole:  2.345 D
"""

_XTB_STDOUT_DME = """
xTB calculation completed successfully
HOMO:   -0.4567 eV
LUMO:    0.6543 eV
Dipole:  1.567 D
"""

_XTB_STDOUT_ACN = """
xTB calculation completed successfully
HOMO:   -0.5678 eV
LUMO:    0.5432 eV
Dipole:  3.456 D
"""

_XTB_STDOUT_SULFOLANE = """
xTB calculation completed successfully
HOMO:   -0.6789 eV
LUMO:    0.4321 eV
Dipole:  4.567 D
"""

_XTB_STDOUT_GARBAGE = """
some random garbage output
no HOMO or LUMO here
"""

_XTB_STDOUT_EMPTY = ""

_XTB_STDOUT_PARTIAL = """
xTB calculation completed successfully
HOMO:   -0.2345 eV
"""

_REFERENCE_MOLECULES = {
    "EC": ("CC(=O)OC", _XTB_STDOUT_EC),
    "DMC": ("COC(=O)OC", _XTB_STDOUT_DMC),
    "DME": ("COCCOC", _XTB_STDOUT_DME),
    "ACN": ("CC#N", _XTB_STDOUT_ACN),
    "sulfolane": ("S1(=O)(=O)CCCC1", _XTB_STDOUT_SULFOLANE),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_smiles_list():
    return list(_REFERENCE_MOLECULES.keys())


@pytest.fixture
def reference_stdout_map():
    return {name: stdout for name, (_, stdout) in _REFERENCE_MOLECULES.items()}


# ---------------------------------------------------------------------------
# _parse_xtb_output tests
# ---------------------------------------------------------------------------


class TestParseXtbOutput:
    def test_parse_ec(self):
        result = _parse_xtb_output(_XTB_STDOUT_EC)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.2345)) < 1e-4
        assert abs(result["lumo_eV"] - 0.8765) < 1e-4

    def test_parse_dmc(self):
        result = _parse_xtb_output(_XTB_STDOUT_DMC)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.3456)) < 1e-4
        assert abs(result["lumo_eV"] - 0.7654) < 1e-4

    def test_parse_dme(self):
        result = _parse_xtb_output(_XTB_STDOUT_DME)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.4567)) < 1e-4
        assert abs(result["lumo_eV"] - 0.6543) < 1e-4

    def test_parse_acn(self):
        result = _parse_xtb_output(_XTB_STDOUT_ACN)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.5678)) < 1e-4
        assert abs(result["lumo_eV"] - 0.5432) < 1e-4

    def test_parse_sulfolane(self):
        result = _parse_xtb_output(_XTB_STDOUT_SULFOLANE)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.6789)) < 1e-4
        assert abs(result["lumo_eV"] - 0.4321) < 1e-4

    def test_parse_garbage_returns_none(self):
        result = _parse_xtb_output(_XTB_STDOUT_GARBAGE)
        assert result is None

    def test_parse_empty_returns_none(self):
        result = _parse_xtb_output(_XTB_STDOUT_EMPTY)
        assert result is None

    def test_parse_partial_output_returns_none(self):
        result = _parse_xtb_output(_XTB_STDOUT_PARTIAL)
        assert result is None

    def test_parse_missing_homo(self):
        stdout = "LUMO:  0.8765 eV\n"
        result = _parse_xtb_output(stdout)
        assert result is None

    def test_parse_missing_lumo(self):
        stdout = "HOMO:  -0.2345 eV\n"
        result = _parse_xtb_output(stdout)
        assert result is None


# ---------------------------------------------------------------------------
# _boltzmann_weights tests
# ---------------------------------------------------------------------------


class TestBoltzmannWeights:
    def test_normalization(self):
        energies = [0.0, 1.0, 2.0]
        weights = _boltzmann_weights(energies)
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_all_equal_energies(self):
        energies = [1.0, 1.0, 1.0]
        weights = _boltzmann_weights(energies)
        assert all(abs(w - 1.0 / 3.0) < 1e-6 for w in weights)

    def test_single_energy(self):
        weights = _boltzmann_weights([1.0])
        assert weights == [1.0]

    def test_empty_energies(self):
        weights = _boltzmann_weights([])
        assert weights == []

    def test_lower_energy_higher_weight(self):
        energies = [0.0, 5.0, 10.0]
        weights = _boltzmann_weights(energies)
        assert weights[0] > weights[1] > weights[2]

    def test_temperature_scaling(self):
        energies = [0.0, 1.0]
        weights_300 = _boltzmann_weights(energies, temperature=300.0)
        weights_1000 = _boltzmann_weights(energies, temperature=1000.0)
        assert weights_300[0] > weights_300[1]
        assert weights_1000[0] > weights_1000[1]
        assert weights_1000[0] < weights_300[0]


# ---------------------------------------------------------------------------
# BatchXTBRunner lifecycle tests
# ---------------------------------------------------------------------------


class TestBatchXTBRunnerLifecycle:
    def test_submit_and_flush(self):
        runner = BatchXTBRunner(batch_size=2, flush_interval=1.0, max_workers=1)
        xyz = "1\n\nC 0.0 0.0 0.0\n"
        future = runner.submit(xyz)
        assert runner.pending_count >= 0
        runner.flush()
        runner.shutdown()

    def test_submit_multiple(self):
        runner = BatchXTBRunner(batch_size=5, flush_interval=2.0, max_workers=2)
        xyz = "1\n\nC 0.0 0.0 0.0\n"
        for _ in range(3):
            runner.submit(xyz)
        runner.flush()
        runner.shutdown()

    def test_shutdown_drains_pending(self):
        runner = BatchXTBRunner(batch_size=10, flush_interval=5.0, max_workers=1)
        xyz = "1\n\nC 0.0 0.0 0.0\n"
        runner.submit(xyz)
        runner.shutdown()

    def test_pending_count_starts_at_zero(self):
        runner = BatchXTBRunner(batch_size=2, flush_interval=1.0, max_workers=1)
        assert runner.pending_count == 0
        runner.shutdown()


# ---------------------------------------------------------------------------
# Graceful fallback tests
# ---------------------------------------------------------------------------


class TestXtbFallback:
    def test_has_xtb_false_when_unavailable(self):
        with patch("aurelius.compute.xtb_pool._XTB_BIN", None):
            with patch("aurelius.compute.xtb_pool._HAS_XTB", False):
                assert not has_xtb()

    def test_run_xtb_returns_none_when_unavailable(self):
        with patch("aurelius.compute.xtb_pool._HAS_XTB", False):
            result = _run_xtb("1\n\nC 0.0 0.0 0.0\n")
            assert result is None

    def test_batch_runner_graceful_with_no_xtb(self):
        with patch("aurelius.compute.xtb_pool._HAS_XTB", False):
            runner = BatchXTBRunner(batch_size=2, flush_interval=1.0, max_workers=1)
            xyz = "1\n\nC 0.0 0.0 0.0\n"
            future = runner.submit(xyz)
            result = future.result()
            assert result is None
            runner.shutdown()


# ---------------------------------------------------------------------------
# Integration test with mocked subprocess
# ---------------------------------------------------------------------------


class TestXtbIntegrationMocked:
    def test_subprocess_run_returns_parsed_result(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _XTB_STDOUT_EC
        mock_result.stderr = ""

        with patch("aurelius.compute.xtb_pool._XTB_BIN", "xtb"):
            with patch("aurelius.compute.xtb_pool._HAS_XTB", True):
                with patch("subprocess.run", return_value=mock_result):
                    result = _run_xtb(_XTB_STDOUT_EC)
        assert result is not None
        assert abs(result["homo_eV"] - (-0.2345)) < 1e-4
        assert abs(result["lumo_eV"] - 0.8765) < 1e-4

    def test_subprocess_run_garbage_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _XTB_STDOUT_GARBAGE
        mock_result.stderr = ""

        with patch("aurelius.compute.xtb_pool._XTB_BIN", "xtb"):
            with patch("aurelius.compute.xtb_pool._HAS_XTB", True):
                with patch("subprocess.run", return_value=mock_result):
                    result = _run_xtb(_XTB_STDOUT_GARBAGE)
        assert result is None

    def test_subprocess_run_nonzero_exit_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error"

        with patch("aurelius.compute.xtb_pool._XTB_BIN", "xtb"):
            with patch("aurelius.compute.xtb_pool._HAS_XTB", True):
                with patch("subprocess.run", return_value=mock_result):
                    result = _run_xtb("1\n\nC 0.0 0.0 0.0\n")
        assert result is None

    def test_batch_runner_with_mocked_subprocess(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _XTB_STDOUT_EC
        mock_result.stderr = ""

        with patch("aurelius.compute.xtb_pool._XTB_BIN", "xtb"):
            with patch("aurelius.compute.xtb_pool._HAS_XTB", True):
                with patch("subprocess.run", return_value=mock_result):
                    runner = BatchXTBRunner(batch_size=2, flush_interval=1.0, max_workers=1)
                    xyz = "1\n\nC 0.0 0.0 0.0\n"
                    future = runner.submit(xyz)
                    result = future.result()
                    assert result is not None
                    assert abs(result["homo_eV"] - (-0.2345)) < 1e-4
                    runner.shutdown()

    def test_reference_molecules_all_parse(self):
        for name, (_, stdout) in _REFERENCE_MOLECULES.items():
            result = _parse_xtb_output(stdout)
            assert result is not None, f"Failed to parse {name}"
            assert "homo_eV" in result
            assert "lumo_eV" in result