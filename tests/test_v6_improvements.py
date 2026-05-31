"""Tests for Project Aurelius v9.0 improvements.

Tests for:
- Centralized dependency detection (PyTorch / RDKit)
- CLI doctor command
"""

from __future__ import annotations


class TestDependencyManager:
    """Tests for centralized dependency detection."""

    def test_has_torch_export(self):
        from aurelius.utils.dependencies import HAS_TORCH

        assert isinstance(HAS_TORCH, bool)

    def test_has_rdkit_export(self):
        from aurelius.utils.dependencies import HAS_RDKIT

        assert isinstance(HAS_RDKIT, bool)


class TestDoctorCommand:
    """Tests for `aurelius doctor` CLI command."""

    def test_doctor_command_exists(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "--help"],
            capture_output=True,
            text=True,
        )
        assert "doctor" in result.stdout or "doctor" in result.stderr

    def test_doctor_command_runs(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "doctor"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Framework" in output or "Hardware" in output or "Summary" in output
