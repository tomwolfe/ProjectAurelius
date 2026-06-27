"""Tests for dependency detection and CLI doctor command."""

from __future__ import annotations


class TestDependencyManager:
    """Tests for centralized dependency detection."""

    def test_has_rdkit_export(self):
        from aurelius.utils.dependencies import HAS_RDKIT

        assert isinstance(HAS_RDKIT, bool)

    def test_check_xtb_benchmark_returns_string_or_none(self):
        from aurelius.utils.dependencies import check_xtb_with_benchmark

        result = check_xtb_with_benchmark()
        assert result is None or isinstance(result, str)
        if result is not None:
            assert "xTB" in result


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

    def test_doctor_includes_xtb_benchmark_output(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "doctor"],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert "Framework" in output
        if "xtb" in output:
            # If xTB was detected, benchmark output should be present
            has_benchmark = "xTB Active" in output
            has_missing = "MISSING" in output and "xtb" in output
            # Either xTB was found (with benchmark) or not found
            assert has_benchmark or has_missing
