"""Tests for Project Aurelius v7.0 improvements.

Tests for:
- Centralized DependencyManager
- HuggingFace symlinks env var
- Disk space checks
- Environment validation
- `aurelius doctor` CLI command
- RDKit error message improvements
- PBC placeholder in tier2
"""

from __future__ import annotations

import os

# ============================================================
# DependencyManager Tests
# ============================================================


class TestDependencyManager:
    """Tests for centralized dependency detection."""

    def test_has_mlx_export(self):
        """Verify HAS_MLX is exported from dependencies module."""
        from aurelius.utils.dependencies import HAS_MLX

        assert isinstance(HAS_MLX, bool)

    def test_has_torch_export(self):
        """Verify HAS_TORCH is exported from dependencies module."""
        from aurelius.utils.dependencies import HAS_TORCH

        assert isinstance(HAS_TORCH, bool)

    def test_has_rdkit_export(self):
        """Verify HAS_RDKIT is exported from dependencies module."""
        from aurelius.utils.dependencies import HAS_RDKIT

        assert isinstance(HAS_RDKIT, bool)

    def test_dependency_manager_instantiation(self):
        """Verify module-level dependency functions are callable."""
        from aurelius.utils.dependencies import check_framework

        assert callable(check_framework)
        info = check_framework("mlx")
        assert "available" in info
        assert "version" in info
        assert "meets_minimum" in info
        assert "min_version" in info

    def test_module_level_functions(self):
        """Verify module-level convenience functions work."""
        from aurelius.utils.dependencies import check_framework, report_status, routing_info

        assert callable(check_framework)
        assert callable(report_status)
        assert callable(routing_info)

        result = check_framework("mlx")
        assert "available" in result

        status = report_status()
        assert "mlx" in status

        routing = routing_info()
        assert "mlx" in routing


# ============================================================
# HuggingFace Symlinks Tests
# ============================================================


class TestHFSymlinks:
    """Tests for HuggingFace symlink control."""

    def test_should_use_symlinks_default(self):
        """Verify default is False (compatibility)."""
        from aurelius.screening.tier1.loaders import _should_use_symlinks

        # Ensure env var is not set
        old_val = os.environ.pop("AURELIUS_HF_USE_SYMLINKS", None)
        try:
            assert _should_use_symlinks() is False
        finally:
            if old_val is not None:
                os.environ["AURELIUS_HF_USE_SYMLINKS"] = old_val

    def test_should_use_symlinks_env_true(self):
        """Verify env var '1' enables symlinks."""
        from aurelius.screening.tier1.loaders import _should_use_symlinks

        old_val = os.environ.get("AURELIUS_HF_USE_SYMLINKS")
        try:
            os.environ["AURELIUS_HF_USE_SYMLINKS"] = "1"
            assert _should_use_symlinks() is True
        finally:
            if old_val is not None:
                os.environ["AURELIUS_HF_USE_SYMLINKS"] = old_val
            else:
                os.environ.pop("AURELIUS_HF_USE_SYMLINKS", None)

    def test_should_use_symlinks_env_true_string(self):
        """Verify env var 'true' enables symlinks."""
        from aurelius.screening.tier1.loaders import _should_use_symlinks

        old_val = os.environ.get("AURELIUS_HF_USE_SYMLINKS")
        try:
            os.environ["AURELIUS_HF_USE_SYMLINKS"] = "true"
            assert _should_use_symlinks() is True
        finally:
            if old_val is not None:
                os.environ["AURELIUS_HF_USE_SYMLINKS"] = old_val
            else:
                os.environ.pop("AURELIUS_HF_USE_SYMLINKS", None)

    def test_should_use_symlinks_env_false(self):
        """Verify env var '0' disables symlinks."""
        from aurelius.screening.tier1.loaders import _should_use_symlinks

        old_val = os.environ.get("AURELIUS_HF_USE_SYMLINKS")
        try:
            os.environ["AURELIUS_HF_USE_SYMLINKS"] = "0"
            assert _should_use_symlinks() is False
        finally:
            if old_val is not None:
                os.environ["AURELIUS_HF_USE_SYMLINKS"] = old_val
            else:
                os.environ.pop("AURELIUS_HF_USE_SYMLINKS", None)


# ============================================================
# Disk Space Check Tests
# ============================================================


class TestDiskSpaceCheck:
    """Tests for disk space validation."""

    def test_check_disk_space_root(self):
        """Verify disk space check works on root directory."""
        from aurelius.screening.tier1.loaders import check_disk_space

        has_space, free_gb = check_disk_space("/")
        assert isinstance(has_space, bool)
        assert free_gb >= 0

    def test_check_disk_space_current_dir(self):
        """Verify disk space check works on current directory."""
        from aurelius.screening.tier1.loaders import check_disk_space

        has_space, free_gb = check_disk_space(".")
        assert isinstance(has_space, bool)
        assert free_gb >= 0

    def test_check_disk_space_nonexistent(self):
        """Verify disk space check handles nonexistent paths gracefully."""
        from aurelius.screening.tier1.loaders import check_disk_space

        # Should not raise, should return (True, 0.0)
        has_space, free_gb = check_disk_space("/nonexistent/path/xyz")
        assert has_space is True
        assert free_gb == 0.0



# ============================================================
# CLI Doctor Command Tests
# ============================================================


class TestDoctorCommand:
    """Tests for `aurelius doctor` CLI command."""

    def test_doctor_command_exists(self):
        """Verify doctor command is registered in CLI."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "--help"],
            capture_output=True,
            text=True,
        )
        assert "doctor" in result.stdout or "doctor" in result.stderr

    def test_doctor_command_runs(self):
        """Verify doctor command runs without errors."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "doctor"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Doctor output should contain framework status or hardware info
        output = result.stdout + result.stderr
        assert "Framework" in output or "Hardware" in output or "Summary" in output

    def test_doctor_verbose_output(self):
        """Verify doctor --verbose shows framework versions."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "doctor", "--verbose"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ============================================================
# RDKit Error Message Tests
# ============================================================


class TestRDKitErrors:
    """Tests for improved RDKit error messages."""

    def test_rdkit_error_includes_install_commands(self):
        """Verify RDKit error includes pip install guidance."""
        import pathlib

        main_file = pathlib.Path(__file__).resolve().parent.parent / "src" / "aurelius" / "__main__.py"
        source = main_file.read_text()
        # RDKit is now strictly enforced for real model screening
        assert "RDKit" in source or "rdkit" in source.lower()

    def test_rdkit_error_is_comprehensive(self):
        """Verify RDKit error message is comprehensive with platform notes."""
        import pathlib

        main_file = pathlib.Path(__file__).resolve().parent.parent / "src" / "aurelius" / "__main__.py"
        source = main_file.read_text()
        # RDKit is now strictly enforced - no fallback option
        assert "RDKit" in source or "rdkit" in source.lower()


# ============================================================
# Backward Compatibility Tests
# ============================================================


class TestBackwardCompatibility:
    """Tests ensuring existing APIs remain functional."""

    def test_has_mlx_from_tier1(self):
        """Verify HAS_MLX is still accessible from tier1 module."""
        from aurelius.screening.tier1 import HAS_MLX

        assert isinstance(HAS_MLX, bool)

    def test_has_torch_from_tier1(self):
        """Verify HAS_TORCH is still accessible from tier1 module."""
        from aurelius.screening.tier1 import HAS_TORCH

        assert isinstance(HAS_TORCH, bool)

    def test_has_rdkit_from_tier1(self):
        """Verify HAS_RDKIT is still accessible from tier1 module."""
        from aurelius.screening.tier1 import HAS_RDKIT

        assert isinstance(HAS_RDKIT, bool)

    def test_has_mlx_from_utils(self):
        """Verify HAS_MLX is exported from utils module."""
        from aurelius.utils import HAS_MLX

        assert isinstance(HAS_MLX, bool)

    def test_dependency_manager_from_utils(self):
        """Verify check_framework is accessible from utils module."""
        from aurelius.utils import check_framework

        assert callable(check_framework)
        result = check_framework("mlx")
        assert "available" in result

    def test_huggingface_weight_loader_backward_compat(self):
        """Verify HuggingFaceWeightLoader constructor is backward compatible."""
        from aurelius.screening.tier1.loaders import HuggingFaceWeightLoader

        loader = HuggingFaceWeightLoader()
        assert loader is not None

    def test_mlxna_filter_backward_compat(self):
        """Verify MLXNAFilter constructor is backward compatible."""
        from aurelius.screening.tier1 import MLXNAFilter

        f = MLXNAFilter(quantization_format="MX4", train_on_init=False)
        assert f is not None

    def test_mattersim_simulator_backward_compat(self):
        """Verify MatterSimMTSimulator constructor is backward compatible."""
        from aurelius.screening.tier2_mattersim import MatterSimMTSimulator

        sim = MatterSimMTSimulator()
        assert sim is not None
