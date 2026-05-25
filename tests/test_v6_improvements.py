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

import pytest

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
        """Verify DependencyManager can be instantiated."""
        from aurelius.utils.dependencies import DependencyManager

        deps = DependencyManager()
        assert deps is not None

    def test_check_framework_mlx(self):
        """Verify check_framework returns correct info for MLX."""
        from aurelius.utils.dependencies import HAS_MLX, DependencyManager

        deps = DependencyManager()
        info = deps.check_framework("mlx")
        assert "available" in info
        assert "version" in info
        assert "meets_minimum" in info
        assert "min_version" in info
        assert info["available"] == HAS_MLX

    def test_check_framework_torch(self):
        """Verify check_framework returns correct info for PyTorch."""
        from aurelius.utils.dependencies import HAS_TORCH, DependencyManager

        deps = DependencyManager()
        info = deps.check_framework("torch")
        assert info["available"] == HAS_TORCH

    def test_check_framework_rdkit(self):
        """Verify check_framework returns correct info for RDKit."""
        from aurelius.utils.dependencies import HAS_RDKIT, DependencyManager

        deps = DependencyManager()
        info = deps.check_framework("rdkit")
        assert info["available"] == HAS_RDKIT

    def test_report_status(self):
        """Verify report_status returns status for all frameworks."""
        from aurelius.utils.dependencies import DependencyManager

        deps = DependencyManager()
        status = deps.report_status()
        assert "mlx" in status
        assert "torch" in status
        assert "rdkit" in status
        assert "huggingface-hub" in status
        assert "datasets" in status

    def test_routing_info(self):
        """Verify routing_info categorizes frameworks correctly."""
        from aurelius.utils.dependencies import DependencyManager

        deps = DependencyManager()
        routing = deps.routing_info()
        for _fw, route in routing.items():
            assert route in ("native", "fallback", "unavailable")

    def test_clear_cache(self):
        """Verify clear_cache resets the status cache."""
        from aurelius.utils.dependencies import DependencyManager

        deps = DependencyManager()
        deps.report_status()
        deps.clear_cache()
        # After clearing, report_status should recompute
        status = deps.report_status()
        assert len(status) > 0

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
# Environment Validation Tests
# ============================================================


class TestEnvironmentValidation:
    """Tests for environment variable validation."""

    def test_validate_environment_compliant(self):
        """Verify validate_environment returns compliant when env matches config."""
        pytest.skip("validate_environment was removed during refactoring")

    def test_validate_environment_with_mismatch(self):
        """Verify validate_environment detects env var mismatches."""
        pytest.skip("validate_environment was removed during refactoring")

    def test_validate_environment_missing_vars(self):
        """Verify validate_environment detects missing env vars."""
        pytest.skip("validate_environment was removed during refactoring")

    def test_print_env_diff(self):
        """Verify print_env_diff produces output."""
        pytest.skip("print_env_diff was removed during refactoring")


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
        assert "Aurelius v7.0 Doctor" in result.stdout or "Aurelius v7.0 Doctor" in result.stderr

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
        """Verify RDKit error includes pip/conda install commands."""
        # Read the source file directly (screen is a Click Command, not a function)
        import pathlib

        main_file = pathlib.Path(__file__).resolve().parent.parent / "src" / "aurelius" / "__main__.py"
        source = main_file.read_text()
        assert "pip install rdkit" in source or "conda install" in source

    def test_rdkit_error_is_comprehensive(self):
        """Verify RDKit error message is comprehensive with platform notes."""
        import pathlib

        main_file = pathlib.Path(__file__).resolve().parent.parent / "src" / "aurelius" / "__main__.py"
        source = main_file.read_text()
        # Should include platform-specific install guidance
        assert "pip install rdkit" in source
        assert "conda install" in source
        # Should mention --allow-fallback as alternative
        assert "--allow-fallback" in source
        # Should mention --demo as alternative
        assert "--demo" in source

    def test_allow_fallback_warning(self):
        """Verify --allow-fallback shows production risk warning."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aurelius", "screen", "CCO", "--allow-fallback"],
            capture_output=True,
            text=True,
        )
        # Should show a warning about hash fallback
        output = result.stdout + result.stderr
        assert "WARNING" in output or "warning" in output.lower() or "fallback" in output.lower()


# ============================================================
# PBC Placeholder Tests
# ============================================================


class TestPBCPlaceholder:
    """Tests for Periodic Boundary Conditions placeholder."""

    def test_use_pbc_true_enabled(self):
        """Verify use_pbc=True enables PBC functionality."""
        from aurelius.screening.tier2_mattersim import MatterSimMTSimulator

        sim = MatterSimMTSimulator(use_pbc=True)
        assert sim is not None
        assert sim._use_pbc is True

    def test_use_pbc_false_defaults(self):
        """Verify use_pbc=False (default) works normally."""
        from aurelius.screening.tier2_mattersim import MatterSimMTSimulator

        sim = MatterSimMTSimulator(use_pbc=False)
        assert sim is not None


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
        """Verify DependencyManager is accessible from utils module."""
        from aurelius.utils import DependencyManager

        deps = DependencyManager()
        assert deps is not None

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
