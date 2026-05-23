"""Dynamic environment configuration for Apple M-series memory management.

Detects system RAM dynamically using psutil and allocates memory
across MLX, Metal Shader Cache, and PyTorch MPS without hardcoding
for a specific chip generation.

Memory allocation strategy:
    - MLX:          50% of available RAM (capped at 12GB)
    - Metal Shader: 10% of available RAM (capped at 2GB)
    - PyTorch MPS:  Remaining RAM

Thread-safety: Environment variable setup is delegated to the CLI
entry point (__main__.py) to avoid race conditions during
concurrent pipeline initialization.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class AureliusConfig:
    """Dynamic memory configuration for Apple M-series chips.

    Detects total system RAM at instantiation and computes
    tiered allocations accordingly. User-provided non-zero values
    override computed defaults.

    Uses a standard mutable @dataclass with __post_init__ validation.
    Dynamic RAM detection via psutil runs in __post_init__ after all
    fields are initialized by the dataclass __init__.
    """

    # Dynamically set from psutil at construction time
    total_memory_gb: float = 0.0

    # MLX memory allocation (50% of RAM, capped at 12GB)
    mlx_max_mem_gb: float = 0.0

    # PyTorch MPS async compilation
    pytorch_mps_async: bool = True

    # Metal shader pre-compilation buffer (10% of RAM, capped at 2GB)
    metal_shader_cache_gb: float = 0.0

    # GCMD kMC simulation parameters
    turquant_max_context: int = 8192

    # Desolvation energy barrier rejection threshold (eV)
    desolvation_barrier_threshold_eV: float = 0.5

    # MWSE solvent exchange rate screening window (ps)
    kex_screening_window_ps: float = 10.0

    # Aurelius Score weights (v5.2 formula)
    weight_sigma: float = 0.3
    weight_desolvation_barrier: float = 0.2
    weight_sei_homogeneity: float = 0.2
    weight_mx_synthesis_score: float = 0.2
    weight_gwp: float = 0.1

    # Quantization presets
    chemvlm_quantization: str = "MX4"
    mattersim_quantization: str = "MX4"
    gcmd_quantization: str = "standard"

    # Screening pipeline tiers
    tier1_mlxfilter_enabled: bool = True
    tier2_mattersim_enabled: bool = True
    tier3_gcmtwin_enabled: bool = True

    # Tier 2 neighbor list settings
    use_neighbor_list: bool = False
    neighbor_list_cutoff: float = 12.0

    def __post_init__(self) -> None:
        """Compute dynamic memory allocations after dataclass initialization.

        Detects system RAM via psutil and computes MLX / shader cache
        allocations. User-provided non-zero values override computed
        defaults.
        """
        # Detect total system RAM
        total_bytes = psutil.virtual_memory().total
        total_gb = total_bytes / (1024**3)

        # Use user override if provided
        if self.total_memory_gb > 0:
            total_gb = self.total_memory_gb

        # MLX: 50% of RAM, capped at 12GB
        mlx_alloc = min(total_gb * 0.5, 12.0)

        # Metal Shader Cache: 10% of RAM, capped at 2GB
        shader_alloc = min(total_gb * 0.1, 2.0)

        # Only apply computed defaults when user hasn't provided non-zero values
        if self.total_memory_gb == 0:
            self.total_memory_gb = round(total_gb, 1)
        if self.mlx_max_mem_gb == 0.0:
            self.mlx_max_mem_gb = round(mlx_alloc, 1)
        if self.metal_shader_cache_gb == 0.0:
            self.metal_shader_cache_gb = round(shader_alloc, 1)

    def apply_environment(self) -> dict[str, str]:
        """Return environment variables to set (thread-safe).

        Does NOT mutate os.environ directly. Callers (e.g., CLI
        entry point) should apply these in a thread-safe manner.

        Returns:
            Dictionary of environment variable name -> value.
        """
        return {
            "PYTORCH_MPS_ENABLE_ASYNC_COMPILATION": "1" if self.pytorch_mps_async else "0",
            "MLX_MAX_MEM_CACHE": f"{int(self.mlx_max_mem_gb)}G",
            "AURELIUS_VERSION": "6.0.0",
            "AURELIUS_QUANT_PRESET": self.chemvlm_quantization,
        }

    def validate_memory_budget(self) -> bool:
        """Validate that memory budget fits within physical RAM."""
        if self.mlx_max_mem_gb > 12.0:
            return False
        if self.metal_shader_cache_gb > 2.0:
            return False
        used = self.mlx_max_mem_gb + self.metal_shader_cache_gb
        remaining = self.total_memory_gb - used
        # Require at least 2GB for PyTorch MPS (scales down for smaller machines)
        min_required = min(8.0, self.total_memory_gb * 0.25)
        return remaining >= min_required

    def memory_report(self) -> str:
        """Generate a human-readable memory partition report."""
        mlx_reserved = self.mlx_max_mem_gb
        shader_reserved = self.metal_shader_cache_gb
        pytorch_available = self.total_memory_gb - mlx_reserved - shader_reserved
        return (
            f"=== Aurelius v5.2 Memory Partition ({self.total_memory_gb:.0f}GB system RAM) ===\n"
            f"  MLX (ChemVLM-2 MX4):         {mlx_reserved:>5.1f}GB reserved\n"
            f"  Metal Shader Cache:          {shader_reserved:>5.1f}GB reserved\n"
            f"  PyTorch MPS (MatterSim+GCMD): {pytorch_available:>5.1f}GB available\n"
            f"  GCMD kMC Steps:              {self.turquant_max_context:,} steps\n"
            f"  Desolvation Barrier Cutoff:  {self.desolvation_barrier_threshold_eV} eV\n"
        )


def get_config(
    *,
    total_memory_gb: float = 0.0,
    mlx_max_mem_gb: float = 0.0,
    metal_shader_cache_gb: float = 0.0,
) -> AureliusConfig:
    """Retrieve the dynamically-configured M-series memory layout.

    Args:
        total_memory_gb: Override total RAM (for testing).
        mlx_max_mem_gb: Override MLX memory (for testing).
        metal_shader_cache_gb: Override shader cache size (for testing).

    Returns:
        An AureliusConfig instance with dynamically computed memory allocations.
    """
    config = AureliusConfig(
        total_memory_gb=total_memory_gb,
        mlx_max_mem_gb=mlx_max_mem_gb,
        metal_shader_cache_gb=metal_shader_cache_gb,
    )
    if not config.validate_memory_budget():
        raise RuntimeError(
            f"Memory budget {config.mlx_max_mem_gb + config.metal_shader_cache_gb:.1f}GB exceeds "
            f"physical memory {config.total_memory_gb:.1f}GB. Adjust MLX_MAX_MEM_CACHE."
        )
    return config


# Backward compatibility alias
AureliusConfig = AureliusConfig


def apply_global_config() -> AureliusConfig:
    """Apply configuration globally and return for use across modules.

    Note: This function still calls apply_environment() for backward
    compatibility. In new code, prefer get_config() and apply the
    returned env vars in a thread-safe manner via __main__.py.
    """
    config = get_config()
    env_vars = config.apply_environment()
    # Apply env vars (backward-compatible; new callers should do this
    # in __main__.py instead to avoid thread-safety issues)
    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v
    print(config.memory_report())
    return config


def initialize_environment() -> dict[str, str]:
    """Initialize Aurelius environment variables before any framework imports.

    This function sets the minimum required environment variables for
    Apple Silicon optimization (PyTorch MPS async compilation, MLX
    memory budget) BEFORE torch or mlx are imported anywhere in the
    package. This eliminates the need for users to manually source
    setup_env.sh or configure their shell profiles.

    Only sets variables that are not already set, preserving user
    overrides from shell profiles, cluster job scripts, or testing.

    Returns:
        Dictionary of environment variables that were applied (only
        those that were not already present in os.environ).
    """
    config = get_config()
    env_vars = config.apply_environment()
    applied = {}
    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v
            applied[k] = v
    return applied


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

# Expected env var defaults (mirrors config.py defaults)
_EXPECTED_ENV_VARS: dict[str, str | None] = {
    "PYTORCH_MPS_ENABLE_ASYNC_COMPILATION": "1",
    "MLX_MAX_MEM_CACHE": None,  # Computed dynamically
    "AURELIUS_VERSION": "6.0.0",
    "AURELIUS_QUANT_PRESET": "MX4",
}


def validate_environment(
    config: AureliusConfig | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate that os.environ matches config defaults.

    Compares current environment variables against the expected
    values from AureliusConfig.apply_environment() and logs any
    discrepancies.

    Args:
        config: Configuration to validate against. If None, creates
            a default config.
        strict: If True, log warnings for any mismatch. If False,
            only log INFO messages.

    Returns:
        Dict with keys:
            - mismatches: list of {env_var, expected, actual} dicts
            - missing: list of env var names not set in os.environ
            - compliant: True if no mismatches (or strict=False)
    """
    if config is None:
        config = get_config()

    env_vars = config.apply_environment()
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []

    for var_name, expected_value in env_vars.items():
        actual_value = os.environ.get(var_name)
        if actual_value is None:
            missing.append(var_name)
            if strict:
                logger.warning(
                    "Environment variable '%s' is not set. Expected: %s",
                    var_name,
                    expected_value,
                )
        elif actual_value != expected_value:
            mismatches.append(
                {
                    "env_var": var_name,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
            if strict:
                logger.warning(
                    "Environment variable '%s' mismatch: "
                    "expected '%s', got '%s'. "
                    "This may cause performance or memory mismatches.",
                    var_name,
                    expected_value,
                    actual_value,
                )

    compliant = len(mismatches) == 0 and len(missing) == 0

    result: dict[str, Any] = {
        "mismatches": mismatches,
        "missing": missing,
        "compliant": compliant,
    }

    if mismatches or missing:
        logger.info(
            "Environment validation: %d mismatch(es), %d missing var(s).",
            len(mismatches),
            len(missing),
        )

    return result


def print_env_diff(config: AureliusConfig | None = None) -> None:
    """Print a human-readable diff of environment vs config defaults.

    Args:
        config: Configuration to validate against.
    """
    if config is None:
        config = get_config()

    env_vars = config.apply_environment()

    print("\n[Environment Check]")
    for var_name, expected_value in env_vars.items():
        actual_value = os.environ.get(var_name)
        if actual_value is None:
            print(f"  [MISSING]  {var_name} = (not set)")
            print(f"            Expected: {expected_value}")
        elif actual_value != expected_value:
            print(f"  [MISMATCH] {var_name} = {actual_value!r}")
            print(f"            Expected: {expected_value!r}")
        else:
            print(f"  [OK]       {var_name} = {actual_value!r}")
    print()
