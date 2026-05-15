"""Dynamic environment configuration for Apple M-series memory management.

Detects system RAM dynamically using psutil and allocates memory
across MLX, Metal Shader Cache, and PyTorch MPS without hardcoding
for a specific chip generation.

Memory allocation strategy:
    - MLX:          50% of available RAM (capped at 12GB)
    - Metal Shader: 10% of available RAM (capped at 2GB)
    - PyTorch MPS:  Remaining RAM
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass(frozen=True)
class M5ProConfig:
    """Dynamic memory configuration for Apple M-series chips.

    Detects total system RAM at instantiation and computes
    tiered allocations accordingly.
    """

    # Dynamically set from psutil at construction time
    total_memory_gb: float = 0.0

    # MLX memory allocation (50% of RAM, capped at 12GB)
    mlx_max_mem_gb: float = 0.0

    # PyTorch MPS async compilation
    pytorch_mps_async: bool = True

    # Metal shader pre-compilation buffer (10% of RAM, capped at 2GB)
    metal_shader_cache_gb: float = 0.0

    # GCMD TurboQuant context window limit (tokens)
    turquant_max_context: int = 8192

    # Desolvation energy barrier rejection threshold (eV)
    desolvation_barrier_threshold_eV: float = 0.5

    # MWSE solvent exchange rate screening window (ps)
    kex_screening_window_ps: float = 10.0

    # Aurelius Score weights (v5.1 formula)
    weight_sigma: float = 0.3
    weight_desolvation_barrier: float = 0.2
    weight_sei_homogeneity: float = 0.2
    weight_mx_synthesis_score: float = 0.2
    weight_gwp: float = 0.1

    # Quantization presets
    chemvlm_quantization: str = "MX4"
    mattersim_quantization: str = "MX4"
    gcmd_quantization: str = "TurboQuant"

    # Screening pipeline tiers
    tier1_mlxfilter_enabled: bool = True
    tier2_mattersim_enabled: bool = True
    tier3_gcmtwin_enabled: bool = True

    def __post_init__(self) -> None:
        """Compute dynamic memory allocations from system RAM."""
        # Detect total system RAM
        total_bytes = psutil.virtual_memory().total
        total_gb = total_bytes / (1024 ** 3)

        # MLX: 50% of RAM, capped at 12GB
        mlx_alloc = min(total_gb * 0.5, 12.0)

        # Metal Shader Cache: 10% of RAM, capped at 2GB
        shader_alloc = min(total_gb * 0.1, 2.0)

        # PyTorch MPS gets the remainder
        pytorch_available = total_gb - mlx_alloc - shader_alloc

        # Replace frozen fields via object.__setattr__
        object.__setattr__(self, "total_memory_gb", total_gb)
        object.__setattr__(self, "mlx_max_mem_gb", round(mlx_alloc, 1))
        object.__setattr__(self, "metal_shader_cache_gb", round(shader_alloc, 1))
        object.__setattr__(self, "_pytorch_available_gb", round(pytorch_available, 1))

    def apply_environment(self) -> None:
        """Apply hard-partitioning environment variables."""
        os.environ["PYTORCH_MPS_ENABLE_ASYNC_COMPILATION"] = "1" if self.pytorch_mps_async else "0"
        os.environ["MLX_MAX_MEM_CACHE"] = f"{int(self.mlx_max_mem_gb)}G"
        os.environ["AURELIUS_VERSION"] = "5.1.0"
        os.environ["AURELIUS_QUANT_PRESET"] = self.chemvlm_quantization

    def validate_memory_budget(self) -> bool:
        """Validate that memory budget fits within physical RAM."""
        used = self.mlx_max_mem_gb + self.metal_shader_cache_gb
        remaining = self.total_memory_gb - used
        return remaining >= 8.0  # Minimum 8GB for PyTorch MPS

    def memory_report(self) -> str:
        """Generate a human-readable memory partition report."""
        mlx_reserved = self.mlx_max_mem_gb
        shader_reserved = self.metal_shader_cache_gb
        pytorch_available = self.total_memory_gb - mlx_reserved - shader_reserved
        return (
            f"=== Aurelius v5.1 Memory Partition ({self.total_memory_gb:.0f}GB system RAM) ===\n"
            f"  MLX (ChemVLM-2 MX4):         {mlx_reserved:>5.1f}GB reserved\n"
            f"  Metal Shader Cache:          {shader_reserved:>5.1f}GB reserved\n"
            f"  PyTorch MPS (MatterSim+GCMD): {pytorch_available:>5.1f}GB available\n"
            f"  TurboQuant Context Window:   {self.turquant_max_context:,} tokens\n"
            f"  Desolvation Barrier Cutoff:  {self.desolvation_barrier_threshold_eV} eV\n"
        )

    @property
    def pytorch_available_gb(self) -> float:
        """Return PyTorch MPS available memory in GB."""
        return self._pytorch_available_gb  # type: ignore[attr-defined]


def get_config() -> M5ProConfig:
    """Retrieve the dynamically-configured M-series memory layout."""
    config = M5ProConfig()
    config.apply_environment()
    if not config.validate_memory_budget():
        raise RuntimeError(
            f"Memory budget {config.mlx_max_mem_gb + config.metal_shader_cache_gb:.1f}GB exceeds "
            f"physical memory {config.total_memory_gb:.1f}GB. Adjust MLX_MAX_MEM_CACHE."
        )
    return config


def apply_global_config() -> M5ProConfig:
    """Apply configuration globally and return for use across modules."""
    config = get_config()
    print(config.memory_report())
    return config
