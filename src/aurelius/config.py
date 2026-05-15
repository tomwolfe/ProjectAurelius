"""Environment configuration for M5 Pro (24GB) hard-partitioning.

This module ensures MLX (infrared) and PyTorch (MD simulation) do not
compete for the same memory pages, preventing kernel panics during
concurrent AI/Simulation workloads.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class M5ProConfig:
    """Hard-partitioned memory configuration for M5 Pro (24GB).

    MLX_MAX_MEM_CACHE: Reserved for MLX Neural Accelerator inference.
    PYTORCH_MPS_ASYNC: Enables async Metal compilation to avoid JIT lag.
    """

    # Total physical memory (24GB for M5 Pro)
    total_memory_gb: int = 24

    # MLX memory allocation for ChemVLM-2 MX4 inference
    mlx_max_mem_gb: int = 12

    # PyTorch MPS async compilation (disables JIT compilation lag)
    pytorch_mps_async: bool = True

    # Metal-4 shader pre-compilation buffer
    metal_shader_cache_gb: int = 2

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

    def apply_environment(self) -> None:
        """Apply hard-partitioning environment variables."""
        os.environ["PYTORCH_MPS_ENABLE_ASYNC_COMPILATION"] = "1" if self.pytorch_mps_async else "0"
        os.environ["MLX_MAX_MEM_CACHE"] = f"{self.mlx_max_mem_gb}G"
        os.environ["AURELIUS_VERSION"] = "5.1.0"
        os.environ["AURELIUS_QUANT_PRESET"] = self.chemvlm_quantization

    def validate_memory_budget(self) -> bool:
        """Validate that memory budget fits within physical RAM."""
        used = self.mlx_max_mem_gb + self.metal_shader_cache_gb
        # PyTorch MPS will use remaining ~10GB
        remaining = self.total_memory_gb - used
        return remaining >= 8  # Minimum 8GB for PyTorch MPS

    def memory_report(self) -> str:
        """Generate a human-readable memory partition report."""
        mlx_reserved = self.mlx_max_mem_gb
        shader_reserved = self.metal_shader_cache_gb
        pytorch_available = self.total_memory_gb - mlx_reserved - shader_reserved
        return (
            f"=== Aurelius v5.1 Memory Partition (M5 Pro {self.total_memory_gb}GB) ===\n"
            f"  MLX (ChemVLM-2 MX4):         {mlx_reserved:>3}GB reserved\n"
            f"  Metal Shader Cache:           {shader_reserved:>3}GB reserved\n"
            f"  PyTorch MPS (MatterSim+GCMD): ~{pytorch_available:>3}GB available\n"
            f"  TurboQuant Context Window:    {self.turquant_max_context:,} tokens\n"
            f"  Desolvation Barrier Cutoff:   {self.desolvation_barrier_threshold_eV} eV\n"
        )


def get_config() -> M5ProConfig:
    """Retrieve the default M5 Pro hard-partitioned configuration."""
    config = M5ProConfig()
    config.apply_environment()
    if not config.validate_memory_budget():
        raise RuntimeError(
            f"Memory budget {config.mlx_max_mem_gb + config.metal_shader_cache_gb}GB exceeds "
            f"physical memory {config.total_memory_gb}GB. Adjust MLX_MAX_MEM_CACHE."
        )
    return config


def apply_global_config() -> M5ProConfig:
    """Apply configuration globally and return for use across modules."""
    config = get_config()
    print(config.memory_report())
    return config
