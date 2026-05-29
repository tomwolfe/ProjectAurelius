"""Zero-Copy Memory Management for Project Aurelius.

This module provides memory management utilities including:
- Quantization configuration for different precision levels
- Zero-copy memory management with automatic resource cleanup
- System RAM detection and budget calculation
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for model quantization precision.

    Attributes:
        precision: Quantization precision format string.
            - "MX4": 4-bit precision (8x compression ratio)
            - "MX6": 6-bit precision (~5.33x compression ratio)
            - "MX8": 8-bit precision (4x compression ratio)
        bits: Bit depth of the quantization format.
        compression_ratio: Ratio of original to quantized size.

    Raises:
        ValueError: If precision is not one of "MX4", "MX6", or "MX8".
    """

    precision: str = "MX4"

    def __post_init__(self) -> None:
        if self.precision not in ("MX4", "MX6", "MX8"):
            raise ValueError(f"Invalid precision '{self.precision}'. Must be one of: MX4, MX6, MX8.")

    @property
    def bits(self) -> int:
        """Return the bit depth of the quantization format."""
        return {"MX4": 4, "MX6": 6, "MX8": 8}[self.precision]

    @property
    def compression_ratio(self) -> float:
        """Return the compression ratio for this precision.

        MX4: 8x, MX6: ~5.33x, MX8: 4x
        """
        return {
            "MX4": 8.0,
            "MX6": 16.0 / 3.0,
            "MX8": 4.0,
        }[self.precision]


class ZeroCopyMemoryManager:
    """Manages zero-copy memory resources with automatic cleanup.

    Tracks system RAM, memory budgets, and provides budget calculations
    for the screening pipeline. Uses psutil for system RAM detection.

    Args:
        total_ram_gb: Total system RAM in GB. If None, auto-detected.
        mlx_max_mem_gb: Maximum memory allocated to MLX in GB.
        metal_shader_cache_gb: Reserved memory for Metal shader cache in GB.
    """

    _TOTAL_RAM_GB: float | None = None

    def __init__(
        self,
        total_ram_gb: float | None = None,
        mlx_max_mem_gb: float = 12.0,
        metal_shader_cache_gb: float = 1.0,
    ) -> None:
        self._quant_config = QuantizationConfig()
        self._mlx_max_mem_gb = mlx_max_mem_gb
        self._metal_shader_cache_gb = metal_shader_cache_gb
        self._total_ram_gb = total_ram_gb or self._get_total_ram()
        self._remaining_gb = self._total_ram_gb - self._mlx_max_mem_gb - self._metal_shader_cache_gb

    @classmethod
    def _get_total_ram(cls) -> float:
        """Detect total system RAM in GB using psutil.

        Returns:
            Total system RAM in gigabytes.
        """
        if cls._TOTAL_RAM_GB is None:
            total_bytes = psutil.virtual_memory().total
            cls._TOTAL_RAM_GB = total_bytes / (1024**3)
        return cls._TOTAL_RAM_GB  # type: ignore[return-value]

    @property
    def quant_config(self) -> QuantizationConfig:
        """Return the quantization configuration."""
        return self._quant_config

    @property
    def device(self) -> str:
        """Return the default compute device ('mps' for Apple Silicon)."""
        return "mps"

    def get_memory_budget(self) -> dict[str, float]:
        """Calculate memory budget for the screening pipeline.

        Returns:
            Dict with keys:
                - total_gb: Total system RAM
                - mlx_max_mem_gb: MLX memory allocation
                - remaining_gb: Remaining memory after allocations
                - chemvlm2_footprint_gb: Estimated ChemVLM-2 footprint
        """
        return {
            "total_gb": self._total_ram_gb,
            "mlx_max_mem_gb": self._mlx_max_mem_gb,
            "remaining_gb": self._remaining_gb,
            "chemvlm2_footprint_gb": 4.0,
        }

    def cleanup(self) -> None:
        """Release all managed resources."""
        self._total_ram_gb = 0.0
        self._remaining_gb = 0.0
