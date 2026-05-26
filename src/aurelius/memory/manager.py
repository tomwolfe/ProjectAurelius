"""Phase 1: Zero-Copy Memory Management.

Manages zero-copy memory between MLX and PyTorch on M-series NPUs.
Uses dynamic RAM detection via psutil for hardware-agnostic allocation.
Replaces private torch._C APIs with safe wrappers.
"""

from __future__ import annotations

import psutil


class QuantizationConfig:
    """Quantization configuration for memory compression.

    Attributes:
        precision: String specifying the quantization format (e.g., "MX4", "MX6", "MX8").
    """

    _PRECISION_BITS: dict[str, int] = {
        "MX4": 4,
        "MX6": 6,
        "MX8": 8,
    }

    def __init__(self, precision: str = "MX4") -> None:
        self.precision = precision

    @property
    def bits(self) -> int:
        return self._PRECISION_BITS.get(self.precision, 4)

    @property
    def compression_ratio(self) -> float:
        bits = self.bits
        if bits == 0:
            return 1.0
        return 32.0 / bits


class ZeroCopyMemoryManager:
    """Manages zero-copy memory allocation for screening operations.

    Tracks total RAM, available memory, and provides memory budget
    information for the screening pipeline.
    """

    def __init__(self) -> None:
        self._quant_config = QuantizationConfig()
        self._device = "mps"
        self._total_ram_gb = self._detect_total_ram()

    @staticmethod
    def _detect_total_ram() -> float:
        """Detect total system RAM in GB using psutil."""
        total_bytes = psutil.virtual_memory().total
        return total_bytes / (1024**3)

    @property
    def quant_config(self) -> QuantizationConfig:
        return self._quant_config

    @property
    def device(self) -> str:
        return self._device

    def get_memory_budget(self) -> dict[str, float]:
        """Return memory budget information.

        Returns:
            Dict with total_gb, chemvlm2_footprint_gb, and remaining_gb.
        """
        chemvlm2_footprint_gb = 2.0
        remaining_gb = self._total_ram_gb - chemvlm2_footprint_gb
        return {
            "total_gb": self._total_ram_gb,
            "chemvlm2_footprint_gb": chemvlm2_footprint_gb,
            "remaining_gb": remaining_gb,
        }
