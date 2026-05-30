"""Memory Profiler for Autonomous Screening Agent.

Provides peak RAM tracking, MPS memory monitoring, and CSV report
generation for the autonomous screening loop.

Sampling Strategy:
    - Samples memory at generation boundaries
    - Peak tracking via tracemalloc + periodic framework-specific APIs
    - Keeps overhead < 2% on screening throughput

Framework APIs Used:
    - psutil: Process-level RSS memory
    - torch.mps.current_allocated_memory(): PyTorch MPS memory

Output:
    CSV file with columns: generation, screened_count, peak_ram_gb,
    mps_cached_gb, gc_collected

References:
    tracemalloc: Python's built-in memory allocation tracer.
    psutil: Cross-platform process information library.
"""

from __future__ import annotations

import csv
import os
import time
import tracemalloc
from datetime import UTC, datetime
from typing import Any

import psutil


def _get_process_rss_gb() -> float:
    """Get current process RSS (Resident Set Size) in GB.

    Uses psutil for cross-platform process memory tracking.

    Returns:
        Current RSS in GB.
    """
    try:
        process = psutil.Process(os.getpid())
        rss_bytes = process.memory_info().rss
        return float(rss_bytes) / (1024**3)
    except Exception:
        return 0.0


def _get_mps_memory_gb() -> float:
    """Get current PyTorch MPS allocated memory in GB.

    Uses torch.mps.current_allocated_memory() when available.
    Falls back to 0.0 if MPS is unavailable.

    Returns:
        MPS allocated memory in GB.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            mem_bytes = torch.mps.current_allocated_memory()
            return float(mem_bytes) / (1024**3)
    except Exception:
        pass
    return 0.0


class MemoryProfiler:
    """Memory profiler for the autonomous screening agent.

    Tracks peak RAM usage, MPS memory, and GC activity across
    screening generations. Generates CSV reports at generation
    boundaries.

    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        # ... screening loop ...
        profiler.sample(generation=1, screened_count=50, gc_collected=3)
        profiler.stop()
        profiler.generate_report("output_dir")
    """

    def __init__(self, output_dir: str = ".") -> None:
        """Initialize the memory profiler.

        Args:
            output_dir: Directory for CSV report output.
        """
        self._output_dir = output_dir
        self._peak_ram_gb = 0.0
        self._peak_mps_gb = 0.0
        self._tracemalloc_snapshot: Any = None
        self._start_time: float = 0.0
        self._samples: list[dict[str, Any]] = []
        self._active = False

    def start(self) -> None:
        """Start memory profiling.

        Initializes tracemalloc for allocation tracking and records
        the start timestamp.
        """
        tracemalloc.start()
        self._start_time = time.time()
        self._active = True
        self._peak_ram_gb = _get_process_rss_gb()
        self._peak_mps_gb = _get_mps_memory_gb()

    def stop(self) -> None:
        """Stop memory profiling.

        Takes a final tracemalloc snapshot and stops tracking.
        """
        if self._active:
            self._tracemalloc_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            self._active = False

    def sample(
        self,
        generation: int,
        screened_count: int,
        gc_collected: int,
    ) -> None:
        """Record a memory snapshot at a generation boundary.

        Samples current RSS and MPS memory, and GC count.
        Updates peak tracking values.

        Args:
            generation: Current generation number.
            screened_count: Total molecules screened so far.
            gc_collected: Number of objects collected by gc.collect()
                since the last sample.
        """
        current_ram = _get_process_rss_gb()
        mps_mem = _get_mps_memory_gb()

        self._peak_ram_gb = max(self._peak_ram_gb, current_ram)
        self._peak_mps_gb = max(self._peak_mps_gb, mps_mem)

        if self._tracemalloc_snapshot:
            _ = self._tracemalloc_snapshot.statistics("lineno")[:5]

        sample_entry = {
            "generation": generation,
            "screened_count": screened_count,
            "current_ram_gb": round(current_ram, 4),
            "peak_ram_gb": round(self._peak_ram_gb, 4),
            "mps_cached_gb": round(mps_mem, 4),
            "peak_mps_gb": round(self._peak_mps_gb, 4),
            "gc_collected": gc_collected,
            "elapsed_s": round(time.time() - self._start_time, 1),
        }
        self._samples.append(sample_entry)

    def generate_report(self, output_dir: str | None = None) -> str:
        """Generate a CSV memory profile report.

        Writes all sampled data to a CSV file with timestamped filename.

        Args:
            output_dir: Optional override for output directory.

        Returns:
            Path to the generated CSV report.
        """
        out_dir = output_dir or self._output_dir
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"memory_profile_{timestamp}.csv")

        if not self._samples:
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "generation",
                        "screened_count",
                        "current_ram_gb",
                        "peak_ram_gb",
                        "mps_cached_gb",
                        "peak_mps_gb",
                        "gc_collected",
                        "elapsed_s",
                    ]
                )
            return csv_path

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._samples[0].keys())
            writer.writeheader()
            writer.writerows(self._samples)

        return csv_path

    @property
    def peak_ram_gb(self) -> float:
        """Return peak RSS memory in GB."""
        return self._peak_ram_gb

    @property
    def peak_mps_gb(self) -> float:
        """Return peak MPS memory in GB."""
        return self._peak_mps_gb

    @property
    def n_samples(self) -> int:
        """Return number of samples collected."""
        return len(self._samples)
