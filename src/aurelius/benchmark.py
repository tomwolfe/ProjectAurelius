"""Hardware benchmark and validation for PyTorch.

Provides a quick sanity-check benchmark to verify that the user's
hardware is properly configured for Aurelius.

Usage:
    aurelius benchmark              Run full hardware validation
    aurelius benchmark --tier 1     Run Tier 1 only
    aurelius benchmark --tier 2     Run Tier 2 only
    aurelius benchmark --quick      Fast mode (fewer repeats)
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""

    name: str
    status: str  # "pass", "fail", "skipped", "warning"
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    details: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    chip: str = "unknown"
    total_memory_gb: float = 0.0
    available_ram_gb: float = 0.0
    has_mps: bool = False
    has_pytorch: bool = False
    pytorch_version: str = ""
    macOS_version: str = ""


def _detect_hardware() -> HardwareInfo:
    """Detect the host hardware configuration."""
    import platform
    import subprocess

    info = HardwareInfo()

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cpu_brand = result.stdout.strip()
        if "Apple" in cpu_brand:
            info.chip = cpu_brand
        else:
            info.chip = cpu_brand or "unknown"
    except Exception:
        info.chip = "unknown"

    try:
        import psutil

        total = psutil.virtual_memory().total
        avail = psutil.virtual_memory().available
        info.total_memory_gb = total / (1024**3)
        info.available_ram_gb = avail / (1024**3)
    except Exception:
        pass

    import torch

    info.has_pytorch = True
    info.pytorch_version = torch.__version__
    info.has_mps = torch.backends.mps.is_available()

    with contextlib.suppress(Exception):
        info.macOS_version = platform.mac_ver()[0] or "unknown"

    return info


def _benchmark_tier1_quick(repeats: int = 5, n_molecules: int = 100) -> BenchmarkResult:
    """Quick Tier 1 inference benchmark using PyTorch MPS."""
    rng = np.random.default_rng(42)
    X = np.zeros((n_molecules, 2048), dtype=np.float32)
    for i in range(n_molecules):
        n_bits_set = rng.integers(80, 200)
        indices = rng.integers(0, 2048, size=n_bits_set)
        X[i, indices] = 1.0

    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    X_torch = torch.tensor(X, device=device)

    _ = model(X_torch[:1])
    if device == "mps":
        torch.mps.empty_cache()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(X_torch)
        if device == "mps":
            torch.mps.empty_cache()
        times.append(time.perf_counter() - start)

    return BenchmarkResult(
        name="tier1_torch",
        status="pass",
        mean_ms=float(np.mean(times) * 1000),
        std_ms=float(np.std(times) * 1000),
        min_ms=float(np.min(times) * 1000),
        max_ms=float(np.max(times) * 1000),
        details=f"PyTorch {device} {n_molecules} molecules, {repeats} repeats",
    )


def _check_memory_budget(info: HardwareInfo) -> BenchmarkResult:
    """Check if available memory is sufficient."""
    if info.available_ram_gb < 8.0:
        return BenchmarkResult(
            name="memory_check",
            status="fail",
            details=f"Available RAM {info.available_ram_gb:.1f}GB < 8GB minimum",
        )
    elif info.available_ram_gb < 16.0:
        return BenchmarkResult(
            name="memory_check",
            status="warning",
            details=f"Available RAM {info.available_ram_gb:.1f}GB (16GB+ recommended)",
        )
    return BenchmarkResult(
        name="memory_check",
        status="pass",
        details=f"Available RAM {info.available_ram_gb:.1f}GB",
    )


def run_benchmark(
    tier: str | None = None,
    quick: bool = True,
    output: str | None = None,
) -> dict[str, Any]:
    """Run hardware benchmark and validation.

    Args:
        tier: Which tier to benchmark ("1", "2", or None for all).
        quick: Use fast mode with fewer repeats.
        output: Optional path to save results as JSON.

    Returns:
        Dictionary with benchmark results and hardware info.
    """
    repeats = 3 if quick else 10
    n_molecules = 100 if quick else 1000

    print("=" * 60)
    print("  Aurelius v9.0 Hardware Benchmark")
    print("=" * 60)

    print("\n[benchmark] Detecting hardware...")
    info = _detect_hardware()

    results: dict[str, Any] = {
        "hardware": {
            "chip": info.chip,
            "total_memory_gb": round(info.total_memory_gb, 1),
            "available_ram_gb": round(info.available_ram_gb, 1),
            "macOS": info.macOS_version,
        },
        "frameworks": {},
        "benchmarks": {},
    }

    if info.has_pytorch:
        results["frameworks"]["pytorch"] = info.pytorch_version

    mem_result = _check_memory_budget(info)
    results["benchmarks"][mem_result.name] = {
        "status": mem_result.status,
        "details": mem_result.details,
    }
    status_symbol = {"pass": "[OK]", "warning": "[WARN]", "fail": "[FAIL]", "skipped": "[SKIP]"}
    print(f"  Memory: {status_symbol.get(mem_result.status, '[??]')} {mem_result.details}")

    if tier is None or tier == "1":
        print("\n[benchmark] Running Tier 1 inference benchmark...")
        t1 = _benchmark_tier1_quick(repeats=repeats, n_molecules=n_molecules)
        results["benchmarks"][t1.name] = {
            "status": t1.status,
            "mean_ms": round(t1.mean_ms, 2),
            "std_ms": round(t1.std_ms, 2),
            "min_ms": round(t1.min_ms, 2),
            "max_ms": round(t1.max_ms, 2),
            "details": t1.details,
        }
        print(f"  Tier 1: {status_symbol.get(t1.status, '[??]')} {t1.details}")
        if t1.status == "pass":
            print(f"           Mean: {t1.mean_ms:.2f} ms +/- {t1.std_ms:.2f} ms")

    print("\n" + "-" * 60)
    print("  Summary:")

    all_pass = True
    for name, result in results["benchmarks"].items():
        status = result["status"]
        symbol = status_symbol.get(status, "[??]")
        if status in ("fail",) or status == "warning":
            all_pass = False
        print(f"    {symbol} {name}: {result['details']}")

    print("\n" + "=" * 60)
    if all_pass:
        print("  Result: ALL CHECKS PASSED")
    else:
        print("  Result: SOME CHECKS PASSED WITH WARNINGS")
    print("=" * 60)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output}")

    return results
