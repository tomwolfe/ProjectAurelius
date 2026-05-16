"""Hardware benchmark and validation for Apple Silicon.

Provides a quick sanity-check benchmark to verify that the user's
Apple Silicon hardware is properly configured for Aurelius.

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

from aurelius.constants import COULOMB_EV_A


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
    has_mlx: bool = False
    has_pytorch: bool = False
    pytorch_version: str = ""
    mlx_version: str = ""
    macOS_version: str = ""


def _detect_hardware() -> HardwareInfo:
    """Detect the host hardware configuration."""
    import platform
    import subprocess

    info = HardwareInfo()

    # Try to detect chip
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        cpu_brand = result.stdout.strip()
        if "Apple" in cpu_brand:
            info.chip = cpu_brand
        else:
            info.chip = cpu_brand or "unknown"
    except Exception:
        info.chip = "unknown"

    # Memory
    try:
        import psutil
        total = psutil.virtual_memory().total
        avail = psutil.virtual_memory().available
        info.total_memory_gb = total / (1024 ** 3)
        info.available_ram_gb = avail / (1024 ** 3)
    except Exception:
        pass

    # PyTorch MPS
    try:
        import torch
        info.has_pytorch = True
        info.pytorch_version = torch.__version__
        info.has_mps = torch.backends.mps.is_available()
    except ImportError:
        pass

    # MLX
    try:
        import mlx
        info.has_mlx = True
        info.mlx_version = mlx.__version__  # type: ignore[attr-defined]
    except ImportError:
        pass

    # macOS version
    with contextlib.suppress(Exception):
        info.macOS_version = platform.mac_ver()[0] or "unknown"

    return info


def _benchmark_tier1_quick(repeats: int = 5, n_molecules: int = 100) -> BenchmarkResult:
    """Quick Tier 1 inference benchmark."""
    rng = np.random.RandomState(42)
    X = np.zeros((n_molecules, 2048), dtype=np.float32)
    for i in range(n_molecules):
        n_bits_set = rng.randint(80, 200)
        indices = rng.randint(0, 2048, size=n_bits_set)
        X[i, indices] = 1.0

    # Try MLX first
    try:
        import mlx.core as mx
        import mlx.nn as nn

        model = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(2048, 128),  # type: ignore[attr-defined]
            nn.ReLU(),  # type: ignore[attr-defined]
            nn.Linear(128, 1),  # type: ignore[attr-defined]
        )
        X_mx = mx.array(X)

        # Warmup
        _ = model(X_mx[:1])
        mx.metal.clear_cache()

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(X_mx)
            mx.metal.clear_cache()
            times.append(time.perf_counter() - start)

        return BenchmarkResult(
            name="tier1_mlx",
            status="pass",
            mean_ms=float(np.mean(times) * 1000),
            std_ms=float(np.std(times) * 1000),
            min_ms=float(np.min(times) * 1000),
            max_ms=float(np.max(times) * 1000),
            details=f"MLX {n_molecules} molecules, {repeats} repeats",
        )
    except ImportError:
        pass

    # Try PyTorch MPS
    try:
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        X_torch = torch.tensor(X, device=device)

        # Warmup
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

        status = "pass" if device == "mps" else "warning"
        detail = f"PyTorch {device} ({n_molecules} molecules, {repeats} repeats)"
        if device == "cpu":
            detail += " - MPS not available, running on CPU"

        return BenchmarkResult(
            name="tier1_pytorch",
            status=status,
            mean_ms=float(np.mean(times) * 1000),
            std_ms=float(np.std(times) * 1000),
            min_ms=float(np.min(times) * 1000),
            max_ms=float(np.max(times) * 1000),
            details=detail,
        )
    except ImportError:
        pass

    return BenchmarkResult(
        name="tier1",
        status="skipped",
        details="No MLX or PyTorch available",
    )


def _benchmark_tier2_quick(repeats: int = 3, n_atoms: int = 30) -> BenchmarkResult:
    """Quick Tier 2 vectorized physics benchmark."""
    rng = np.random.RandomState(42)
    elements = [1, 6, 7, 8, 11]
    atomic_numbers = np.array([elements[rng.randint(0, len(elements))] for _ in range(n_atoms)])

    radius = 5.0
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = rng.uniform(0.5, radius)
        coords[i, 0] = r * np.sin(phi) * np.cos(theta)
        coords[i, 1] = r * np.sin(phi) * np.sin(theta)
        coords[i, 2] = r * np.cos(phi)

    diffs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
    distances = np.linalg.norm(diffs, axis=-1)

    _LJ_PARAMS = {
        (1, 1): (0.015, 2.650),
        (1, 6): (0.01500, 2.650),
        (1, 8): (0.01700, 2.750),
        (1, 11): (0.00800, 2.500),
        (6, 6): (0.01094, 3.3996),
        (6, 8): (0.01200, 3.1500),
        (6, 11): (0.02000, 3.000),
        (8, 8): (0.01200, 3.1200),
        (8, 11): (0.03000, 2.800),
        (11, 11): (0.00800, 2.500),
    }
    _CHARGES = {1: 0.0, 6: 0.0, 7: 0.0, 8: -0.417, 11: 0.889}

    # Vectorized LJ
    z_i = atomic_numbers[np.newaxis, :]
    z_j = atomic_numbers[:, np.newaxis]
    z_min = np.minimum(z_i, z_j)
    z_max = np.maximum(z_i, z_j)
    eps_tensor = np.zeros((n_atoms, n_atoms))
    sig_tensor = np.zeros((n_atoms, n_atoms))
    for (zi, zj), (eps, sig) in _LJ_PARAMS.items():
        pair_mask = (z_min == zi) & (z_max == zj)
        eps_tensor = np.where(pair_mask, eps, eps_tensor)
        sig_tensor = np.where(pair_mask, sig, sig_tensor)
    eps_tensor = np.where(eps_tensor == 0, 0.02, eps_tensor)
    sig_tensor = np.where(sig_tensor == 0, 2.5, sig_tensor)
    mask = np.triu(np.ones((n_atoms, n_atoms)), k=1)
    cutoff_mask = (distances < 12.0) & mask
    r_soft = np.sqrt(distances * distances + sig_tensor ** 2)
    sig_over_r = sig_tensor / r_soft
    sig_over_r6 = sig_over_r ** 6
    lj = 4.0 * eps_tensor * (sig_over_r6 ** 2 - sig_over_r6)
    lj_energy = float(np.sum(lj * cutoff_mask))

    # Vectorized Coulomb
    charges = np.array([_CHARGES.get(z, 0.0) for z in atomic_numbers])
    q_i = charges[np.newaxis, :]
    q_j = charges[:, np.newaxis]
    q_product = q_i * q_j
    charge_mask = (q_product != 0.0)
    r_soft_c = np.sqrt(distances * distances + 1.0)
    coul_energy = float(np.sum(COULOMB_EV_A * q_product / r_soft_c * mask * charge_mask))

    # Time the vectorized computation
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        # Recompute to get wall time
        r_soft2 = np.sqrt(distances * distances + sig_tensor ** 2)
        sig_over_r2 = sig_tensor / r_soft2
        sig_over_r6_2 = sig_over_r2 ** 6
        lj2 = 4.0 * eps_tensor * (sig_over_r6_2 ** 2 - sig_over_r6_2)
        _ = float(np.sum(lj2 * cutoff_mask))
        charges2 = np.array([_CHARGES.get(z, 0.0) for z in atomic_numbers])
        q_i2 = charges2[np.newaxis, :]
        q_j2 = charges2[:, np.newaxis]
        q_product2 = q_i2 * q_j2
        charge_mask2 = (q_product2 != 0.0)
        r_soft_c2 = np.sqrt(distances * distances + 1.0)
        _ = float(np.sum(COULOMB_EV_A * q_product2 / r_soft_c2 * mask * charge_mask2))
        times.append(time.perf_counter() - start)

    mean_s = float(np.mean(times))
    status = "pass" if mean_s < 1.0 else "warning"

    return BenchmarkResult(
        name="tier2_vectorized",
        status=status,
        mean_ms=mean_s * 1000,
        std_ms=float(np.std(times) * 1000),
        min_ms=float(np.min(times) * 1000),
        max_ms=float(np.max(times) * 1000),
        details=f"Vectorized LJ+Coulomb, {n_atoms} atoms, {repeats} repeats",
        metadata={
            "lj_energy_eV": lj_energy,
            "coulomb_energy_eV": coul_energy,
            "n_atoms": n_atoms,
        },
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
    n_atoms = 30 if quick else 50

    print("=" * 60)
    print("  Aurelius v5.2 Hardware Benchmark")
    print("=" * 60)

    # Hardware detection
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
    if info.has_mlx:
        results["frameworks"]["mlx"] = info.mlx_version

    # Memory check
    mem_result = _check_memory_budget(info)
    results["benchmarks"][mem_result.name] = {
        "status": mem_result.status,
        "details": mem_result.details,
    }
    status_symbol = {"pass": "[OK]", "warning": "[WARN]", "fail": "[FAIL]", "skipped": "[SKIP]"}
    print(f"  Memory: {status_symbol.get(mem_result.status, '[??]')} {mem_result.details}")

    # Tier 1 benchmark
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

    # Tier 2 benchmark
    if tier is None or tier == "2":
        print("\n[benchmark] Running Tier 2 physics benchmark...")
        t2 = _benchmark_tier2_quick(repeats=repeats, n_atoms=n_atoms)
        results["benchmarks"][t2.name] = {
            "status": t2.status,
            "mean_ms": round(t2.mean_ms, 2),
            "std_ms": round(t2.std_ms, 2),
            "min_ms": round(t2.min_ms, 2),
            "max_ms": round(t2.max_ms, 2),
            "details": t2.details,
        }
        print(f"  Tier 2: {status_symbol.get(t2.status, '[??]')} {t2.details}")
        if t2.status == "pass":
            print(f"           Mean: {t2.mean_ms:.2f} ms")
        lj_e = t2.metadata.get("lj_energy_eV", 0)
        coul_e = t2.metadata.get("coulomb_energy_eV", 0)
        print(f"           LJ energy: {lj_e:.4f} eV, Coulomb: {coul_e:.4f} eV")

    # Summary
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

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output}")

    return results
