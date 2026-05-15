"""Benchmark: MLX vs PyTorch MPS vs CPU for Tier 1 inference.

Compares inference time across three backends:
1. MLX (Apple Silicon Neural Accelerator)
2. PyTorch MPS (Metal Performance Shaders)
3. PyTorch CPU

Usage:
    python benchmarks/benchmark_tier1.py --n-molecules 1000 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def generate_test_fingerprints(n: int, n_bits: int = 2048, seed: int = 42) -> np.ndarray:
    """Generate random ECFP4-like fingerprints for benchmarking.

    Args:
        n: Number of fingerprints.
        n_bits: Fingerprint size.
        seed: Random seed.

    Returns:
        numpy array of shape (n, n_bits).
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n, n_bits), dtype=np.float32)
    for i in range(n):
        n_bits_set = rng.randint(80, 200)
        indices = rng.randint(0, n_bits, size=n_bits_set)
        X[i, indices] = 1.0
    return X


def benchmark_mlx(X: np.ndarray, repeats: int = 10) -> dict:
    """Benchmark inference on MLX backend.

    Args:
        X: Input fingerprints.
        repeats: Number of repeats.

    Returns:
        Dictionary with timing results.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return {"status": "not_available", "message": "MLX not installed"}

    model = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )

    X_mx = mx.array(X)

    # Warmup
    _ = model(X_mx[:1])
    mx.metal.clear_cache()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(X_mx)
        mx.metal.clear_cache()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "status": "success",
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
    }


def benchmark_pytorch_mps(X: np.ndarray, repeats: int = 10) -> dict:
    """Benchmark inference on PyTorch MPS backend.

    Args:
        X: Input fingerprints.
        repeats: Number of repeats.

    Returns:
        Dictionary with timing results.
    """
    try:
        import torch
    except ImportError:
        return {"status": "not_available", "message": "PyTorch not installed"}

    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    X_torch = torch.tensor(X, device=device)

    # Warmup
    _ = model(X_torch[:1])
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(X_torch)
        if device == "mps":
            torch.mps.empty_cache()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "status": "success",
        "backend": "mps" if torch.backends.mps.is_available() else "cpu",
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
    }


def benchmark_pytorch_cpu(X: np.ndarray, repeats: int = 10) -> dict:
    """Benchmark inference on PyTorch CPU.

    Args:
        X: Input fingerprints.
        repeats: Number of repeats.

    Returns:
        Dictionary with timing results.
    """
    try:
        import torch
    except ImportError:
        return {"status": "not_available", "message": "PyTorch not installed"}

    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )

    X_torch = torch.tensor(X, dtype=torch.float32)

    # Warmup
    _ = model(X_torch[:1])

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(X_torch)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "status": "success",
        "backend": "cpu",
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Tier 1 inference")
    parser.add_argument("--n-molecules", type=int, default=1000, help="Number of test molecules")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per benchmark")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"[benchmark] Generating {args.n_molecules} test fingerprints...")
    X = generate_test_fingerprints(args.n_molecules)

    results = {
        "n_molecules": args.n_molecules,
        "repeats": args.repeats,
    }

    print("[benchmark] Running MLX benchmark...")
    results["mlx"] = benchmark_mlx(X, args.repeats)

    print("[benchmark] Running PyTorch MPS benchmark...")
    results["pytorch_mps"] = benchmark_pytorch_mps(X, args.repeats)

    print("[benchmark] Running PyTorch CPU benchmark...")
    results["pytorch_cpu"] = benchmark_pytorch_cpu(X, args.repeats)

    # Print summary
    print("\n" + "=" * 60)
    print("  Tier 1 Inference Benchmark Results")
    print("=" * 60)

    for backend_name, backend_results in results.items():
        if isinstance(backend_results, dict) and backend_results.get("status") == "success":
            print(f"\n  {backend_name}:")
            print(f"    Mean: {backend_results['mean_ms']:.2f} ms +/- {backend_results['std_ms']:.2f} ms")
            print(f"    Min:  {backend_results['min_ms']:.2f} ms")
            print(f"    Max:  {backend_results['max_ms']:.2f} ms")

    # Compute speedups
    print("\n" + "-" * 60)
    print("  Speedup Summary:")

    cpu_result = results.get("pytorch_cpu", {})
    if cpu_result.get("status") == "success":
        cpu_time = cpu_result["mean_ms"]

        for name, res in results.items():
            if name == "pytorch_cpu" or res.get("status") != "success":
                continue
            speedup = cpu_time / res["mean_ms"] if res["mean_ms"] > 0 else float("inf")
            print(f"    {name} vs CPU: {speedup:.2f}x faster")

        # MLX vs MPS
        mlx_r = results.get("mlx", {})
        mps_r = results.get("pytorch_mps", {})
        if mlx_r.get("status") == "success" and mps_r.get("status") == "success":
            speedup = mps_r["mean_ms"] / mlx_r["mean_ms"]
            print(f"    MLX vs MPS: {speedup:.2f}x")

    print("=" * 60)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
