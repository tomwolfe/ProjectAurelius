"""Benchmark: Vectorized vs Loop-based Physics for Tier 2.

Compares fully vectorized physics computation against
loop-based computation for Tier 2 desolvation simulation.

Usage:
    python benchmarks/benchmark_tier2.py --n-atoms 50 --n-cycles 500
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aurelius.constants import COULOMB_EV_A


def generate_random_molecule(n_atoms: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random molecular structure for benchmarking.

    Args:
        n_atoms: Number of atoms.
        seed: Random seed.

    Returns:
        Tuple of (atomic_numbers, coordinates).
    """
    rng = np.random.RandomState(seed)
    # Common elements: H(1), C(6), N(7), O(8), Na(11)
    elements = [1, 6, 7, 8, 11]
    atomic_numbers = np.array([elements[rng.randint(0, len(elements))] for _ in range(n_atoms)])

    # Random coordinates in a sphere
    radius = 5.0
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = rng.uniform(0.5, radius)
        coords[i, 0] = r * np.sin(phi) * np.cos(theta)
        coords[i, 1] = r * np.sin(phi) * np.sin(theta)
        coords[i, 2] = r * np.cos(phi)

    return atomic_numbers, coords


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


def compute_lj_loop(atomic_numbers: np.ndarray, distances: np.ndarray) -> float:
    """Compute LJ energy using Python loops (baseline).

    Args:
        atomic_numbers: Array of atomic numbers.
        distances: Pairwise distance matrix.

    Returns:
        Total LJ energy.
    """
    n = len(atomic_numbers)
    total = 0.0
    cutoff = 12.0

    for i in range(n):
        for j in range(i + 1, n):
            r = distances[i, j]
            if r >= cutoff:
                continue

            zi, zj = min(atomic_numbers[i], atomic_numbers[j]), max(atomic_numbers[i], atomic_numbers[j])
            key = (zi, zj)
            eps, sig = _LJ_PARAMS.get(key, (0.02, 2.5))

            r_soft = np.sqrt(r * r + sig * sig)
            sig_over_r = sig / r_soft
            sig_over_r6 = sig_over_r ** 6
            lj = 4.0 * eps * (sig_over_r6 ** 2 - sig_over_r6)
            total += lj

    return total


def compute_lj_vectorized(atomic_numbers: np.ndarray, distances: np.ndarray) -> float:
    """Compute LJ energy using numpy vectorized operations.

    Args:
        atomic_numbers: Array of atomic numbers.
        distances: Pairwise distance matrix.

    Returns:
        Total LJ energy.
    """
    n = len(atomic_numbers)

    # Build pairwise atomic number matrix
    z_i = atomic_numbers[np.newaxis, :]
    z_j = atomic_numbers[:, np.newaxis]
    z_min = np.minimum(z_i, z_j)
    z_max = np.maximum(z_i, z_j)

    # Lookup LJ parameters via broadcasting
    eps_tensor = np.zeros((n, n))
    sig_tensor = np.zeros((n, n))

    for (zi, zj), (eps, sig) in _LJ_PARAMS.items():
        pair_mask = (z_min == zi) & (z_max == zj)
        eps_tensor = np.where(pair_mask, eps, eps_tensor)
        sig_tensor = np.where(pair_mask, sig, sig_tensor)

    # Default for unknown pairs
    eps_tensor = np.where(eps_tensor == 0, 0.02, eps_tensor)
    sig_tensor = np.where(sig_tensor == 0, 2.5, sig_tensor)

    # Mask for cutoff and upper triangle
    mask = np.triu(np.ones((n, n)), k=1)
    cutoff_mask = (distances < 12.0) & mask

    # Vectorized LJ computation
    r_soft = np.sqrt(distances * distances + sig_tensor ** 2)
    sig_over_r = sig_tensor / r_soft
    sig_over_r6 = sig_over_r ** 6
    lj = 4.0 * eps_tensor * (sig_over_r6 ** 2 - sig_over_r6)

    return float(np.sum(lj * cutoff_mask))


def compute_coulomb_loop(atomic_numbers: np.ndarray, distances: np.ndarray) -> float:
    """Compute Coulomb energy using Python loops (baseline)."""
    n = len(atomic_numbers)
    total = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            r = distances[i, j]
            if r < 1e-10:
                continue

            qi = _CHARGES.get(atomic_numbers[i], 0.0)
            qj = _CHARGES.get(atomic_numbers[j], 0.0)
            if qi == 0.0 or qj == 0.0:
                continue

            r_soft = np.sqrt(r * r + 1.0)
            total += COULOMB_EV_A * qi * qj / r_soft

    return total


def compute_coulomb_vectorized(atomic_numbers: np.ndarray, distances: np.ndarray) -> float:
    """Compute Coulomb energy using numpy vectorized operations."""
    n = len(atomic_numbers)

    charges = np.array([_CHARGES.get(z, 0.0) for z in atomic_numbers])
    q_i = charges[np.newaxis, :]
    q_j = charges[:, np.newaxis]
    q_product = q_i * q_j

    mask = np.triu(np.ones((n, n)), k=1)
    charge_mask = (q_product != 0.0)
    r_soft = np.sqrt(distances * distances + 1.0)

    return float(np.sum(COULOMB_EV_A * q_product / r_soft * mask * charge_mask))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Tier 2 physics computation")
    parser.add_argument("--n-atoms", type=int, default=50, help="Number of atoms")
    parser.add_argument("--n-cycles", type=int, default=500, help="Number of simulation cycles")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    print(f"[benchmark] Generating molecule with {args.n_atoms} atoms...")
    atomic_numbers, coordinates = generate_random_molecule(args.n_atoms)

    # Compute pairwise distances
    diffs = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
    distances = np.linalg.norm(diffs, axis=-1)

    results = {
        "n_atoms": args.n_atoms,
        "n_cycles": args.n_cycles,
        "repeats": args.repeats,
    }

    # LJ benchmark
    print("[benchmark] Benchmarking LJ potential (loop vs vectorized)...")
    lj_loop_times = []
    lj_vec_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.n_cycles):
            compute_lj_loop(atomic_numbers, distances)
        lj_loop_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for _ in range(args.n_cycles):
            compute_lj_vectorized(atomic_numbers, distances)
        lj_vec_times.append(time.perf_counter() - t0)

    results["lj_potential"] = {
        "loop_mean_s": float(np.mean(lj_loop_times)),
        "vectorized_mean_s": float(np.mean(lj_vec_times)),
        "speedup": float(np.mean(lj_loop_times) / np.mean(lj_vec_times)) if np.mean(lj_vec_times) > 0 else 0,
    }

    # Coulomb benchmark
    print("[benchmark] Benchmarking Coulomb potential (loop vs vectorized)...")
    coul_loop_times = []
    coul_vec_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.n_cycles):
            compute_coulomb_loop(atomic_numbers, distances)
        coul_loop_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for _ in range(args.n_cycles):
            compute_coulomb_vectorized(atomic_numbers, distances)
        coul_vec_times.append(time.perf_counter() - t0)

    results["coulomb_potential"] = {
        "loop_mean_s": float(np.mean(coul_loop_times)),
        "vectorized_mean_s": float(np.mean(coul_vec_times)),
        "speedup": float(np.mean(coul_loop_times) / np.mean(coul_vec_times)) if np.mean(coul_vec_times) > 0 else 0,
    }

    # Print results
    print("\n" + "=" * 60)
    print("  Tier 2 Physics Benchmark Results")
    print("=" * 60)

    for category, data in results.items():
        if isinstance(data, dict) and "speedup" in data:
            print(f"\n  {category}:")
            print(f"    Loop-based:     {data['loop_mean_s']:.4f} s")
            print(f"    Vectorized:     {data['vectorized_mean_s']:.4f} s")
            print(f"    Speedup:        {data['speedup']:.2f}x")

    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
