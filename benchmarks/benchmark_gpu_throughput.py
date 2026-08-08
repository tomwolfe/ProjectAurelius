"""GPU throughput benchmark for Project Aurelius v11.0 Objective 1.

Measures batch GC prediction and Tanimoto similarity throughput
on Apple Silicon (MPS/MLX) vs CPU baseline.

Usage:
    python -m benchmarks.benchmark_gpu_throughput
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from aurelius.scoring.oracle.gc import (
    _count_fragments_batch,
    predict_dielectric_proxy_batch,
    predict_ionic_conductivity_proxy_batch,
    predict_li_solvation_proxy_batch,
    predict_viscosity_proxy_batch,
)
from aurelius.scoring.oracle.oracle import (
    PropertyOracle,
    _fp_batch_to_numpy,
    _select_batch_backend,
    batch_tanimoto_similarity,
)
from aurelius.scoring.oracle.quantum import (
    predict_tom_orbitals_batch,
)
from aurelius.types import MoleculeContext

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_MOLECULES = 1000
N_REPEATS = 5


def _generate_test_molecules(n: int) -> list[MoleculeContext]:
    templates = [
        "CCO", "CC(=O)O", "CCN", "c1ccccc1", "CCOC", "CC(C)O",
        "CC(=O)N", "CN(C)C", "CCS", "CCCl", "CCBr", "CCF", "CCI",
        "CC#N", "CC=O", "CC(=O)C", "CC(=O)CC", "CCOCC", "CCOC(=O)C",
        "CC(=O)OC", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1",
        "c1ccc(Cl)cc1", "c1ccc(Br)cc1", "c1ccc(I)cc1", "c1ccc(OC)cc1",
        "c1ccc(CC)cc1", "c1ccc(CN)cc1", "c1ccc(S)cc1", "c1ccc(SC)cc1",
        "c1ccc(NC)cc1", "c1ccc(OC(=O)C)cc1", "c1ccc(CC(=O)O)cc1",
        "c1ccc(CCN)cc1", "c1ccc(CCO)cc1", "c1ccc(CCF)cc1",
        "c1ccc(CCCl)cc1", "c1ccc(CCBr)cc1", "c1ccc(CCI)cc1",
        "c1ccc(CC#N)cc1", "c1ccc(CC=O)cc1", "c1ccc(CC(=O)C)cc1",
        "c1ccc(CC(=O)CC)cc1", "c1ccc(CCOCC)cc1",
        "c1ccc(CC(=O)N)cc1", "c1ccc(CCS(=O)(=O)C)cc1",
        "c1ccc(CCN(C)C)cc1", "c1ccc(CCOC)cc1",
        "c1ccc(CC(=O)OC)cc1", "c1ccc(CCBr)cc1", "c1ccc(CCF)cc1",
    ]
    smiles = (templates * (n // len(templates) + 1))[:n]
    contexts: list[MoleculeContext] = []
    for s in smiles:
        ctx = MoleculeContext.from_smiles(s)
        if ctx is not None:
            contexts.append(ctx)
    return contexts


def benchmark_gc_batch(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark vectorized GC batch prediction."""
    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        counts = _count_fragments_batch(contexts)
        tpsa = np.array([c.tpsa for c in contexts], dtype=np.float32)
        mw = np.array([c.mw for c in contexts], dtype=np.float32)
        n_rot = np.array([c.rotatable_bonds for c in contexts], dtype=np.int32)
        from aurelius.scoring.oracle.gc import _count_branch_points, _count_stereocenters
        n_branch = np.array([_count_branch_points(c.mol) for c in contexts], dtype=np.int32)
        n_stereo = np.array([_count_stereocenters(c.mol) for c in contexts], dtype=np.int32)

        d = predict_dielectric_proxy_batch(counts, tpsa, contexts)
        v = predict_viscosity_proxy_batch(counts, mw, n_rot, n_branch, n_stereo, contexts)
        ls = predict_li_solvation_proxy_batch(counts, mw)
        predict_ionic_conductivity_proxy_batch(d, v, ls)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    return {
        "n_molecules": len(contexts),
        "median_time_ms": median_time * 1000,
        "throughput_mols_per_sec": len(contexts) / median_time,
    }


def benchmark_tanimoto_mps(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark batch Tanimoto similarity with MPS/MLX backend."""
    fps = [ctx.get_ecfp4() for ctx in contexts]
    backend = _select_batch_backend()

    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        sim_matrix = batch_tanimoto_similarity(fps)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    n_pairs = len(contexts) * (len(contexts) - 1) // 2
    return {
        "n_molecules": len(contexts),
        "backend": backend,
        "median_time_ms": median_time * 1000,
        "throughput_pairs_per_sec": n_pairs / median_time,
        "matrix_shape": list(sim_matrix.shape),
    }


def benchmark_fp_conversion(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark fingerprint batch conversion to numpy."""
    fps = [ctx.get_ecfp4() for ctx in contexts]

    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        arr = _fp_batch_to_numpy(fps)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    return {
        "n_molecules": len(contexts),
        "median_time_ms": median_time * 1000,
        "throughput_mols_per_sec": len(contexts) / median_time,
        "matrix_shape": list(arr.shape),
    }


def benchmark_oracle_batch(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark PropertyOracle.predict_batch_properties end-to-end."""
    oracle = PropertyOracle(use_xtb=False)

    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        result = oracle.predict_batch_properties(contexts)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    return {
        "n_molecules": len(contexts),
        "backend": _select_batch_backend(),
        "median_time_ms": median_time * 1000,
        "throughput_mols_per_sec": len(contexts) / median_time,
        "keys": list(result.keys()),
    }


def benchmark_tom_batch(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark batch TOM orbital prediction (quantum oracle)."""
    mols = [ctx.mol for ctx in contexts]

    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        homo, lumo = predict_tom_orbitals_batch(mols)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    return {
        "n_molecules": len(contexts),
        "backend": _select_batch_backend(),
        "median_time_ms": median_time * 1000,
        "throughput_mols_per_sec": len(contexts) / median_time,
        "homo_mean": round(float(np.mean(homo)), 4) if len(homo) else 0.0,
        "lumo_mean": round(float(np.mean(lumo)), 4) if len(lumo) else 0.0,
    }


def main() -> None:
    print("=" * 60)
    print("Project Aurelius v11.0 — GPU Throughput Benchmark")
    print("=" * 60)

    contexts = _generate_test_molecules(N_MOLECULES)
    print(f"Test molecules: {len(contexts)}")
    print(f"MPS available: {__import__('torch').backends.mps.is_available()}")
    print(f"Selected backend: {_select_batch_backend()}")
    print()

    results: dict[str, dict[str, float]] = {}

    print("--- GC Batch Prediction ---")
    gc_result = benchmark_gc_batch(contexts)
    results["gc_batch"] = gc_result
    print(f"  Throughput: {gc_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print()

    print("--- Fingerprint Batch Conversion ---")
    fp_result = benchmark_fp_conversion(contexts)
    results["fp_conversion"] = fp_result
    print(f"  Throughput: {fp_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print()

    print("--- Batch Tanimoto Similarity ---")
    tanimoto_result = benchmark_tanimoto_mps(contexts)
    results["tanimoto_mps"] = tanimoto_result
    print(f"  Backend: {tanimoto_result['backend']}")
    print(f"  Throughput: {tanimoto_result['throughput_pairs_per_sec']:.0f} pairs/sec")
    print()

    print("--- Oracle Batch Properties (end-to-end) ---")
    oracle_result = benchmark_oracle_batch(contexts)
    results["oracle_batch"] = oracle_result
    print(f"  Backend: {oracle_result['backend']}")
    print(f"  Throughput: {oracle_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print()

    print("--- TOM Batch Orbital Prediction ---")
    tom_result = benchmark_tom_batch(contexts)
    results["tom_batch"] = tom_result
    print(f"  Backend: {tom_result['backend']}")
    print(f"  Throughput: {tom_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print(f"  HOMO mean: {tom_result['homo_mean']:.4f}, LUMO mean: {tom_result['lumo_mean']:.4f}")
    print()

    # Save results
    import json
    results_path = RESULTS_DIR / "gpu_throughput.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
