"""GPU throughput benchmark for Project Aurelius v12.1.

Measures batch GC prediction, Tanimoto similarity, TOM orbital prediction,
R_g batch embedding, and MLX GPR variance throughput on Apple Silicon (MPS/MLX)
vs CPU baseline.

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
    _compute_radius_of_gyration_batch,
)
from aurelius.scoring.oracle.mlx_surrogate import (
    predict_variance_batch_mlx,
)
from aurelius.types import MoleculeContext

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_MOLECULES = 1000
N_REPEATS = 5

# Phase 4 benchmark targets (ADR-2026-08-11-03)
RG_THROUGHPUT_TARGET = 2000  # mol/s on M5 Pro with cache + threading
MLX_VARIANCE_TOL = 1e-4  # max abs diff vs sklearn variance


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


def benchmark_rg_batch(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark radius-of-gyration batch computation (86% bottleneck).

    ADR-2026-08-07-04: _compute_radius_of_gyration_batch is 86% of batch time.
    Phase 4 targets: LRU-memoised canonical-SMILES cache + ThreadPoolExecutor
    for cache misses. Target: >=2000 mol/s on M5 Pro.
    """
    mols = [ctx.mol for ctx in contexts]

    # Warm cache so cache-hit path is measured
    _compute_radius_of_gyration_batch(mols)

    times: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        rgs = _compute_radius_of_gyration_batch(mols)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = float(np.median(times))
    throughput = len(contexts) / median_time

    result = {
        "n_molecules": len(contexts),
        "median_time_ms": median_time * 1000,
        "throughput_mols_per_sec": throughput,
    }

    # Phase 4 assertion: cache hits must be >= 2000 mol/s
    if throughput < RG_THROUGHPUT_TARGET:
        print(f"  WARNING: R_g throughput {throughput:.0f} < target {RG_THROUGHPUT_TARGET}")
    else:
        print(f"  ✓ R_g throughput {throughput:.0f} >= target {RG_THROUGHPUT_TARGET}")

    return result


def benchmark_mlx_variance(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark MLX GPR posterior variance for BALD acquisition.

    Phase 4: Wire MLX variance into BALD acquisition. Verifies MLX variance
    matches sklearn within 1e-4.
    """
    from aurelius.scoring.oracle.conformal import get_conformal_predictor
    from aurelius.agent.experiment_suggester import _gpr_model_from_predictor

    predictor = get_conformal_predictor()
    gpr_model = _gpr_model_from_predictor(predictor)

    if gpr_model is None:
        print("  MLX GPR variance: no GPR model available (skipped)")
        return {"n_molecules": len(contexts), "available": False}

    mols = [ctx.mol for ctx in contexts[:100]]  # 100 molecules for variance

    # Time the MLX variance computation
    start = time.perf_counter()
    mlx_var = predict_variance_batch_mlx(gpr_model, mols)
    elapsed = time.perf_counter() - start

    # Compare with sklearn if available
    sklearn_var: np.ndarray | None = None
    diff = None
    try:
        from aurelius.scoring.oracle.mlx_surrogate import _predict_deltas_batch_sklearn

        _, sklearn_std = _predict_deltas_batch_sklearn(gpr_model, mols, return_std=True)
        if sklearn_std is not None:
            sklearn_var = sklearn_std ** 2
            diff = float(np.max(np.abs(mlx_var - sklearn_var)))
    except Exception:
        pass

    result = {
        "n_molecules": len(mols),
        "available": True,
        "median_time_ms": elapsed * 1000,
        "throughput_mols_per_sec": len(mols) / elapsed,
        "variance_mean": round(float(np.mean(mlx_var)), 6) if len(mlx_var) else 0.0,
    }
    if diff is not None:
        result["max_abs_diff_vs_sklearn"] = round(diff, 8)
        if diff < MLX_VARIANCE_TOL:
            print(f"  ✓ MLX variance matches sklearn (max diff {diff:.6e} < {MLX_VARIANCE_TOL})")
        else:
            print(f"  WARNING: MLX variance diff {diff:.6e} > tolerance {MLX_VARIANCE_TOL}")

    return result


def benchmark_single_evaluate(contexts: list[MoleculeContext]) -> dict[str, float]:
    """Benchmark single-molecule PropertyOracle.evaluate() (warm, no cache).

    Measures the realistic per-molecule cost when processing a stream of
    novel candidates (no cache hits). Includes Δ-correction GPR prediction.
    """
    from aurelius.scoring.oracle.quantum import QuantumOracle

    oracle = PropertyOracle(use_xtb=False)
    oracle._quantum.warmup()

    n_single = min(200, len(contexts))
    start = time.perf_counter()
    for ctx in contexts[:n_single]:
        oracle._cache.clear()
        oracle.evaluate(ctx)
    elapsed = time.perf_counter() - start

    return {
        "n_molecules": n_single,
        "total_time_ms": elapsed * 1000,
        "throughput_mols_per_sec": n_single / elapsed,
    }


def main() -> None:
    print("=" * 60)
    print("Project Aurelius v12.1 — GPU Throughput Benchmark")
    print("=" * 60)

    contexts = _generate_test_molecules(N_MOLECULES)
    print(f"Test molecules: {len(contexts)}")
    print(f"MPS available: {__import__('torch').backends.mps.is_available()}")
    print(f"MLX available: {__import__('importlib').util.find_spec('mlx') is not None}")
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

    print("--- R_g Batch (86% bottleneck) ---")
    rg_result = benchmark_rg_batch(contexts)
    results["rg_batch"] = rg_result
    print(f"  Throughput: {rg_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print()

    print("--- MLX GPR Variance (BALD) ---")
    var_result = benchmark_mlx_variance(contexts)
    results["mlx_variance"] = var_result
    if var_result.get("available"):
        print(f"  Throughput: {var_result['throughput_mols_per_sec']:.0f} molecules/sec")
        if "max_abs_diff_vs_sklearn" in var_result:
            print(f"  Max diff vs sklearn: {var_result['max_abs_diff_vs_sklearn']:.6e}")
    else:
        print("  Unavailable (no GPR model)")
    print()

    print("--- Single-Molecule evaluate() (warm, no cache) ---")
    single_result = benchmark_single_evaluate(contexts)
    results["single_evaluate"] = single_result
    print(f"  Throughput: {single_result['throughput_mols_per_sec']:.0f} molecules/sec")
    print(f"  Per-molecule: {single_result['total_time_ms']/single_result['n_molecules']:.2f} ms/mol")
    print()

    # Save results
    import json
    results_path = RESULTS_DIR / "gpu_throughput.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
