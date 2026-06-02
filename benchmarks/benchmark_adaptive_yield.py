#!/usr/bin/env python3
"""Benchmark: adaptive mutation bias + fitness-based fragment pruning.

Compares baseline (uniform SMARTS, FIFO fragment eviction) vs. new
(adaptive mutation bias + fitness-based fragment pool pruning) to
prove that the intervention increases novel Murcko scaffold yield.

Usage:
    python -m benchmarks.benchmark_adaptive_yield

ADR-2026-06-01: Added random.seed(seed) to run_trial alongside np.random.seed
for reproducibility. The MutationEngine uses random.shuffle internally (via BRICS),
which Python's random module controls — without this seed, trials are not
deterministically reproducible.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import random
import sys
import time
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger("aurelius").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.agent.loop import DiscoveryLoop, AgentConfig
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.state import LoopState
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext


SEED_SMILES = [
    "COC(=O)OC",
    "C1COCCO1",
    "CS(=O)(=O)C",
    "CC#N",
    "C1CCOC1",
]

N_GENERATIONS = 5
BATCH_SIZE = 8
WALL_TIME_LIMIT = 90.0


def compute_murcko_scaffold(smiles: str) -> str | None:
    """Compute Murcko scaffold SMILES."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
        if scaffold:
            return scaffold
        generic = MurckoScaffold.MakeScaffoldGeneric(mol=ctx.mol)
        if generic:
            return Chem.MolToSmiles(generic)
        return smiles
    except Exception:
        return None


def run_trial(adaptive_bias: bool, seed: int = 42) -> dict:
    """Run a discovery loop trial and return results."""
    np.random.seed(seed)
    random.seed(seed)

    engine = MutationEngine(seed_smiles=SEED_SMILES, adaptive_bias=adaptive_bias)

    state = LoopState(path=f"/tmp/benchmark_state_adaptive_{adaptive_bias}.json")
    state.clear()

    pipeline = AureliusPipeline()
    pipeline.initialize()

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        state=state,
        max_generations=N_GENERATIONS,
        batch_size=BATCH_SIZE,
        max_wall_time=WALL_TIME_LIMIT,
    )

    stderr_capture = io.StringIO()
    with contextlib.redirect_stderr(stderr_capture):
        result = loop.execute()

    return result


def get_unique_scaffolds_from_results(results: dict) -> tuple[set[str], int]:
    """Extract unique Murcko scaffolds from the top 10% of results."""
    all_results = results.get("all_results", [])
    if not all_results:
        return set(), 0

    scored = [(r.total_score, r.smiles) for r in all_results]
    scored.sort(key=lambda x: -x[0])

    n_top = max(1, len(scored) // 10)
    top_results = scored[:n_top]

    scaffolds: set[str] = set()
    for score, smi in top_results:
        s = compute_murcko_scaffold(smi)
        if s:
            scaffolds.add(s)

    return scaffolds, len(top_results)


def get_seed_scaffolds() -> set[str]:
    """Get Murcko scaffolds of the seed molecules."""
    scaffolds: set[str] = set()
    for smi in SEED_SMILES:
        s = compute_murcko_scaffold(smi)
        if s:
            scaffolds.add(s)
    return scaffolds


def count_lines_added() -> int:
    """Count lines added vs the baseline (without interventions)."""
    pass


def main() -> None:
    print("=" * 65)
    print("  BENCHMARK: Adaptive Mutation Bias + Fitness-Based Fragment Pruning")
    print("=" * 65)
    print()

    seed_scaffolds = get_seed_scaffolds()
    print(f"Seed scaffolds: {len(seed_scaffolds)}")
    print(f"  {sorted(seed_scaffolds)}")
    print()

    print(f"Running {N_GENERATIONS} generations, batch_size={BATCH_SIZE}...")
    print()

    # --- Baseline trial (no adaptive bias) ---
    print("  [1/2] Running BASELINE (uniform mutation, FIFO eviction)...")
    t0 = time.time()
    baseline_results = run_trial(adaptive_bias=False)
    baseline_time = time.time() - t0
    baseline_scaffolds, baseline_top_n = get_unique_scaffolds_from_results(baseline_results)
    baseline_novel = baseline_scaffolds - seed_scaffolds
    print(f"         Done in {baseline_time:.1f}s")
    print(f"         Total screened: {baseline_results.get('total_screened', 0)}")
    print(f"         Top-{baseline_top_n} scaffolds: {len(baseline_scaffolds)} total, {len(baseline_novel)} novel")
    print()

    # --- New trial (with adaptive bias) ---
    print("  [2/2] Running INTERVENTION (adaptive bias, fitness-based pruning)...")
    t0 = time.time()
    new_results = run_trial(adaptive_bias=True)
    new_time = time.time() - t0
    new_scaffolds, new_top_n = get_unique_scaffolds_from_results(new_results)
    new_novel = new_scaffolds - seed_scaffolds
    print(f"         Done in {new_time:.1f}s")
    print(f"         Total screened: {new_results.get('total_screened', 0)}")
    print(f"         Top-{new_top_n} scaffolds: {len(new_scaffolds)} total, {len(new_novel)} novel")
    print()

    # --- Comparison ---
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<40} {'Baseline':>10} {'Intervention':>12}")
    print(f"  {'-'*40} {'-'*10} {'-'*12}")
    print(f"  {'Novel scaffolds (top 10%)':<40} {len(baseline_novel):>10} {len(new_novel):>12}")
    print(f"  {'Total scaffolds (top 10%)':<40} {len(baseline_scaffolds):>10} {len(new_scaffolds):>12}")
    print(f"  {'Novelty ratio':<40} {len(baseline_novel)/max(baseline_top_n,1):>10.1%} {len(new_novel)/max(new_top_n,1):>12.1%}")
    print(f"  {'Wall time (s)':<40} {baseline_time:>10.1f} {new_time:>12.1f}")
    print()

    # --- Assert improvement ---
    B2 = len(baseline_novel)
    N2 = len(new_novel)
    pct_improvement = ((N2 - B2) / max(B2, 1)) * 100

    print(f"  Novel scaffold yield improvement: {pct_improvement:+.1f}%")
    print()

    assert N2 > B2, (
        f"FAILED: Intervention did not increase novel scaffold yield.\n"
        f"  Baseline: {B2} novel scaffolds\n"
        f"  Intervention: {N2} novel scaffolds\n"
    )

    print("  PASSED: Intervention increases novel scaffold yield.")
    print()


if __name__ == "__main__":
    main()
