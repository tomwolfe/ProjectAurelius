#!/usr/bin/env python3
"""Benchmark: Reality Check — surrogate discoveries vs. known commercial electrolytes.

Compares the top 50 EA-discovered molecules from a 5-generation run against
the known_electrolytes.json set. Asserts:
  1. Mean Aurelius Score of top discoveries > mean score of known set
  2. >80% of top discoveries have novel Murcko scaffolds vs. known set

Usage:
    python -m benchmarks.benchmark_reality_check

ADR-2026-06-02: Initial benchmark. Physical justification: The EA optimises a
surrogate world (the Oracle). Without a reality check, the EA may discover
molecules that score well in-silico but are physically unrealistic or already
commercially available. This benchmark ensures the EA's top discoveries are
both high-scoring AND structurally novel compared to known commercial electrolytes.
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
from pathlib import Path

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
WALL_TIME_LIMIT = 120.0
TOP_N = 50


def _load_known_electrolytes() -> list[str]:
    data_path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json"
    with open(data_path) as f:
        return json.load(f)


def _compute_murcko_scaffold(smiles: str) -> str | None:
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


def _screen_smiles_set(pipeline: AureliusPipeline, smiles_list: list[str]) -> dict[str, float]:
    """Screen a list of SMILES and return {smiles: total_score} dict."""
    scores: dict[str, float] = {}
    for smi in smiles_list:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        try:
            result = pipeline.screen_molecule(ctx)
            score = result.get("score", {}).get("total_score", 0.0)
            scores[smi] = score
        except Exception:
            continue
    return scores


def run_discovery_trial(seed: int = 42) -> dict:
    """Run a discovery loop trial and return results."""
    np.random.seed(seed)
    random.seed(seed)

    engine = MutationEngine(seed_smiles=SEED_SMILES)

    state = LoopState(path="/tmp/benchmark_reality_check_state.json")
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


def main() -> None:
    print("=" * 65)
    print("  REALITY CHECK BENCHMARK: EA Discoveries vs. Known Electrolytes")
    print("=" * 65)
    print()

    # --- Step 1: Screen known electrolytes ---
    print("  [1/4] Screening known commercial electrolytes...")
    known_smiles = _load_known_electrolytes()
    known_scaffolds: set[str] = set()
    for smi in known_smiles:
        s = _compute_murcko_scaffold(smi)
        if s:
            known_scaffolds.add(s)
    print(f"         Known electrolytes: {len(known_smiles)} molecules, {len(known_scaffolds)} unique scaffolds")
    print()

    pipeline = AureliusPipeline()
    pipeline.initialize()

    known_scores = _screen_smiles_set(pipeline, known_smiles)
    known_mean = np.mean(list(known_scores.values())) if known_scores else 0.0
    print(f"         Known set mean Aurelius Score: {known_mean:.2f}")
    print()

    # --- Step 2: Run discovery loop ---
    print(f"  [2/4] Running {N_GENERATIONS}-generation discovery loop...")
    t0 = time.time()
    results = run_discovery_trial()
    trial_time = time.time() - t0
    print(f"         Done in {trial_time:.1f}s")
    print(f"         Total screened: {results.get('total_screened', 0)}")
    print()

    # --- Step 3: Analyze top discoveries ---
    print("  [3/4] Analyzing top discoveries...")
    all_results = results.get("all_results", [])
    scored = [(r.total_score, r.smiles) for r in all_results]
    scored.sort(key=lambda x: -x[0])
    top_results = scored[:TOP_N]

    # Get unique scaffolds of top discoveries
    top_scaffolds: set[str] = set()
    for score, smi in top_results:
        s = _compute_murcko_scaffold(smi)
        if s:
            top_scaffolds.add(s)

    novel_scaffolds = top_scaffolds - known_scaffolds
    novelty_ratio = len(novel_scaffolds) / max(len(top_scaffolds), 1)

    top_scores = [s for s, _ in top_results]
    top_mean = np.mean(top_scores) if top_scores else 0.0

    print(f"         Top {TOP_N} discoveries: mean score={top_mean:.2f}")
    print(f"         Unique scaffolds in top {TOP_N}: {len(top_scaffolds)}")
    print(f"         Novel scaffolds (not in known set): {len(novel_scaffolds)} ({novelty_ratio:.1%})")
    print()

    # --- Step 4: Assertions ---
    print("  [4/4] Verifying assertions...")
    print()

    # Assertion 1: Mean score of top discoveries > mean score of known set
    score_improvement = top_mean - known_mean
    print(f"         Score gap: top discoveries ({top_mean:.2f}) - known ({known_mean:.2f}) = {score_improvement:+.2f}")
    assert top_mean > known_mean, (
        f"FAILED: Top discovery mean score ({top_mean:.2f}) does not exceed "
        f"known electrolyte mean ({known_mean:.2f})."
    )
    print("         PASSED: Discoveries score higher than known commercial set.")
    print()

    # Assertion 2: >80% novel Murcko scaffolds
    print(f"         Novel scaffold ratio: {novelty_ratio:.1%} (target >80%)")
    assert novelty_ratio > 0.80, (
        f"FAILED: Only {novelty_ratio:.1%} of top discovery scaffolds are novel "
        f"({len(novel_scaffolds)}/{len(top_scaffolds)}). Target >80%."
    )
    print("         PASSED: >80% of top discoveries have novel scaffolds.")
    print()

    print("=" * 65)
    print("  REALITY CHECK: ALL ASSERTIONS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
