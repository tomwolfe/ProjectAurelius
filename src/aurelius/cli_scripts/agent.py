#!/usr/bin/env python3
"""Autonomous Screening Agent — Project Aurelius v7.0

Implements the full autonomous discovery loop:
  Generation (RDKit mutation engine) -> Screening (3-tier pipeline) ->
  Feedback-driven mutation -> Convergence check -> Report generation

Usage:
    aurelius agent run --max-generations 100 --batch-size 100
    aurelius agent run --profile-memory
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.agent import MutationEngine
from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.reporting import (
    generate_chemical_insights,
    generate_discovery_results,
    generate_manifest,
    generate_screening_statistics,
    write_top_discoveries,
)
from aurelius.agent.state import CheckpointManager
from aurelius.config import AureliusConfig, initialize_environment
from aurelius.memory.profiler import MemoryProfiler
from aurelius.pipeline import AureliusPipeline
from aurelius.screening.tier0.predictor import Tier0ActivationPredictor
from aurelius.utils.chem_utils import _deserialize_fp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aurelius_agent")

_error_handler = logging.FileHandler("errors.log", mode="w")
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(logging.Formatter("%(asctime)s [ERROR] %(message)s"))
log.addHandler(_error_handler)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

np.random.seed(42)


def _check_apple_silicon() -> bool:
    """Detect if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon (arm64 Darwin).
    """
    import platform

    return platform.machine() in ("arm64",) and platform.system() == "Darwin"


def _load_smiles_file(path: str) -> list[str]:
    """Load SMILES from a .smi file, skipping comments and blank lines.

    Args:
        path: Path to the SMILES file.

    Returns:
        List of SMILES strings.
    """
    smiles_list: list[str] = []
    p = Path(path)
    if not p.exists():
        return smiles_list
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " #" in line:
                line = line.split(" #")[0].strip()
            if line:
                smiles_list.append(line)
    return smiles_list


def run_screening(args: Any, checkpoint: CheckpointManager) -> None:
    """Main autonomous screening loop.

    Args:
        args: Parsed argparse arguments containing screening parameters.
        checkpoint: CheckpointManager instance for saving progress.
    """

    # Initialize memory profiler if requested
    profiler: MemoryProfiler | None = None
    if getattr(args, "profile_memory", False):
        profiler = MemoryProfiler()
        profiler.start()
        print("[AGENT] Memory profiling enabled. CSV reports will be generated.")

    # ---- Phase 1: Environment & Pipeline Initialization ----
    print("=" * 60)
    print("  PROJECT AURELIUS v7.0 — Autonomous Screening Agent")
    print("  The 2nm Fusion Edition | M5 Pro Neural Accelerators")
    print("=" * 60)

    if _check_apple_silicon():
        print("[AGENT] Running on Apple Silicon (optimized).\n")
    else:
        print("[AGENT] [CPU_FALLBACK] Not running on Apple Silicon. Performance may be reduced.\n")

    initialize_environment()

    config = AureliusConfig()
    pipeline = AureliusPipeline(config)
    pipeline.initialize()

    # Inject Tier 0 Activation Energy Predictor (MPNN + linear fallback)
    if pipeline._gcmtwin:
        tier0_pred = Tier0ActivationPredictor(
            model_path="models/tier0/mpnn_weights.pth",
        )
        pipeline._gcmtwin._tier0_predictor = tier0_pred
        pipeline._gcmtwin._use_tier0_prediction = True
        print("[AGENT] Tier 0 Activation Energy Predictor (MPNN) injected successfully.")
    else:
        raise RuntimeError("Pipeline GCMD Digital Twin not initialized. Cannot run Tier 3.")

    # ---- Phase 2: Chemical Generation Engine ----
    print("\n[AGENT] Loading seed molecules...")

    seed_smiles = _load_smiles_file("discovery_candidates.smi")
    seed_smiles.extend(_load_smiles_file("examples/molecules.smi"))
    seed_smiles.extend(_load_smiles_file("homogeneity_targeted_candidates.smi"))
    seed_smiles.extend(_load_smiles_file("phase6_refined_candidates.smi"))
    seed_smiles.extend(_load_smiles_file("refined_candidates.smi"))
    seed_smiles = list(set(s for s in seed_smiles if s.strip()))
    print(f"[AGENT] Seed pool: {len(seed_smiles)} unique molecules")

    engine = MutationEngine(seed_smiles)

    # ---- Phase 5: Checkpoint & Resume ----
    state = checkpoint.load()

    known_fps_hex = state.get("known_fps_hex", [])
    engine.known_fps = []
    for h in known_fps_hex:
        with contextlib.suppress(Exception):
            engine.known_fps.append(_deserialize_fp(h))

    resumed = state["screened_count"] > 0
    start_batch = state.get("batch", 0)
    screened_so_far = state.get("screened_count", 0)
    best_score_so_far = state.get("best_score", 0.0)

    if resumed:
        print(
            f"[AGENT] Resuming from checkpoint: batch={start_batch}, "
            f"screened={screened_so_far}, best_score={best_score_so_far:.1f}"
        )
    else:
        print("[AGENT] Fresh start. No checkpoint found.")

    wall_start = time.time()

    # Build the discovery loop and run it
    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        checkpoint=checkpoint,
        max_generations=args.max_generations or 50,
        batch_size=args.batch_size or 50,
    )
    results = loop.execute()

    # ---- Post-loop: Generate all deliverables ----
    print("\n" + "=" * 60)
    print("  GENERATING DELIVERABLES")
    print("=" * 60)

    all_results = results["all_results"]
    discoveries = results["discoveries"]
    convergence = loop.convergence  # type: ignore[union-attr]

    generate_discovery_results(all_results)
    write_top_discoveries(discoveries)
    generate_screening_statistics(convergence, all_results)
    generate_chemical_insights(all_results, discoveries)
    generate_manifest(convergence, discoveries, all_results)
    checkpoint.save()

    if profiler:
        profiler.stop()
        report_path = profiler.generate_report()
        print(f"\n[AGENT] Memory profile report: {report_path}")
        print(f"  Peak RAM:      {profiler.peak_ram_gb:.2f} GB")
        print(f"  Peak MPS:      {profiler.peak_mps_gb:.2f} GB")
        print(f"  Peak MLX:      {profiler.peak_mlx_gb:-.2f} GB")
        print(f"  Samples:       {profiler.n_samples}")

    print("\n" + "=" * 60)
    print("  SCREENING COMPLETE")
    print("=" * 60)
    print(f"  Total screened:     {results['total_screened']}")
    print(f"  Generations run:    {convergence.generations}")
    print(f"  Viable discoveries: {results['total_viable']}")
    print(f"  Best score:         {checkpoint.state['best_score']:.1f}")
    print(f"  Invalid discarded:  {results['total_invalid']}")
    print(f"  Wall time:          {time.time() - wall_start:.0f}s")
    print("\n  Output files:")
    print("    - discovery_results_final.json")
    print("    - top_discoveries.smi")
    print("    - screening_statistics.md")
    print("    - chemical_insights.md")
    print("    - agent_discovery_manifest.json")
    print("    - agent_state.json")
    print("    - mutation_rationale.md")
    print("    - errors.log")
    print()


def _save_checkpoint_safe(checkpoint: CheckpointManager) -> None:
    """Safely save checkpoint, suppressing any exceptions.

    Args:
        checkpoint: CheckpointManager instance to save.
    """
    with contextlib.suppress(Exception):
        checkpoint.save()


def main() -> None:
    """CLI entry point for the autonomous screening agent."""
    parser = argparse.ArgumentParser(description="Aurelius v7.0 Autonomous Screening Agent")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-generations", type=int, default=50, help="Maximum generations to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Candidates per batch")
    parser.add_argument("--profile-memory", action="store_true", help="Enable memory profiling with CSV report output")
    args = parser.parse_args()

    output_dir = os.environ.get("AURELIUS_OUTPUT_DIR")
    checkpoint = CheckpointManager(output_dir=output_dir)
    try:
        run_screening(args, checkpoint)
    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted by user. Saving state and exiting.")
        if checkpoint is not None:
            _save_checkpoint_safe(checkpoint)
        sys.exit(1)
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        print(f"\n[FATAL] {e}")
        if checkpoint is not None:
            _save_checkpoint_safe(checkpoint)
        sys.exit(1)


if __name__ == "__main__":
    main()
