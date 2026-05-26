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
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.agent import MutationEngine
from aurelius.agent.reporting import (
    generate_chemical_insights,
    generate_discovery_results,
    generate_manifest,
    generate_screening_statistics,
    write_top_discoveries,
)
from aurelius.agent.state import CheckpointManager, ConvergenceChecker, FeedbackAdapter
from aurelius.config import AureliusConfig, initialize_environment
from aurelius.memory.profiler import MemoryProfiler
from aurelius.pipeline import AureliusPipeline
from aurelius.screening.tier0.predictor import Tier0ActivationPredictor
from aurelius.utils.chem_utils import (
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
    _serialize_fp,
)

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

    convergence = ConvergenceChecker()
    if resumed and state.get("batch", 0) > 0:
        convergence.total_screened = screened_so_far
        convergence.viable_count = state.get("viable_count", 0)
        convergence.generations = start_batch

    feedback = FeedbackAdapter()
    all_results: list[dict[str, Any]] = []
    discoveries: list[dict[str, Any]] = []

    # ---- Main Loop ----
    max_generations = args.max_generations or 50
    batch_size = args.batch_size or 50
    max_wall_time = 43200  # 12 hours

    wall_start = time.time()
    current_batch = start_batch
    generation = 0

    print(f"\n[AGENT] Starting screening loop. Batch size: {batch_size}, Max generations: {max_generations}")
    print(f"[AGENT] Time limit: {max_wall_time}s (12 hours)\n")

    screened_smiles: set[str] = set()

    while generation < max_generations:
        elapsed = time.time() - wall_start
        if elapsed > max_wall_time:
            print(f"\n[AGENT] Time cap reached ({elapsed:.0f}s). Exiting gracefully.")
            break

        generation += 1
        current_batch += 1

        # ---- Generation: Mutate seeds ----
        if generation == 1:
            candidates = engine.mutate_batch(seed_smiles, batch_size * 3)
        else:
            if all_results:
                scored_results = [
                    (r["score"].total_score, r["score"].molecule_smiles) for r in all_results if r.get("score")
                ]
                scored_results.sort(key=lambda x: -x[0])
                top_seeds = [s for _, s in scored_results[: max(5, len(scored_results) // 5)]]
            else:
                top_seeds = seed_smiles[:5]
            candidates = engine.mutate_batch(top_seeds, batch_size * 3)

        # Filter invalid & duplicate
        valid_candidates: list[str] = []
        invalid_count = 0
        for smi in candidates:
            if smi in screened_smiles:
                invalid_count += 1
                continue
            mol = _safe_mol_from_smiles(smi)
            if mol is None:
                invalid_count += 1
                continue
            if not _is_valid_mol(mol):
                invalid_count += 1
                continue
            valid_candidates.append(smi)

        if len(valid_candidates) > batch_size:
            valid_candidates = valid_candidates[:batch_size]

        if not valid_candidates:
            print(f"[AGENT] Generation {generation}: No valid candidates. Skipping.")
            continue

        print(
            f"[AGENT] Generation {generation}: Screening {len(valid_candidates)} candidates "
            f"(invalid discarded: {invalid_count})"
        )

        # ---- Screening: Batch processing ----
        batch_scores: list[float] = []
        batch_viable = 0
        batch_discoveries: list[dict[str, Any]] = []
        batch_fps_hex: list[str] = []

        batch_file = f"candidates_batch_{current_batch}.smi"
        with open(batch_file, "w") as f:
            for smi in valid_candidates:
                f.write(f"{smi}\n")

        for smi in valid_candidates:
            try:
                result = pipeline.screen_molecule(smi)
            except Exception as e:
                log.error("Pipeline error for %s: %s", smi, e, exc_info=True)
                continue

            score = result.get("score")
            if score is None:
                continue

            screened_smiles.add(smi)
            engine.add_to_db(smi)

            total_score = score.total_score
            batch_scores.append(total_score)

            is_discovery = (
                total_score >= 65.0
                and score.tier1_viable
                and score.tier2_viable
                and score.tier3_viable
                and len(score.rejection_reasons) == 0
            )

            if is_discovery:
                batch_viable += 1
                discovery_entry = {
                    "smiles": smi,
                    "total_score": total_score,
                    "sigma": score.sigma_score,
                    "desolvation": score.desolvation_score,
                    "sei_homogeneity": score.sei_homogeneity_score,
                    "mx_synthesis": score.mx_synthesis_score,
                    "gwp_penalty": score.gwp_penalty,
                    "is_viable": True,
                    "rejection_reasons": score.rejection_reasons,
                    "components": score.rejection_reasons,
                }
                batch_discoveries.append(discovery_entry)
                discoveries.append(discovery_entry)
                checkpoint.add_discovery(discovery_entry)
                print(f"  ** DISCOVERY ** {smi} (score={total_score:.1f})")

            all_results.append(result)
            feedback.record(result)

        new_fps_count = 0
        for smi in valid_candidates:
            mol = _safe_mol_from_smiles(smi)
            if mol is not None:
                fp_hex = _serialize_fp(_mol_to_fp(mol))
                batch_fps_hex.append(fp_hex)
                checkpoint.add_fps_hex(fp_hex)
                new_fps_count += 1

        convergence.record_batch(batch_scores, batch_viable, new_fps_count)
        checkpoint.update_stats(valid_candidates, batch_scores, batch_viable, invalid_count)

        print(
            f"  Generation {generation} complete: "
            f"{len(valid_candidates)} screened, {batch_viable} viable, "
            f"best={max(batch_scores) if batch_scores else 0:.1f}"
        )

        if profiler:
            profiler.sample(
                generation=generation,
                screened_count=convergence.total_screened,
                gc_collected=0,
            )

        strategy = feedback.get_adaptation_strategy()
        if generation % 5 == 0:
            print(f"  [Feedback] Strategy: {strategy['recommendation']}")

        should_stop, reason = convergence.should_terminate()
        if should_stop:
            print(f"\n[AGENT] Convergence reached: {reason}")
            break

        # ---- Atomic checkpoint save after every molecule ----
        checkpoint.save()

        print(
            f"  [Progress] Screened: {convergence.total_screened}, "
            f"Viable: {convergence.viable_count}, "
            f"Generations: {generation}/{max_generations}\n"
        )

    # ---- Post-loop: Generate all deliverables ----
    print("\n" + "=" * 60)
    print("  GENERATING DELIVERABLES")
    print("=" * 60)

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
    print(f"  Total screened:     {convergence.total_screened}")
    print(f"  Generations run:    {convergence.generations}")
    print(f"  Viable discoveries: {convergence.viable_count}")
    print(f"  Best score:         {checkpoint.state['best_score']:.1f}")
    print(f"  Invalid discarded:  {checkpoint.state['invalid_discarded']}")
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

    checkpoint = CheckpointManager()
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
