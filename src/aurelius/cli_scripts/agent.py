#!/usr/bin/env python3
"""Autonomous Screening Agent — Project Aurelius v9.0

Implements the full autonomous discovery loop:
  Generation (RDKit mutation engine) -> Screening (filter + oracle) ->
  Feedback-driven mutation -> Convergence check -> Report generation

Usage:
    aurelius agent run --max-generations 100 --batch-size 100
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aurelius.agent import MutationEngine
from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.reporting import (
    generate_discoveries_sdf,
    generate_run_summary,
)
from aurelius.agent.state import CheckpointManager
from aurelius.pipeline import AureliusPipeline
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


@dataclass(frozen=True)
class AgentConfig:
    """Parameters for the autonomous screening agent."""

    max_generations: int = 50
    batch_size: int = 50


def _check_apple_silicon() -> bool:
    """Detect if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon (arm64 Darwin).
    """
    import platform

    return platform.machine() in ("arm64",) and platform.system() == "Darwin"


def _load_smiles_file(path: str) -> list[str]:
    """Load SMILES from a .smi file, skipping comments and blank lines."""
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


def run_screening(agent_cfg: AgentConfig, checkpoint: CheckpointManager) -> None:
    """Main autonomous screening loop.

    Args:
        agent_cfg: Agent configuration with screening parameters.
        checkpoint: CheckpointManager instance for saving progress.
    """

    # ---- Phase 5: Checkpoint & Resume ----
    state = checkpoint.load()

    known_fps_hex = state.get("known_fps_hex", [])
    engine = MutationEngine()
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
    pipeline = AureliusPipeline()
    pipeline.initialize()
    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        checkpoint=checkpoint,
        max_generations=agent_cfg.max_generations,
        batch_size=agent_cfg.batch_size,
    )
    results = loop.execute()

    # ---- Post-loop: Generate all deliverables ----
    print("\n" + "=" * 60)
    print("  GENERATING DELIVERABLES")
    print("=" * 60)

    all_results = results["all_results"]
    discoveries = results["discoveries"]
    convergence = loop.convergence  # type: ignore[union-attr]

    generate_run_summary(convergence, all_results, discoveries)
    generate_discoveries_sdf(discoveries)
    checkpoint.save()

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
    print("    - discoveries.sdf")
    print("    - run_summary.json")
    print("    - agent_state.json")
    print("    - errors.log")
    print()


def _save_checkpoint_safe(checkpoint: CheckpointManager) -> None:
    """Safely save checkpoint, suppressing any exceptions."""
    with contextlib.suppress(Exception):
        checkpoint.save()


def main() -> None:
    """CLI entry point for the autonomous screening agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius v9.0 Autonomous Screening Agent")
    parser.add_argument("--max-generations", type=int, default=50, help="Maximum generations to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Candidates per batch")
    args = parser.parse_args()

    output_dir = os.environ.get("AURELIUS_OUTPUT_DIR")
    checkpoint = CheckpointManager(output_dir=output_dir)
    try:
        agent_cfg = AgentConfig(
            max_generations=args.max_generations,
            batch_size=args.batch_size,
        )
        run_screening(agent_cfg, checkpoint)
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
