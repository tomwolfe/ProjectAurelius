"""Report generation for autonomous screening agent.

Consolidated output: exactly two files:
  - discoveries.sdf — top discovery molecules with properties
  - run_summary.json — full structured run report
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.agent.loop import ScreeningResult
from aurelius.agent.state import ConvergenceChecker


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


def generate_run_summary(
    convergence: ConvergenceChecker,
    all_results: list[ScreeningResult],
    discoveries: list[ScreeningResult],
    path: str = "run_summary.json",
    output_dir: str | Path | None = None,
) -> None:
    """Write a single consolidated run_summary.json with all screening results.

    Replaces the previous multi-file output (manifest, statistics markdown,
    chemical insights markdown, CSV, SMI, etc.) with one structured JSON.

    Args:
        convergence: ConvergenceChecker instance.
        all_results: All screening results.
        discoveries: Discovery list (score >= 65).
        path: Output JSON path.
        output_dir: Directory to write to.
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)
    scores = [r.total_score for r in all_results]

    plateau = convergence.check_score_plateau()
    saturation = convergence.check_structural_saturation()

    reasons = []
    if plateau:
        reasons.append("score plateau confirmed")
    if saturation:
        reasons.append("structural saturation reached")
    if not reasons:
        reasons.append("partial convergence — volume threshold met but some criteria pending")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": "Project Aurelius v9.1",
        "search_statistics": {
            "total_screened": convergence.total_screened,
            "generations_run": convergence.generations,
            "viable_count": convergence.viable_count,
            "seed_pool_size": convergence.seed_pool_size,
            "final_score_variance": convergence.final_score_variance(),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        },
        "convergence": {
            "score_plateau": plateau,
            "structural_saturation": saturation,
            "new_clusters_last_batch": convergence.new_clusters_per_batch[-1]
            if convergence.new_clusters_per_batch
            else 0,
            "summary": (
                f"After {convergence.total_screened} molecules across "
                f"{convergence.generations} generations: "
                f"{convergence.viable_count} viable discoveries found. "
                f"Criteria: {', '.join(reasons)}."
            ),
        },
        "new_clusters_per_batch": convergence.new_clusters_per_batch,
        "discoveries": [
            {
                "smiles": d.smiles,
                "total_score": d.total_score,
                "is_viable": d.is_viable,
                "rejection_reasons": d.rejection_reasons,
                "novelty_to_seed": d.novelty_to_seed,
            }
            for d in sorted(discoveries, key=lambda r: -r.total_score)[:50]
        ],
        "all_results_count": len(all_results),
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Run summary written to %s", path)


def generate_discoveries_sdf(
    discoveries: list[ScreeningResult],
    path: str = "discoveries.sdf",
    output_dir: str | Path | None = None,
) -> None:
    """Write top-50 discoveries to SDF format for molecular viewer.

    Args:
        discoveries: List of discovery ScreeningResult objects.
        path: Output SDF path.
        output_dir: Directory to write to.
    """
    path = _resolve_output_path(path, output_dir)
    log = logging.getLogger("aurelius_agent")

    from rdkit import Chem

    top = sorted(discoveries, key=lambda r: -r.total_score)[:50]

    writer = Chem.SDWriter(str(path))
    for r in top:
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue
        mol.SetProp("SMILES", r.smiles)
        mol.SetProp("total_score", f"{r.total_score:.2f}")
        if r.novelty_to_seed is not None:
            mol.SetProp("novelty_to_seed", f"{r.novelty_to_seed:.4f}")
        writer.write(mol)
    writer.close()
    log.info("Discoveries SDF written to %s (%d molecules)", path, len(top))
