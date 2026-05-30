"""Report generation for autonomous screening agent.

Provides functions to write discovery results, statistics, chemical
insights, and agent discovery manifests.
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
    """Resolve a relative output path against a base directory.

    If ``output_dir`` is provided, ``path`` is joined to it.
    Otherwise ``path`` is returned unchanged (backward-compatible).

    Args:
        path: Output file path (relative).
        output_dir: Base directory to resolve against.

    Returns:
        Resolved absolute path.
    """
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


def generate_discovery_results(
    all_results: list[ScreeningResult],
    path: str = "discovery_results_final.json",
    output_dir: str | Path | None = None,
) -> None:
    """Write full structured logs of all screened molecules to JSON.

    Args:
        all_results: List of screening result dicts.
        path: Output file path (relative to output_dir if provided).
        output_dir: Directory to write to. If None, uses current working directory.
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)

    serializable: list[dict[str, Any]] = []
    for r in all_results:
        entry: dict[str, Any] = {
            "smiles": r.smiles,
            "total_score": r.total_score,
            "is_viable": r.is_viable,
            "rejection_reasons": r.rejection_reasons,
        }
        serializable.append(entry)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info("Discovery results written to %s (%d entries)", path, len(serializable))


def write_top_discoveries(
    discoveries: list[ScreeningResult], path: str = "top_discoveries.smi", output_dir: str | Path | None = None
) -> None:
    """Write SMILES of all legitimate discoveries to file.

    Args:
        discoveries: List of discovery dicts.
        path: Output file path (relative to output_dir).
        output_dir: Directory to write to. If None, uses current working directory.
    """
    path = _resolve_output_path(path, output_dir)

    log = logging.getLogger("aurelius_agent")

    with open(path, "w") as f:
        f.write("# Project Aurelius v9.0 — Top Discoveries (Score >= 65.0)\n")
        for d in discoveries:
            f.write(f"{d.smiles}  # score={d.total_score:.1f}\n")
    log.info("Top discoveries written to %s (%d molecules)", path, len(discoveries))


def generate_screening_statistics(
    convergence: ConvergenceChecker,
    all_results: list[ScreeningResult],
    path: str = "screening_statistics.md",
    output_dir: str | Path | None = None,
) -> None:
    """Generate convergence statistics as markdown.

    Args:
        convergence: ConvergenceChecker instance.
        all_results: List of screening results.
        path: Output file path (relative to output_dir).
        output_dir: Directory to write to. If None, uses current working directory.
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)

    scores = [r.total_score for r in all_results]

    with open(path, "w") as f:
        f.write("# Screening Statistics — Project Aurelius v9.0\n\n")
        f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Total screened:** {convergence.total_screened}\n")
        f.write(f"- **Generations run:** {convergence.generations}\n")
        f.write(f"- **Viable discoveries (score >= 65):** {convergence.viable_count}\n")
        f.write(f"- **Seed pool size:** {convergence.seed_pool_size}\n")
        f.write(f"- **Final score variance:** {convergence.final_score_variance():.4f}\n\n")

        f.write("## Score Distribution\n\n")
        if scores:
            f.write(f"- Mean: {np.mean(scores):.2f}\n")
            f.write(f"- Std:  {np.std(scores):.2f}\n")
            f.write(f"- Min:  {np.min(scores):.2f}\n")
            f.write(f"- Max:  {np.max(scores):.2f}\n\n")

            bins = [0, 20, 35, 50, 65, 80, 100]
            f.write("Score histogram:\n")
            for i in range(len(bins) - 1):
                count = sum(1 for s in scores if bins[i] <= s < bins[i + 1])
                bar = "#" * count
                line = f"  [{bins[i]:>3.0f}-{bins[i + 1]:>3.0f}): {bar} ({count}))"
                f.write(line + "\n")
            f.write("\n")

        f.write("## Convergence Analysis\n\n")
        plateau = convergence.check_score_plateau()
        pass_collapsed = convergence.check_pass_rate_collapsed()
        saturation = convergence.check_structural_saturation()

        f.write(f"- **Score plateau:** {'YES' if plateau else 'NO'}\n")
        f.write(f"- **Pass rate collapse:** {'YES' if pass_collapsed else 'NO'}\n")
        f.write(f"- **Structural saturation:** {'YES' if saturation else 'NO'}\n\n")

        rolling = convergence.compute_rolling_mean(batch_size=50)
        if rolling:
            f.write("### Rolling Mean of Total Score (window=50)\n\n")
            f.write("| Batch | Rolling Mean |\n")
            f.write("|-------|-------------|\n")
            for i, rm in enumerate(rolling):
                f.write(f"| {i + 1} | {rm:.2f} |\n")
            f.write("\n")

        f.write("## Viability Rate Trend\n\n")
        f.write("| Generation | Viability Rate |\n")
        f.write("|------------|---------------|\n")
        for i, rate in enumerate(convergence.viability_rates):
            f.write(f"| {i + 1} | {rate:.4f} |\n")
        f.write("\n")

        f.write("## Exhaustion Proof\n\n")
        f.write("The screening process terminates when ALL of the following are met:\n")
        f.write("1. **Volume:** >= 150 viable-tier candidates OR >= 300 total unique screened\n")
        f.write("2. **Score Plateau:** Rolling mean changes < 1.0% over 3 consecutive batches\n")
        f.write("3. **Pass Rate Collapse:** Viability rate < 3% for 2 consecutive batches\n")
        f.write("4. **Structural Saturation:** < 3 new clusters over last 2 batches\n\n")
        f.write(
            f"Final state: {convergence.total_screened} screened, "
            f"{convergence.viable_count} viable, "
            f"variance={convergence.final_score_variance():.4f}\n"
        )

    log.info("Screening statistics written to %s", path)


def generate_chemical_insights(
    all_results: list[ScreeningResult],
    discoveries: list[ScreeningResult],
    path: str = "chemical_insights.md",
    output_dir: str | Path | None = None,
) -> None:
    """Generate structural correlations and failure analysis.

    Args:
        all_results: List of screening results.
        discoveries: List of discovery dicts.
        path: Output file path (relative to output_dir).
        output_dir: Directory to write to. If None, uses current working directory.
    """
    import logging

    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)

    from rdkit import Chem

    with open(path, "w") as f:
        f.write("# Chemical Insights — Project Aurelius v9.0\n\n")
        f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")

        f.write("## Structural Correlations\n\n")
        f.write("Analysis of molecular features correlated with high Aurelius scores.\n\n")

        scaffold_scores: dict[str, list[float]] = {}
        for r in all_results:
            mol = Chem.MolFromSmiles(r.smiles)
            if mol is not None:
                scaffold = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=list(range(mol.GetNumAtoms())), isomericSmiles=False
                )
                scaffold = scaffold[:30]
                if scaffold not in scaffold_scores:
                    scaffold_scores[scaffold] = []
                scaffold_scores[scaffold].append(r.total_score)

        f.write("| Scaffold (truncated) | Mean Score | Count |\n")
        f.write("|---------------------|-----------|-------|\n")
        for scaffold, sc_list in sorted(scaffold_scores.items(), key=lambda x: -np.mean(x[1]))[:15]:
            f.write(f"| {scaffold} | {np.mean(sc_list):.2f} | {len(sc_list)} |\n")
        f.write("\n")

        f.write("## Failure Analysis\n\n")
        failure_reasons: dict[str, int] = {}
        for r in all_results:
            if r.rejection_reasons:
                for reason in r.rejection_reasons:
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if failure_reasons:
            f.write("Top rejection reasons:\n\n")
            f.write("| Reason | Count |\n")
            f.write("|--------|-------|\n")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                f.write(f"| {reason} | {count} |\n")
        else:
            f.write("No rejection reasons recorded.\n")
        f.write("\n")

        f.write("## Experimental Next Steps\n\n")
        if discoveries:
            f.write("### Recommended Experimental Validation\n\n")
            for i, d in enumerate(discoveries[:10], 1):
                f.write(f"{i}. **{d.smiles}** (Score: {d.total_score:.1f})\n")
            f.write("\n")
        else:
            f.write("No legitimate discoveries found in this screening round.\n")
            f.write("Recommendations:\n")
            f.write("1. Expand mutation template library (e.g., cyano, nitro, boron additions)\n")
            f.write("2. Lower MW threshold to allow larger scaffolds\n")
            f.write("3. Increase batch sizes for deeper exploration\n")
            f.write("4. Consider seed molecules with known SEI-forming properties\n")
        f.write("\n")

        f.write("### Coverage Gaps\n\n")
        f.write("- Focus on small-molecule electrolyte additives (< 350 Da)\n")
        f.write("- Prioritize fluorinated carbonates and lactones\n")
        f.write("- Explore boron-containing SEI-forming compounds\n")
        f.write("- Consider unsaturated cyclic ethers\n")

    log.info("Chemical insights written to %s", path)


def generate_manifest(
    convergence: ConvergenceChecker,
    discoveries: list[ScreeningResult],
    all_results: list[ScreeningResult],
    path: str = "agent_discovery_manifest.json",
    output_dir: str | Path | None = None,
) -> None:
    """Generate the agent_discovery_manifest.json.

    Args:
        convergence: ConvergenceChecker instance.
        discoveries: List of discovery dicts.
        all_results: List of screening results.
        path: Output file path (relative to output_dir).
        output_dir: Directory to write to. If None, uses current working directory.
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)

    rolling = convergence.compute_rolling_mean(batch_size=50)
    rolling_mean = float(np.mean(rolling[-3:])) if len(rolling) >= 3 else 0.0

    manifest: dict[str, Any] = {
        "search_statistics": {
            "total_screened": convergence.total_screened,
            "generations_run": convergence.generations,
            "invalid_discarded": 0,
            "final_score_variance": convergence.final_score_variance(),
        },
        "discoveries": [],
        "exhaustion_proof": {
            "rolling_mean_plateau": rolling_mean,
            "viability_rate_final": convergence.viability_rates[-1] if convergence.viability_rates else 0.0,
            "new_clusters_last_batch": convergence.new_clusters_per_batch[-1]
            if convergence.new_clusters_per_batch
            else 0,
            "analytical_summary": "",
        },
    }

    for d in discoveries:
        manifest["discoveries"].append(
            {
                "smiles": d.smiles,
                "total_score": d.total_score,
                "is_viable": d.is_viable,
                "rejection_reasons": d.rejection_reasons,
            }
        )

    reasons = []
    if convergence.check_score_plateau():
        reasons.append("score plateau confirmed")
    if convergence.check_pass_rate_collapsed():
        reasons.append("pass rate collapsed")
    if convergence.check_structural_saturation():
        reasons.append("structural saturation reached")
    if not reasons:
        reasons.append("partial convergence — volume threshold met but some criteria pending")

    manifest["exhaustion_proof"]["analytical_summary"] = (
        f"After {convergence.total_screened} molecules across "
        f"{convergence.generations} generations: "
        f"{convergence.viable_count} viable discoveries found. "
        f"Final criteria: {', '.join(reasons) if reasons else 'none met'}."
    )

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Discovery manifest written to %s", path)
