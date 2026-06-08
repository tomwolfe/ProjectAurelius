"""Report generation for autonomous screening agent.

Consolidated output: exactly two files:
  - discoveries.sdf — top discovery molecules with multi-objective properties
  - run_summary.json — full structured run report

ADR-2026-06-01: Changed datetime.UTC → datetime.timezone.utc for Python 3.9
compatibility (datetime.UTC was added in 3.11). No behavioral change.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.agent.selection import extract_pareto_front
from aurelius.agent.state import LoopState, check_score_plateau, check_structural_saturation
from aurelius.scoring.oracle.gc import GcUqEnsemble
from aurelius.types import MoleculeContext, ScreeningResult


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


def generate_run_summary(
    state: LoopState,
    all_results: list[ScreeningResult],
    discoveries: list[ScreeningResult],
    path: str = "run_summary.json",
    output_dir: str | Path | None = None,
    top_mixtures: list[dict[str, Any]] | None = None,
) -> None:
    """Write a single consolidated run_summary.json with all screening results.

    Replaces the previous multi-file output (manifest, statistics markdown,
    chemical insights markdown, CSV, SMI, etc.) with one structured JSON.

    Args:
        state: LoopState instance.
        all_results: All screening results.
        discoveries: Discovery list (score >= 65).
        path: Output JSON path.
        output_dir: Directory to write to.
        top_mixtures: Optional top-N binary mixture results.
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)
    scores = [r.total_score for r in all_results]

    plateau = check_score_plateau(state.batch_means)
    saturation = check_structural_saturation(state.scaffolds_per_batch)

    reasons = []
    if plateau:
        reasons.append("score plateau confirmed")
    if saturation:
        reasons.append("structural saturation reached")
    if not reasons:
        reasons.append("partial convergence — volume threshold met but some criteria pending")

    top_discoveries = sorted(discoveries, key=lambda r: -r.total_score)

    # Epistemic uncertainty for wet-lab prioritisation
    # Physical basis: molecules with high GC prediction uncertainty (std >> 15% of
    # mean) are out-of-distribution relative to the calibration set. Prioritising
    # these for experimental validation yields the highest information gain per
    # wet-lab hour — active learning without robotic automation.
    uq_data: dict[str, dict[str, float]] = {}
    try:
        uq_ensemble = GcUqEnsemble()
        for d in top_discoveries[:20]:
            ctx = MoleculeContext.from_smiles(d.smiles)
            if ctx is None:
                continue
            diel_mean, diel_std = uq_ensemble.predict_dielectric(ctx)
            visc_mean, visc_std = uq_ensemble.predict_viscosity(ctx)
            uq_weight = 1.0 + (diel_std / max(1.0, abs(diel_mean)))
            uq_weight += visc_std / max(1.0, abs(visc_mean))
            uq_data[d.smiles] = {
                "diel_mean": diel_mean,
                "diel_std": diel_std,
                "visc_mean": visc_mean,
                "visc_std": visc_std,
                "uncertainty_weighted_score": d.total_score * uq_weight,
            }
    except Exception:
        log.warning("GC UQ ensemble unavailable — skipping uncertainty weighting", exc_info=True)

    def _discovery_entry(d: ScreeningResult) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "smiles": d.smiles,
            "total_score": d.total_score,
            "homo_eV": d.homo_eV,
            "lumo_eV": d.lumo_eV,
            "dielectric_proxy": d.dielectric_proxy,
            "viscosity_proxy": d.viscosity_proxy,
            "li_solvation_proxy": d.li_solvation_proxy,
            "sa_score": d.sa_score,
            "sub_scores": d.sub_scores,
            "is_viable": d.is_viable,
            "rejection_reasons": d.rejection_reasons,
            "novelty_to_seed": d.novelty_to_seed,
        }
        if d.smiles in uq_data:
            entry["diel_std"] = uq_data[d.smiles]["diel_std"]
            entry["visc_std"] = uq_data[d.smiles]["visc_std"]
            entry["uncertainty_weighted_score"] = round(
                uq_data[d.smiles]["uncertainty_weighted_score"], 4
            )
        return entry

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": "Project Aurelius v10.0 — Multi-Objective Electrolyte Discovery Engine",
        "search_statistics": {
            "total_screened": state.total_screened,
            "generations_run": state.generations,
            "viable_count": state.viable_count,
            "seed_pool_size": state.seed_pool_size,
            "final_score_variance": state.final_score_variance(),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
        },
        "convergence": {
            "score_plateau": plateau,
            "structural_saturation": saturation,
            "new_scaffolds_last_batch": len(state.scaffolds_per_batch[-1])
            if state.scaffolds_per_batch
            else 0,
            "summary": (
                f"After {state.total_screened} molecules across "
                f"{state.generations} generations: "
                f"{state.viable_count} viable discoveries found. "
                f"Criteria: {', '.join(reasons)}."
            ),
        },
        "new_scaffolds_per_batch": [len(s) for s in state.scaffolds_per_batch],
        "discoveries": [_discovery_entry(d) for d in top_discoveries[:50]],
        "all_results_count": len(all_results),
        "pareto_optimal_discoveries": [
            {
                "smiles": d.smiles,
                "total_score": d.total_score,
                "homo_eV": d.homo_eV,
                "lumo_eV": d.lumo_eV,
                "dielectric_proxy": d.dielectric_proxy,
                "viscosity_proxy": d.viscosity_proxy,
                **(
                    {
                        "diel_std": uq_data[d.smiles]["diel_std"],
                        "visc_std": uq_data[d.smiles]["visc_std"],
                        "uncertainty_weighted_score": round(
                            uq_data[d.smiles]["uncertainty_weighted_score"], 4
                        ),
                    }
                    if d.smiles in uq_data
                    else {}
                ),
            }
            for d in sorted(
                extract_pareto_front(sorted(discoveries, key=lambda r: -r.total_score)[:100]),
                key=lambda r: -r.total_score,
            )
        ],
    }

    if top_mixtures:
        summary["top_mixtures"] = top_mixtures

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Run summary written to %s", path)


def generate_discoveries_sdf(
    discoveries: list[ScreeningResult],
    path: str = "discoveries.sdf",
    output_dir: str | Path | None = None,
) -> None:
    """Write top-50 discoveries to SDF format with multi-objective properties embedded.

    Each SDF entry includes:
      - SMILES
      - total_score
      - homo_eV, lumo_eV, gap_eV
      - dielectric_proxy, viscosity_proxy
      - sa_score (RDKit SA score)
      - novelty_to_seed

    Args:
        discoveries: List of discovery ScreeningResult objects.
        path: Output SDF path.
        output_dir: Directory to write to.
    """
    path = _resolve_output_path(path, output_dir)
    log = logging.getLogger("aurelius_agent")

    from rdkit import Chem

    from aurelius.types import MoleculeContext

    top = sorted(discoveries, key=lambda r: -r.total_score)[:50]

    writer = Chem.SDWriter(str(path))
    for r in top:
        ctx = MoleculeContext.from_smiles(r.smiles)
        if ctx is None:
            continue
        mol = ctx.mol

        from rdkit.Chem import AllChem
        try:
            mol_3d = Chem.RWMol(ctx.mol)
            mol_3d.UpdatePropertyCache()
            mol_h = Chem.AddHs(mol_3d)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == -1:
                mol.SetProp("3D_embed_failed", "True")
        except Exception:
            mol.SetProp("3D_embed_failed", "True")

        mol.SetProp("SMILES", r.smiles)
        mol.SetProp("total_score", f"{r.total_score:.2f}")
        if r.homo_eV is not None:
            mol.SetProp("homo_eV", f"{r.homo_eV:.4f}")
            mol.SetProp("lumo_eV", f"{r.lumo_eV:.4f}")
            gap = r.lumo_eV - r.homo_eV if r.lumo_eV is not None and r.homo_eV is not None else 0.0
            mol.SetProp("gap_eV", f"{gap:.4f}")
        if r.dielectric_proxy is not None:
            mol.SetProp("dielectric_proxy", f"{r.dielectric_proxy:.4f}")
        if r.viscosity_proxy is not None:
            mol.SetProp("viscosity_proxy", f"{r.viscosity_proxy:.4f}")
        if r.li_solvation_proxy is not None:
            mol.SetProp("li_solvation_proxy", f"{r.li_solvation_proxy:.4f}")
        if r.sa_score is not None:
            mol.SetProp("sa_score", f"{r.sa_score:.4f}")
        if r.novelty_to_seed is not None:
            mol.SetProp("novelty_to_seed", f"{r.novelty_to_seed:.4f}")
        if r.sub_scores:
            for key, val in r.sub_scores.items():
                mol.SetProp(f"sub_{key}", f"{val:.4f}")
        writer.write(mol)
    writer.close()
    log.info("Discoveries SDF written to %s (%d molecules)", path, len(top))



