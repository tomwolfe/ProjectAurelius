"""Report generation for autonomous screening agent.

Consolidated output: exactly two files:
  - discoveries.sdf — top discovery molecules with multi-objective properties
  - run_summary.json — full structured run report

Also provides:
  - generate_xtb_input — exports top-10 candidates as .xyz files for xTB/DFT validation
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.agent.state import LoopState
from aurelius.types import ScreeningResult


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
    """
    log = logging.getLogger("aurelius_agent")

    path = _resolve_output_path(path, output_dir)
    scores = [r.total_score for r in all_results]

    plateau = state.check_score_plateau()
    saturation = state.check_structural_saturation()

    reasons = []
    if plateau:
        reasons.append("score plateau confirmed")
    if saturation:
        reasons.append("structural saturation reached")
    if not reasons:
        reasons.append("partial convergence — volume threshold met but some criteria pending")

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
            "new_clusters_last_batch": state.new_clusters_per_batch[-1]
            if state.new_clusters_per_batch
            else 0,
            "summary": (
                f"After {state.total_screened} molecules across "
                f"{state.generations} generations: "
                f"{state.viable_count} viable discoveries found. "
                f"Criteria: {', '.join(reasons)}."
            ),
        },
        "new_clusters_per_batch": state.new_clusters_per_batch,
        "discoveries": [
            {
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

    top = sorted(discoveries, key=lambda r: -r.total_score)[:50]

    writer = Chem.SDWriter(str(path))
    for r in top:
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue
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


def generate_xtb_input(
    discoveries: list[ScreeningResult],
    output_dir: str | Path = "xtb_input",
    top_n: int = 10,
) -> list[str]:
    """Export top-N discoveries as .xyz files for xTB/DFT single-point energy calculations.

    Each molecule is embedded into 3D coordinates using RDKit's ETKDG conformer
    generation, followed by MMFF94 force-field optimization to produce
    chemically sensible geometries. Output files are named ``<rank>_<smiles>.xyz``
    where special characters in SMILES are replaced with underscores.

    Args:
        discoveries: List of discovery ScreeningResult objects.
        output_dir: Directory to write .xyz files to.
        top_n: Number of top discoveries to export (default 10).

    Returns:
        List of paths to the generated .xyz files.
    """
    log = logging.getLogger("aurelius_agent")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    top = sorted(discoveries, key=lambda r: -r.total_score)[:top_n]
    xyz_paths: list[str] = []

    for rank, r in enumerate(top, start=1):
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue

        try:
            mol_h = Chem.AddHs(mol)
            conf_id = AllChem.EmbedMolecule(
                mol_h,
                randomSeed=42,
                useRandomCoords=False,
                useETKDGv3=True,
            )
            if conf_id < 0:
                # Fall back to random coordinates if ETKDG fails
                conf_id = AllChem.EmbedMolecule(
                    mol_h, randomSeed=42, useRandomCoords=True
                )
            if conf_id < 0:
                continue

            AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)

            xyz_block = Chem.MolToXYZBlock(mol_h, confId=conf_id)

            safe_name = r.smiles.replace("/", "_").replace("\\", "_").replace("[", "_").replace("]", "_")
            safe_name = safe_name.replace("(", "_").replace(")", "_").replace("=", "_")
            safe_name = safe_name.replace("#", "_").replace("@", "_")
            # Truncate to avoid excessively long filenames
            safe_name = safe_name[:60]

            xyz_path = output_path / f"{rank:03d}_{safe_name}.xyz"
            with open(xyz_path, "w") as f:
                # Add comment line with metadata
                f.write(xyz_block.rstrip("\n") + "\n")
                f.write(f"# total_score={r.total_score:.2f}")
                if r.homo_eV is not None:
                    f.write(f" homo={r.homo_eV:.4f}")
                if r.lumo_eV is not None:
                    f.write(f" lumo={r.lumo_eV:.4f}")
                f.write("\n")

            xyz_paths.append(str(xyz_path))

        except Exception as exc:
            log.debug("Failed to generate 3D geometry for %s: %s", r.smiles, exc)
            continue

    log.info("xTB input files written to %s/ (%d molecules)", output_dir, len(xyz_paths))
    return xyz_paths
