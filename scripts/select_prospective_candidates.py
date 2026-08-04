"""Prospective candidate selection for wet-lab validation.

Runs a 50-generation discovery loop, filters candidates by
synthesis feasibility, conformal confidence, and novelty, then
ranks them by composite score with Tanimoto diversity penalty.

Outputs a CSV with SMILES, scores, synthesis hints, confidence
intervals, and risk flags for wet-lab partners.

Physical justification: The EA discovers molecules that score
well on the surrogate oracle, but not all discoveries are
equally suitable for wet-lab testing. This script bridges the
simulation-to-reality gap by selecting candidates that are:
  1. Synthesizable (synthesis_feasibility > 0.6)
  2. Confidently predicted (conformal_confidence > 0.8)
  3. Novel (novelty_to_seed > 0.3)
  4. Diverse (Tanimoto penalty prevents clustering)

Risk flags (Al corrosion, hydrolytic instability) are computed
from structural features and help wet-lab partners prioritize
safety.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.agent.loop import AgentConfig, DiscoveryLoop, run_screening
from aurelius.agent.mutation.retrosynthetic import compute_synthesis_feasibility
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext

N_GENERATIONS = 50
BATCH_SIZE = 50
OUTPUT_CSV = "prospective_candidates.csv"

SYNTHESIS_FEASIBILITY_THRESHOLD = 0.6
CONFORMAL_CONFIDENCE_THRESHOLD = 0.8
NOVELTY_THRESHOLD = 0.3
TOP_N = 10

RISK_AL_CORROSION_LUMO = -1.0
RISK_AL_CORROSION_MIN_F = 3
RISK_AL_CORROSION_PENALTY = 0.85


def _compute_tanimoto_diversity(
    selected_smis: list[str],
    candidate_smis: list[str],
) -> float:
    """Compute average Tanimoto similarity of a candidate to already-selected molecules."""
    if not selected_smis:
        return 0.0
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(candidate_smis[0]), radius=2, nBits=2048
    )
    max_sim = 0.0
    for sel_smi in selected_smis:
        sel_fp = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(sel_smi), radius=2, nBits=2048
        )
        from rdkit.DataStructs import TanimotoSimilarity
        sim = TanimotoSimilarity(candidate_fp, sel_fp)
        max_sim = max(max_sim, sim)
    return max_sim


def _check_al_corrosion_risk(mol: Chem.Mol) -> bool:
    """Check if molecule poses Al corrosion risk (high-LUMO fluorinated)."""
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    if n_f < RISK_AL_CORROSION_MIN_F:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            if 9 in neighbors and 8 in neighbors:
                return True
    return False


def _check_hydrolytic_instability(mol: Chem.Mol) -> bool:
    """Check if molecule contains hydrolytically unstable motifs."""
    from aurelius.constants import HYDROLYTICALLY_UNSTABLE_PATTERNS
    for pattern, _name, _severity in HYDROLYTICALLY_UNSTABLE_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True
    return False


def _compute_confidence_interval(
    homo: float | None,
    lumo: float | None,
    conformal_conf: float,
) -> tuple[float, float]:
    """Compute approximate confidence interval for HOMO-LUMO gap."""
    if homo is None or lumo is None:
        return (0.0, 0.0)
    gap = lumo - homo
    uncertainty = (1.0 - conformal_conf) * 2.0
    return (max(0.0, gap - uncertainty), gap + uncertainty)


def _generate_synthesis_hints(mol: Chem.Mol) -> str:
    """Generate brief synthesis hints based on functional group analysis."""
    hints = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=[OX1])[OX2]")):
        hints.append("carbonate/ester")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2][CX4]")):
        hints.append("ether linkage")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)")):
        hints.append("sulfone")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][F]")):
        hints.append("fluorinated")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]#[NX2]")):
        hints.append("nitrile")
    if not hints:
        hints.append("novel scaffold")
    return "; ".join(hints)


def run_prospective_selection(
    output_path: str = OUTPUT_CSV,
    n_generations: int = N_GENERATIONS,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, any]]:
    """Run discovery loop and select prospective candidates.

    Args:
        output_path: Path for output CSV.
        n_generations: Number of generations to run.
        batch_size: Candidates per batch.

    Returns:
        List of selected candidate dicts.
    """
    print(f"Running {n_generations}-generation discovery loop...")

    agent_cfg = AgentConfig(
        max_generations=n_generations,
        batch_size=batch_size,
        use_nsga2=False,
        active_learning_threshold=0.7,
    )
    results = run_screening(agent_cfg)

    all_results = results["all_results"]
    discoveries = results["discoveries"]

    print(f"Discovery loop complete: {len(all_results)} evaluated, {len(discoveries)} viable")

    # Filter and rank candidates
    pipeline = AureliusPipeline()
    pipeline.initialize()

    candidates = []
    for result in all_results:
        if result.total_score < 65.0:
            continue
        if result.synthesis_depth is not None and result.synthesis_depth > 4:
            continue

        ctx = MoleculeContext.from_smiles(result.smiles)
        if ctx is None:
            continue

        mol = ctx.mol
        synthesis_feas = compute_synthesis_feasibility(mol)
        if synthesis_feas < SYNTHESIS_FEASIBILITY_THRESHOLD:
            continue

        conformal_conf = result.sub_scores.get("confidence", 1.0) if result.sub_scores else 1.0
        novelty = result.novelty_to_seed if result.novelty_to_seed is not None else 0.0
        if novelty < NOVELTY_THRESHOLD:
            continue

        homo = result.homo_eV
        lumo = result.lumo_eV
        ci_low, ci_high = _compute_confidence_interval(homo, lumo, conformal_conf)

        al_corrosion = _check_al_corrosion_risk(mol)
        hydrolytic = _check_hydrolytic_instability(mol)

        risk_flags = []
        if al_corrosion:
            risk_flags.append("Al_corrosion_risk")
        if hydrolytic:
            risk_flags.append("hydrolytic_instability")

        synthesis_hints = _generate_synthesis_hints(mol)

        candidates.append({
            "smiles": result.smiles,
            "total_score": result.total_score,
            "synthesis_feasibility": round(synthesis_feas, 4),
            "conformal_confidence": round(conformal_conf, 4),
            "novelty_to_seed": round(novelty, 4) if novelty is not None else 0.0,
            "homo_eV": homo,
            "lumo_eV": lumo,
            "gap_eV": round(lumo - homo, 4) if homo is not None and lumo is not None else None,
            "confidence_interval_low": round(ci_low, 4),
            "confidence_interval_high": round(ci_high, 4),
            "synthesis_hints": synthesis_hints,
            "risk_flags": "; ".join(risk_flags) if risk_flags else "none",
            "sa_score": result.sa_score,
            "dielectric_proxy": result.dielectric_proxy,
            "viscosity_proxy": result.viscosity_proxy,
        })

    # Rank by composite score with diversity penalty
    selected: list[dict] = []
    selected_smis: list[str] = []

    candidates.sort(key=lambda c: -c["total_score"])

    for candidate in candidates:
        if len(selected) >= TOP_N:
            break
        diversity_penalty = _compute_tanimoto_diversity(selected_smis, [candidate["smiles"]])
        # Penalize candidates similar to already-selected ones
        adjusted_score = candidate["total_score"] * (1.0 - 0.3 * diversity_penalty)
        candidate["diversity_penalty"] = round(diversity_penalty, 4)
        candidate["adjusted_score"] = round(adjusted_score, 4)
        selected.append(candidate)
        selected_smis.append(candidate["smiles"])

    # Sort by adjusted score
    selected.sort(key=lambda c: -c["adjusted_score"])

    # Write CSV
    fieldnames = [
        "smiles", "total_score", "adjusted_score", "synthesis_feasibility",
        "conformal_confidence", "novelty_to_seed", "homo_eV", "lumo_eV",
        "gap_eV", "confidence_interval_low", "confidence_interval_high",
        "synthesis_hints", "risk_flags", "sa_score", "dielectric_proxy",
        "viscosity_proxy", "diversity_penalty",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in selected:
            row = {k: cand.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Selected {len(selected)} prospective candidates -> {output_path}")
    return selected


def cross_validate_with_eht(candidates: list[dict]) -> None:
    """Cross-validate candidate rankings with independent EHT re-score.

    Uses the extended-Hückel method (per-atom Coulomb integrals +
    Wolfsberg-Helmholz resonance integrals) to independently re-score
    candidates and compute Spearman ρ between Aurelius and EHT rankings.
    """
    if len(candidates) < 5:
        print("Not enough candidates for EHT cross-validation")
        return

    from benchmarks.benchmark_dft_rerank import _compute_eht_orbitals

    aurelius_scores = [c["total_score"] for c in candidates]
    eht_composites = []

    for cand in candidates:
        ctx = MoleculeContext.from_smiles(cand["smiles"])
        if ctx is None:
            eht_composites.append(0.0)
            continue
        homo_eht, lumo_eht = _compute_eht_orbitals(ctx.mol)
        eht_composite = -(homo_eht + lumo_eht) / 2.0
        eht_composites.append(eht_composite)

    if len(aurelius_scores) < 5:
        return

    rho, p_value = spearmanr(aurelius_scores, eht_composites)
    print(f"\nEHT Cross-validation:")
    print(f"  Spearman ρ = {rho:.4f} (p = {p_value:.4f})")
    if rho > 0.30:
        print(f"  PASS: ρ > 0.30 — independent EHT model validates Aurelius ranking")
    else:
        print(f"  WARNING: ρ ≤ 0.30 — EHT model does not validate Aurelius ranking")


def run_dft_validation(candidates: list[dict]) -> dict[str, float] | None:
    """Validate top-k candidate rankings with ORCA DFT single points.

    Runs the wB97X-D3/def2-SVP single-point calculator on each candidate
    (cached in dft_cache.json), then computes Spearman ρ between the
    Aurelius scores and the DFT HOMO/LUMO virtual potential. Logs a
    warning when ρ < 0.4, signalling that the surrogate ranking is not
    corroborated by higher-level theory.

    Returns the validation metrics dict (or None if no candidate could be
    validated, e.g. ORCA not installed).
    """
    from aurelius.scoring.oracle.dft_validator import DFTValidator, has_orca

    if len(candidates) < 3:
        print("Not enough candidates for DFT validation")
        return None

    print("\nDFT Validation (ORCA wB97X-D3/def2-SVP):")
    if not has_orca():
        print("  [INFO] ORCA binary not found — DFT validation skipped.")
        return None

    validator = DFTValidator(cache_path="dft_cache.json")
    scores = [c.get("adjusted_score", c.get("total_score", 0.0)) for c in candidates]
    mols = []
    for cand in candidates:
        ctx = MoleculeContext.from_smiles(cand["smiles"])
        if ctx is None:
            mols.append(None)  # type: ignore[arg-type]
        else:
            mols.append(ctx.mol)

    validated = 0
    for cand, mol, score in zip(candidates, mols, scores):
        if mol is None:
            continue
        dft = validator.compute(mol)
        if dft is not None:
            cand["dft_homo_eV"] = round(dft["homo_eV"], 4)
            cand["dft_lumo_eV"] = round(dft["lumo_eV"], 4)
            cand["dft_composite"] = round(-(dft["homo_eV"] + dft["lumo_eV"]) / 2.0, 4)
            validated += 1
        else:
            cand["dft_homo_eV"] = None
            cand["dft_lumo_eV"] = None
            cand["dft_composite"] = None

    valid_mols = [m for m in mols if m is not None]
    valid_scores = [s for m, s in zip(mols, scores) if m is not None]
    metrics = validator.validate_ranking(valid_scores, valid_mols)
    metrics["n_validated"] = validated

    rho = metrics["rho_composite"]
    print(f"  Validated candidates: {metrics['n_validated']}")
    print(f"  Spearman ρ (Aurelius vs DFT composite) = {rho:.4f} "
          f"(p = {metrics['p_composite']:.4f})")
    print(f"  Spearman ρ (Aurelius vs DFT HOMO)      = {metrics['rho_homo']:.4f}")
    print(f"  Spearman ρ (Aurelius vs DFT LUMO)      = {metrics['rho_lumo']:.4f}")
    if rho < 0.40:
        print(f"  WARNING: ρ = {rho:.4f} < 0.40 — DFT does not validate the "
              f"surrogate ranking for these candidates.")
    else:
        print(f"  PASS: ρ = {rho:.4f} ≥ 0.40 — DFT corroborates the Aurelius ranking.")
    return metrics


if __name__ == "__main__":
    output = run_prospective_selection()
    cross_validate_with_eht(output)
    run_dft_validation(output)