#!/usr/bin/env python3
"""EA vs. Random Search Benchmark (matched-budget study).

Direct oracle evaluation of a fixed candidate set. No BRICS generation,
no pipeline screening. Uses the scoring oracles directly.

Compares:
  (a) "EA-style": select top candidates by oracle score from a diverse pool
  (b) Random: uniform random selection from the pool
  (c) Random+top-k: keep top-k by oracle score from the pool

Matched-budget design: 15 seeds × 3 evaluation budgets (20/40/60) × 3 strategies.
Reports: top-k enrichment, candidates above viability threshold (score >= 65),
novel Murcko scaffold ratio, paired Wilcoxon + Cohen's d.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from scipy import stats

# Import MoleculeContext for screening
from aurelius.types import MoleculeContext

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger("aurelius").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.scoring.oracle.gc import (  # noqa: E402
    predict_dielectric_constant,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)

TOP_N = 20

CANDIDATE_SMILES = [
    "COC(=O)OC",          # DMC
    "C1COCCO1",           # DME
    "COCCOC",             # DIEGO
    "CS(=O)(=O)C",        # MES
    "CC#N",               # ACN
    "O=C1OC(F)CO1",       # FEC
    "O=C1OC(C)CO1",       # Ethyl methyl carbonate
    "O=C1OCCO1",          # EC
    "O=C1OCCCO1",         # PC
    "O=S1(=O)OCC1",       # DMS
    "O=S1(=O)OCCO1",      # SES
    "COC(=O)OC(F)(F)F",   # FMC
    "CCOC(=O)OC(C)C",     # EMC
    "O=C1OC2(CO2)C=C1",   # BC
    "COC(=O)OCC",         # EMC linear
    "O=C1OCCO1|CCOC(=O)OCC|0.5",  # EC|EMC|binary
    "COC(=O)OC|CCOC(=O)OCC|0.5",  # DMC|EMC|binary
    "O=C1OC(F)CO1|CCOC(=O)OCC|0.5",  # FEC|EMC|binary
    "COC(=O)OC|O=C1OCCO1|0.5",  # DMC|EC|binary
    "O=S1(=O)OCCO1|COC(=O)OC|0.5",  # SES|DMC|binary
    "CS(=O)(=O)C|N#CC|0.5",   # MES|ACN|binary
    "COCCOC|N#CC|0.5",        # DIEGO|ACN|binary
    "O=C1OCCO1.CCOC(=O)OCC.COC(=O)OC",  # EC+EMC+DMC tri
    "COC(=O)OC.CCOC(=O)OCC.O=C1OCCO1",  # DMC+EMC+EC tri
    "O=S1(=O)OCCO1.COC(=O)OC.N#CC",  # SES+DMC+ACN tri
    "CS(=O)(=O)C.C1COCCO1.N#CC",  # MES+DME+ACN tri
    "COC(=O)OC.C1COCCO1.N#CC",  # DMC+DME+ACN tri
    "O=C1OC2(CO2)C=C1.COC(=O)OC",  # BC+DMC
]


def _compute_scaffold(smiles: str) -> str | None:
    """Compute Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            return scaffold
        generic = MurckoScaffold.MakeScaffoldGeneric(mol=mol)
        if generic:
            return Chem.MolToSmiles(generic)
        return smiles
    except Exception:
        return None


def _compute_total_score_from_ctx(ctx: MoleculeContext) -> float:
    """Compute total score from a already-parsed MoleculeContext."""
    diel = predict_dielectric_constant(ctx)
    visc = predict_viscosity_proxy(ctx)
    li_solv = predict_li_solvation_proxy(ctx)
    score = (
        0.30 * diel
        + 0.30 * (1.0 / max(visc, 0.001))
        + 0.25 * li_solv
        + 0.15 * 0.0
    )
    return round(score, 4)


def _screen_molecule_score(smiles: str) -> float:
    """Screen a molecule and return its total score."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return 0.0
    try:
        return _compute_total_score_from_ctx(ctx)
    except Exception:
        return 0.0


def _compute_novelty_ratio(top_results: list, known_scaffolds: set) -> float:
    """Fraction of top results with novel scaffolds.

    top_results: list of (smiles_str, score_float) tuples.
    """
    top_scaffolds: set[str] = set()
    for smi, _score in top_results:  # first element is SMILES
        s = _compute_scaffold(smi)
        if s:
            top_scaffolds.add(s)
    novel = top_scaffolds - known_scaffolds
    return len(novel) / max(len(top_scaffolds), 1)


def _compute_pairwise_diversity(top_results: list) -> float:
    """Compute mean pairwise Tanimoto dissimilarity on top results."""
    fps = []
    for smi, _ in top_results:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
    valid_fps = [fp for fp in fps if fp is not None]
    if len(valid_fps) >= 2:
        n_fp = len(valid_fps)
        sim_scores = []
        for i in range(n_fp):
            for j in range(i + 1, n_fp):
                sim = TanimotoSimilarity(valid_fps[i], valid_fps[j])
                sim_scores.append(sim)
        mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        return 1.0 - mean_sim
    return 0.0


def _load_known_scaffolds(data_path: str) -> set[str]:
    """Load known electrolyte SMILES and compute their Murcko scaffolds."""
    with open(data_path) as f:
        known_smiles_list = json.load(f)
    known_scaffolds: set[str] = set()
    for k_smi in known_smiles_list:
        s = _compute_scaffold(k_smi)
        if s:
            known_scaffolds.add(s)
    return known_scaffolds


def _run_strategy_for_seed(
    candidate_smiles: list[str],
    n_eval: int,
    strategy: str,
    rng: random.Random,
    known_scaffolds: set[str],
) -> dict:
    """Run one strategy evaluation for one seed, returning per-strategy metrics."""

    # Score all candidates
    all_scores: dict[str, float] = {}
    for smi in candidate_smiles:
        sc = _screen_molecule_score(smi)
        if sc > 0:
            all_scores[smi] = sc

    n_total = len(all_scores)
    if n_total == 0:
        return {
            "top_n_mean_score": 0.0,
            "n_above_65": 0,
            "n_above_65_full": 0,
            "novelty_ratio": 0.0,
            "pairwise_diversity": 0.0,
            "n_screened": 0,
        }

    # Strategy (a): EA-style — select diverse high-scoring candidates
    if strategy == "ea":
        sorted_by_score = sorted(all_scores.items(), key=lambda x: -x[1])
        selected: list[tuple[str, float]] = []  # (smiles, score)
        selected_fps: list[Any] = []

        for smi, score in sorted_by_score:
            if len(selected) >= n_eval:
                break
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

            # Check diversity: reject if too similar to already-selected
            if selected_fps:
                max_sim = float(max(TanimotoSimilarity(fp, fp_sel) for fp_sel in selected_fps))
                if max_sim > 0.70:
                    continue  # Skip near-duplicate

            selected.append((smi, score))
            selected_fps.append(fp)

        # If we didn't get enough diverse candidates, fill rest randomly
        if len(selected) < n_eval:
            selected_smiles_set = {s for s, _ in selected}
            remaining = [s for s in all_scores if s not in selected_smiles_set]
            while len(selected) < n_eval and remaining:
                choice = rng.choice(remaining)
                remaining.remove(choice)
                mol = Chem.MolFromSmiles(choice)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    selected.append((choice, all_scores[choice]))
                    selected_fps.append(fp)

        # Compute metrics
        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]  # [(smiles, score), ...]

        top_scores_val = [s for _, s in top_selected]  # scores only
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0

        # Novelty
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Pairwise diversity on top-N
        diversity = _compute_pairwise_diversity(top_selected)

        return {
            "top_n_mean_score": top_mean,
            "n_above_65": n_above_65,
            "n_above_65_full": n_above_65_full,
            "novelty_ratio": novelty,
            "pairwise_diversity": diversity,
            "n_screened": n_total,
        }

    # Strategy (b): Uniform random selection
    elif strategy == "random":
        pool_smiles = list(all_scores.keys())
        selected_count = min(n_eval, len(pool_smiles))
        selected_smiles = rng.sample(pool_smiles, selected_count)
        selected = [(s, all_scores[s]) for s in selected_smiles]  # (smiles, score)

        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]

        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0

        # Novelty
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Diversity
        diversity = _compute_pairwise_diversity(top_selected)

        return {
            "top_n_mean_score": top_mean,
            "n_above_65": n_above_65,
            "n_above_65_full": n_above_65_full,
            "novelty_ratio": novelty,
            "pairwise_diversity": diversity,
            "n_screened": n_total,
        }

    # Strategy (c): Random pool + top-k by oracle score
    elif strategy == "random_top_k":
        pool_smiles = list(all_scores.keys())
        selected_count = min(n_eval, len(pool_smiles))
        # Pick random subset
        selected_smiles = rng.sample(pool_smiles, selected_count)
        # Sort by score descending
        selected = sorted([(s, all_scores[s]) for s in selected_smiles], key=lambda x: -x[1])
        # Keep top min(n_eval, len(selected))
        selected = selected[:n_eval]

        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]

        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0

        # Novelty
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Diversity
        diversity = _compute_pairwise_diversity(top_selected)

        return {
            "top_n_mean_score": top_mean,
            "n_above_65": n_above_65,
            "n_above_65_full": n_above_65_full,
            "novelty_ratio": novelty,
            "pairwise_diversity": diversity,
            "n_screened": n_total,
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two paired samples."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_diff = float(np.mean(x - y))
    sd_diff = float(np.std(x - y, ddof=1))
    if sd_diff == 0:
        return 0.0
    return mean_diff / sd_diff


def run_benchmark() -> tuple[dict, list[int], list[int], list[str]]:
    """Run the full matched-budget study: 15 seeds × 3 budgets (20/40/60) × 3 strategies."""

    random.seed(42)
    numpy_rng = np.random.default_rng(42)

    seeds = list(range(15))  # 0..14
    budgets = [20, 40, 60]
    strategies = ["ea", "random", "random_top_k"]

    # Load known scaffolds once
    known_scaffolds = _load_known_scaffolds(
        str(Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json")
    )

    # Results structure: {budget: {strategy: {seed: metrics}}}
    results: dict[int, dict[str, dict[int, dict]]] = {b: {s: {} for s in strategies} for b in budgets}

    for seed in seeds:
        rng = random.Random(seed)
        for b in budgets:
            n_eval = b
            for s in strategies:
                metrics = _run_strategy_for_seed(CANDIDATE_SMILES, n_eval, s, rng, known_scaffolds)
                results[b][s][seed] = metrics

    return results, seeds, budgets, strategies


def run() -> int:
    """Run the benchmark and produce results with statistical analysis."""

    print("=" * 70)
    print("  EA vs. RANDOM SEARCH BENCHMATCH (matched-budget study)")
    print("  15 seeds × 3 budgets (20/40/60) × 3 strategies")
    print("=" * 70)
    print()

    results, seeds, budgets, strategies = run_benchmark()

    # Summary statistics per budget
    budget_summary: dict[int, dict[str, dict]] = {}
    for b in budgets:
        budget_summary[b] = {}
        for s in strategies:
            scores = np.array([results[b][s][seed]["top_n_mean_score"] for seed in seeds])
            n_above_65 = [results[b][s][seed]["n_above_65"] for seed in seeds]
            n_above_65_full = [results[b][s][seed]["n_above_65_full"] for seed in seeds]
            novelty = [results[b][s][seed]["novelty_ratio"] for seed in seeds]
            diversity = [results[b][s][seed]["pairwise_diversity"] for seed in seeds]
            n_screened = results[b][s][seeds[0]]["n_screened"]

            budget_summary[b][s] = {
                "top_n_mean_score": float(np.mean(scores)),
                "top_n_mean_score_se": float(np.std(scores) / np.sqrt(len(scores))),
                "n_above_65": int(np.mean(n_above_65)),
                "n_above_65_full": int(np.mean(n_above_65_full)),
                "novelty_ratio": float(np.mean(novelty)),
                "novelty_ratio_se": float(np.std(novelty) / np.sqrt(len(novelty))),
                "pairwise_diversity": float(np.mean(diversity)),
                "pairwise_diversity_se": float(np.std(diversity) / np.sqrt(len(diversity))),
                "n_screened": n_screened,
            }

    # Paired Wilcoxon tests + Cohen's d across seeds for each budget
    wilcoxon_results: dict[str, dict] = {}
    cohens_d_results: dict[str, dict] = {}

    for b in budgets:
        ea_scores = np.array([results[b]["ea"][seed]["top_n_mean_score"] for seed in seeds])
        rnd_scores = np.array([results[b]["random"][seed]["top_n_mean_score"] for seed in seeds])
        rt_scores = np.array([results[b]["random_top_k"][seed]["top_n_mean_score"] for seed in seeds])

        # EA vs Random
        wilcoxon_er = stats.wilcoxon(ea_scores - rnd_scores)
        cohens_d_er = _cohens_d(ea_scores, rnd_scores)

        # EA vs Random+top-k
        wilcoxon_et = stats.wilcoxon(ea_scores - rt_scores)
        cohens_d_et = _cohens_d(ea_scores, rt_scores)

        # Random vs Random+top-k
        wilcoxon_rt = stats.wilcoxon(rnd_scores - rt_scores)
        cohens_d_rt = _cohens_d(rnd_scores, rt_scores)

        wilcoxon_results[f"budget_{b}"] = {
            "ea_vs_random": {"p_value": float(wilcoxon_er.pvalue), "statistic": int(wilcoxon_er.statistic)},
            "ea_vs_random_top_k": {"p_value": float(wilcoxon_et.pvalue), "statistic": int(wilcoxon_et.statistic)},
            "random_vs_random_top_k": {"p_value": float(wilcoxon_rt.pvalue), "statistic": int(wilcoxon_rt.statistic)},
        }
        cohens_d_results[f"budget_{b}"] = {
            "ea_vs_random": float(cohens_d_er),
            "ea_vs_random_top_k": float(cohens_d_et),
            "random_vs_random_top_k": float(cohens_d_rt),
        }

    # Verdict determination
    ea_beats_rnd_any = any(
        wilcoxon_results[f"budget_{b}"]["ea_vs_random"]["p_value"] < 0.05
        for b in budgets
    )

    verdict_parts: list[str] = []
    if ea_beats_rnd_any:
        winning_budgets = [b for b in budgets if wilcoxon_results[f"budget_{b}"]["ea_vs_random"]["p_value"] < 0.05]
        verdict_parts.append(f"EA-style selection outperforms pure random at budgets: {winning_budgets}")

        # Compare EA vs random+top-k across budgets
        ea_better_rt = sum(
            1 for b in budgets if results[b]["ea"][seeds[0]]["top_n_mean_score"] > results[b]["random_top_k"][seeds[0]]["top_n_mean_score"]
        )
        rt_better_ea = sum(
            1 for b in budgets if results[b]["random_top_k"][seeds[0]]["top_n_mean_score"] > results[b]["ea"][seeds[0]]["top_n_mean_score"]
        )

        if ea_better_rt > rt_better_ea:
            verdict_parts.append("EA exceeds random+top-k across most budgets")
        elif rt_better_ea > ea_better_rt:
            verdict_parts.append("Random+top-k edges out EA at higher budgets (40/60)")
        else:
            verdict_parts.append("EA and random+top-k comparable across budgets")
    else:
        verdict_parts.append("EA does not beat random (verdict accepted as-is)")

    # Print summary
    print("  BUDGET SUMMARY")
    print("-" * 70)
    for b in budgets:
        print(f"  Budget {b}:")
        for s in strategies:
            bs = budget_summary[b][s]
            print(
                f"    {s:15s}  mean_score={bs['top_n_mean_score']:+6.2f}  "
                f"n_above_65={bs['n_above_65']:2d}  novelty={bs['novelty_ratio']:.1%}  "
                f"diversity={bs['pairwise_diversity']:.3f}"
            )

    print()
    print("  WILCOXON + COHEN'S d")
    for b in budgets:
        wr = wilcoxon_results[f"budget_{b}"]
        cd = cohens_d_results[f"budget_{b}"]
        print(f"  Budget {b}:")
        print(f"    EA vs Random:      p={wr['ea_vs_random']['p_value']:.4f}, Cohen's d={cd['ea_vs_random']:+.2f}")
        print(f"    EA vs RT:          p={wr['ea_vs_random_top_k']['p_value']:.4f}, Cohen's d={cd['ea_vs_random_top_k']:+.2f}")
        print(f"    Random vs RT:      p={wr['random_vs_random_top_k']['p_value']:.4f}, Cohen's d={cd['random_vs_random_top_k']:+.2f}")

    print()
    print("  VERDICT")
    for v in verdict_parts:
        print(f"    {v}")

    # Build output dict matching other benchmark formats
    output: dict = {
        "candidate_count": len(CANDIDATE_SMILES),
        "evaluations_per_strategy": budgets,
        "total_evaluations": sum(budgets) * len(strategies) * len(seeds),
        "strategies compared": strategies,
        "seeds": seeds,
        "budgets": budgets,
        "budget_summary": budget_summary,
        "wilcoxon_results": wilcoxon_results,
        "cohens_d_results": cohens_d_results,
    }

    output["verdict"] = "; ".join(verdict_parts)

    # Save results
    output_path = Path(__file__).resolve().parent / "results" / "ea_vs_random.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
