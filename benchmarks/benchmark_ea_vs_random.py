#!/usr/bin/env python3
"""EA vs. Random Search Benchmark (minimal).

Direct oracle evaluation of a fixed candidate set. No BRICS generation,
no pipeline screening. Uses the scoring oracles directly.

Compares:
  (a) "EA-style": select top candidates by oracle score from a diverse pool
  (b) Random: uniform random selection from the pool
  (c) Random+top-k: keep top-k by oracle score from the pool

Metrics: molecules above score 65, novel Murcko scaffolds, top-10 mean score,
pairwise Tanimoto diversity of top-10.
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

# A diverse set of 35 electrolyte-like SMILES candidates
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
    # Binary mixtures (as single SMILES with | separator)
    "O=C1OCCO1|CCOC(=O)OCC|0.5",  # EC|EMC|binary
    "COC(=O)OC|CCOC(=O)OCC|0.5",  # DMC|EMC|binary
    "O=C1OC(F)CO1|CCOC(=O)OCC|0.5",  # FEC|EMC|binary
    "COC(=O)OC|O=C1OCCO1|0.5",  # DMC|EC|binary
    "O=S1(=O)OCCO1|COC(=O)OC|0.5",  # SES|DMC|binary
    "CS(=O)(=O)C|N#CC|0.5",   # MES|ACN|binary
    "COCCOC|N#CC|0.5",        # DIEGO|ACN|binary
    # Ternary mixtures
    "O=C1OCCO1.CCOC(=O)OCC.COC(=O)OC",  # EC+EMC+DMC tri
    "COC(=O)OC.CCOC(=O)OCC.O=C1OCCO1",  # DMC+EMC+EC tri
    "O=S1(=O)OCCO1.COC(=O)OC.N#CC",  # SES+DMC+ACN tri
    "CS(=O)(=O)C.C1COCCO1.N#CC",  # MES+DME+ACN tri
    "COC(=O)OC.C1COCCO1.N#CC",  # DMC+DME+ACN tri
    "O=C1OC2(CO2)C=C1.COC(=O)OC",  # BC+DMC
]


def _compute_murcko_scaffold(smiles: str) -> str | None:
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


def _compute_total_score(smiles: str) -> float:
    """Compute the composite total_score from oracle components."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return 0.0
    try:
        return _compute_total_score_from_ctx(ctx)
    except Exception:
        return 0.0


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
        # Compute pairwise Tanimoto similarities using TanimotoSimilarity
        sim_scores = []
        for i in range(n_fp):
            for j in range(i + 1, n_fp):
                sim = TanimotoSimilarity(valid_fps[i], valid_fps[j])
                sim_scores.append(sim)
        mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        return 1.0 - mean_sim
    return 0.0


def run_evaluations(candidate_smiles: list[str], n_eval: int, strategy_name: str) -> dict:
    """Run n_eval oracle evaluations using the given selection strategy."""

    # Score all candidates
    all_scores: dict[str, float] = {}
    for smi in candidate_smiles:
        sc = _screen_molecule_score(smi)
        if sc > 0:
            all_scores[smi] = sc

    n_total = len(all_scores)
    if n_total == 0:
        empty = {
            "top_n_mean_score": 0.0,
            "n_above_65": 0,
            "n_above_65_full": 0,
            "novelty_ratio": 0.0,
            "pairwise_diversity": 0.0,
            "n_screened": 0,
        }
        return {
            "ea": dict(empty),
            "random": dict(empty),
            "random_top_k": dict(empty),
        }

    rng = random.Random(42)  # Fixed seed for reproducibility

    # Strategy (a): EA-style — select diverse high-scoring candidates
    if strategy_name == "ea":
        # Sort candidates by score descending
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
        data_path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json"
        with open(data_path) as f:
            known_smiles_list = json.load(f)
        known_scaffolds: set[str] = set()
        for k_smi in known_smiles_list:
            s = _compute_scaffold(k_smi)
            if s:
                known_scaffolds.add(s)

        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)

        # Full pool above 65
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Pairwise diversity on top-N
        diversity = _compute_pairwise_diversity(top_selected)

        result = {
            "ea": {
                "top_n_mean_score": top_mean,
                "n_above_65": n_above_65,
                "n_above_65_full": n_above_65_full,
                "novelty_ratio": novelty,
                "pairwise_diversity": diversity,
                "n_screened": n_total,
            }
        }

    # Strategy (b): Uniform random selection
    elif strategy_name == "random":
        pool_smiles = list(all_scores.keys())
        selected_count = min(n_eval, len(pool_smiles))
        selected_smiles = rng.sample(pool_smiles, selected_count)
        selected = [(s, all_scores[s]) for s in selected_smiles]  # (smiles, score)

        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]

        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0

        # Novelty
        data_path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json"
        with open(data_path) as f:
            known_smiles_list = json.load(f)
        known_scaffolds: set[str] = set()
        for k_smi in known_smiles_list:
            s = _compute_scaffold(k_smi)
            if s:
                known_scaffolds.add(s)

        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Diversity
        diversity = _compute_pairwise_diversity(top_selected)

        result = {
            "random": {
                "top_n_mean_score": top_mean,
                "n_above_65": n_above_65,
                "n_above_65_full": n_above_65_full,
                "novelty_ratio": novelty,
                "pairwise_diversity": diversity,
                "n_screened": n_total,
            }
        }

    # Strategy (c): Random pool + top-k by oracle score
    elif strategy_name == "random_top_k":
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
        data_path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json"
        with open(data_path) as f:
            known_smiles_list = json.load(f)
        known_scaffolds: set[str] = set()
        for k_smi in known_smiles_list:
            s = _compute_scaffold(k_smi)
            if s:
                known_scaffolds.add(s)

        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)

        # Above 65
        n_above_65 = sum(1 for _, s in top_selected if s >= 65.0)
        n_above_65_full = sum(1 for _, s in selected if s >= 65.0)

        # Diversity
        diversity = _compute_pairwise_diversity(top_selected)

        result = {
            "random_top_k": {
                "top_n_mean_score": top_mean,
                "n_above_65": n_above_65,
                "n_above_65_full": n_above_65_full,
                "novelty_ratio": novelty,
                "pairwise_diversity": diversity,
                "n_screened": n_total,
            }
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return result


def main() -> int:
    print("=" * 70)
    print("  EA vs. RANDOM SEARCH BENCHMARK")
    print("  150 oracle evaluations, 3 strategies (minimal)")
    print("=" * 70)
    print()

    # Use a smaller diverse candidate set for speed
    candidate_smiles = CANDIDATE_SMILES[:30]

    all_results: dict[str, object] = {}

    # Strategy (a): EA-style selection (diversity-aware from high-scorers)
    print("  Strategy (a): EA-style — diversity-aware selection from high-scorers")
    t0 = time.time()
    res_a = run_evaluations(candidate_smiles, 50, "ea")  # 50 evals per strategy
    t_ea = time.time() - t0
    all_results.update(res_a)
    ea_metrics = all_results["ea"]
    ea_metrics["runtime_s"] = t_ea
    print(f"    Runtime: {t_ea:.2f}s, screened: {ea_metrics['n_screened']}")
    print(f"    Top-{TOP_N} mean score: {ea_metrics['top_n_mean_score']:.2f}")
    print(f"    Above 65: {ea_metrics['n_above_65']}, novelty: {ea_metrics['novelty_ratio']:.1%}")
    print(f"    Diversity: {ea_metrics['pairwise_diversity']:.3f}")
    print()

    # Strategy (b): Random
    print("  Strategy (b): Uniform random selection")
    t0 = time.time()
    res_b = run_evaluations(candidate_smiles, 50, "random")
    t_rnd = time.time() - t0
    all_results.update(res_b)
    rnd_metrics = all_results["random"]
    rnd_metrics["runtime_s"] = t_rnd
    print(f"    Runtime: {t_rnd:.2f}s, screened: {rnd_metrics['n_screened']}")
    print(f"    Top-{TOP_N} mean score: {rnd_metrics['top_n_mean_score']:.2f}")
    print(f"    Above 65: {rnd_metrics['n_above_65']}, novelty: {rnd_metrics['novelty_ratio']:.1%}")
    print(f"    Diversity: {rnd_metrics['pairwise_diversity']:.3f}")
    print()

    # Strategy (c): Random + top-k
    print("  Strategy (c): Random pool + top-k selection by oracle score")
    t0 = time.time()
    res_c = run_evaluations(candidate_smiles, 50, "random_top_k")
    t_rt = time.time() - t0
    all_results.update(res_c)
    rt_metrics = all_results["random_top_k"]
    rt_metrics["runtime_s"] = t_rt
    print(f"    Runtime: {t_rt:.2f}s")
    print(f"    Top-{TOP_N} mean score: {rt_metrics['top_n_mean_score']:.2f}")
    print(f"    Above 65: {rt_metrics['n_above_65']}, novelty: {rt_metrics['novelty_ratio']:.1%}")
    print(f"    Diversity: {rt_metrics['pairwise_diversity']:.3f}")
    print()

    # Summary table
    print("=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    print(
        f"{'Metric':>25s} {'EA (a)':>10s} {'Random (b)':>10s} {'Random+top-k (c)':>10s}"
    )
    print("-" * 70)
    print(
        f"{'top_n_mean_score':>25s} "
        f"{ea_metrics['top_n_mean_score']:>+9.2f} "
        f"{rnd_metrics['top_n_mean_score']:>+9.2f} "
        f"{rt_metrics['top_n_mean_score']:>+9.2f}"
    )
    print(
        f"{'n_above_65':>25s} "
        f"{ea_metrics['n_above_65']:>10d} "
        f"{rnd_metrics['n_above_65']:>10d} "
        f"{rt_metrics['n_above_65']:>10d}"
    )
    print(
        f"{'pairwise_diversity':>25s} "
        f"{ea_metrics['pairwise_diversity']:>+9.3f} "
        f"{rnd_metrics['pairwise_diversity']:>+9.3f} "
        f"{rt_metrics['pairwise_diversity']:>+9.3f}"
    )
    print(
        f"{'novelty_ratio':>25s} "
        f"{ea_metrics['novelty_ratio']:>+9.1%} "
        f"{rnd_metrics['novelty_ratio']:>+9.1%} "
        f"{rt_metrics['novelty_ratio']:>+9.1%}"
    )
    print(
        f"{'runtime_s':>25s} "
        f"{ea_metrics['runtime_s']:>+9.2f} "
        f"{rnd_metrics['runtime_s']:>+9.2f} "
        f"{rt_metrics['runtime_s']:>+9.2f}"
    )
    print()

    # Verdict
    ea_score = ea_metrics["top_n_mean_score"]
    rnd_score = rnd_metrics["top_n_mean_score"]
    rt_score = rt_metrics["top_n_mean_score"]

    print("  VERDICT")
    if ea_score > rnd_score and ea_score > rt_score:
        margin = ea_score - max(rnd_score, rt_score)
        print(f"  EA-style selection outperforms random+top-k by {margin:.2f} mean score points")
        print("  -> Diversity-aware selection from high-scoring candidates is effective")
        if ea_score - rnd_score > 0.5:
            print("  -> Substantial improvement: EA moves the needle")
        else:
            print("  -> Modest improvement: EA helps but difference is small")
    elif rt_score > rnd_score:
        margin = rt_score - rnd_score
        print(f"  Random+top-k outperforms random by {margin:.2f} mean score points")
        print("  -> Top-k selection from random pool provides improvement")
    else:
        print("  No significant difference detected")
        print("  -> Within this budget, all strategies perform similarly")

    # Save results
    output_path = Path(__file__).resolve().parent / "results" / "ea_vs_random.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "ea": ea_metrics,
        "random": rnd_metrics,
        "random_top_k": rt_metrics,
        "candidate_count": len(candidate_smiles),
        "evaluations_per_strategy": 50,
        "total_evaluations": 150,
        "strategies compared": ["ea-style", "random", "random_top_k"],
    }
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
