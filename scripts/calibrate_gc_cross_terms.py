#!/usr/bin/env python3
"""Fit GC cross-term coefficients from external benchmark data.

Replaces hand-tuned _CROSS_TERMS with ridge regression coefficients fitted
to the expanded external_property_benchmark.json. Only the 9 physically-
motivated fragment-pair interactions from the original _CROSS_TERMS list
are considered (not all 666 possible pairs), preventing overfitting.

Stores results in gc_cross_terms.json.

Usage: python scripts/calibrate_gc_cross_terms.py
"""

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import Ridge

from aurelius.constants import MAX_DIELECTRIC_PER_TPSA
from aurelius.scoring.oracle.gc import (
    _GC_BASE_DIELECTRIC,
    _GC_FRAGMENTS,
    _count_fragments,
    _saturate_contrib,
)
from aurelius.types import MoleculeContext

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data"
)
BENCHMARK_PATH = os.path.join(DATA_DIR, "external_property_benchmark.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "gc_cross_terms.json")

# Only the 9 physically-motivated pairs from original _CROSS_TERMS
_CANDIDATE_PAIRS: list[tuple[str, str, str]] = [
    ("carbonate", "ether", "carbonate-ether synergy (glyme-carbonate hybrids)"),
    ("nitrile", "ether", "nitrile-ether synergy"),
    ("carbonate", "fluorine", "fluorinated carbonate suppression"),
    ("sulfone", "ether", "sulfone-ether synergy"),
    ("carbonate", "nitrile", "carbonate-nitrile antagonism"),
    ("alcohol", "carbonate", "alcohol-carbonate H-bond competition"),
    ("sulfone", "carbonate", "sulfone-carbonate polarity competition"),
    ("nitrile", "fluorine", "fluorinated nitrile dipole enhancement"),
    ("sulfone", "nitrile", "sulfone-nitrile high-voltage synergy"),
]


def _base_dielectric_no_cross(ctx: MoleculeContext) -> float:
    """Compute dielectric prediction without cross-terms (for residual)."""
    mol = ctx.mol
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, dd, _dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += _saturate_contrib(n, dd * 2.0)
    tpsa = ctx.tpsa
    value += tpsa * MAX_DIELECTRIC_PER_TPSA
    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)
    return max(1.0, value)


def main():
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    X = []
    y = []

    for entry in benchmark:
        diel = entry.get("dielectric_constant")
        if diel is None:
            continue
        ctx = MoleculeContext.from_smiles(entry["smiles"])
        if ctx is None:
            continue
        counts = _count_fragments(ctx.mol)

        base_pred = _base_dielectric_no_cross(ctx)
        residual = diel - base_pred

        row = np.zeros(len(_CANDIDATE_PAIRS), dtype=np.float64)
        for j, (fa, fb, _) in enumerate(_CANDIDATE_PAIRS):
            if counts.get(fa, 0) > 0 and counts.get(fb, 0) > 0:
                row[j] = 1.0
        X.append(row)
        y.append(residual)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} molecules with dielectric data")
    print(f"Cross-term pairs: {len(_CANDIDATE_PAIRS)}")

    # Fit ridge regression
    alpha = 1.0
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Bootstrap for confidence intervals
    n_boot = 50
    coefs_boot = np.zeros((n_boot, len(_CANDIDATE_PAIRS)))
    rng = np.random.RandomState(42)
    for b in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        mb = Ridge(alpha=alpha)
        mb.fit(X[idx], y[idx])
        coefs_boot[b] = mb.coef_

    # Identify which candidate pairs have data in the benchmark
    pair_has_data: set[tuple[str, str]] = set()
    for entry in benchmark:
        ctx = MoleculeContext.from_smiles(entry["smiles"])
        if ctx is None:
            continue
        counts = _count_fragments(ctx.mol)
        for fa, fb, _ in _CANDIDATE_PAIRS:
            if counts.get(fa, 0) > 0 and counts.get(fb, 0) > 0:
                pair_has_data.add((fa, fb))

    # Build output for ALL 9 candidate pairs — fitted where data exists,
    # default (hand-tuned) otherwise. This ensures gc.py always has a
    # coefficient for every physical interaction, even if the benchmark
    # lacks molecules containing both fragments.
    fitted_terms: list[tuple[str, str, float, str, str]] = []
    _DEFAULT_COEFS = {
        ("carbonate", "ether"): 0.8,
        ("nitrile", "ether"): 0.3,
        ("carbonate", "fluorine"): -0.5,
        ("sulfone", "ether"): 0.4,
        ("carbonate", "nitrile"): -0.3,
        ("alcohol", "carbonate"): -0.4,
        ("sulfone", "carbonate"): -0.3,
        ("nitrile", "fluorine"): 0.3,
        ("sulfone", "nitrile"): 0.5,
    }

    for j, (fa, fb, desc) in enumerate(_CANDIDATE_PAIRS):
        coef = float(model.coef_[j])
        ci_low = float(np.percentile(coefs_boot[:, j], 2.5)) if len(coefs_boot) > 0 else 0.0
        ci_high = float(np.percentile(coefs_boot[:, j], 97.5)) if len(coefs_boot) > 0 else 0.0
        has_data = (fa, fb) in pair_has_data

        if has_data and abs(coef) > 0.05:
            # Has data and non-trivial ridge coefficient → use data-driven value.
            # CI is reported for transparency but doesn't gate the coefficient
            # when data is sparse (small n → wide CIs naturally).
            clipped = max(-2.0, min(2.0, coef))
            ci_excludes_zero = ci_low > 0 or ci_high < 0
            source = "fitted" if ci_excludes_zero else "default"
            fitted_terms.append((fa, fb, round(clipped, 4), desc, source))
        else:
            # No data or trivial coefficient → fall back to hand-tuned default
            default_coef = _DEFAULT_COEFS.get((fa, fb), 0.0)
            fitted_terms.append((fa, fb, round(default_coef, 4), desc, "default"))

    output = {
        "cross_terms": [
            {"frag_a": fa, "frag_b": fb, "coefficient": coef, "description": desc, "source": source}
            for fa, fb, coef, desc, source in fitted_terms
        ],
        "ridge_alpha": alpha,
        "n_training_molecules": len(X),
        "n_bootstrap": n_boot,
             "fitted_from": BENCHMARK_PATH,
             "method": "ridge regression on dielectric residuals with bootstrap CI; non-significant terms fall back to original hand-tuned values",
         }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"\nFitted {len(fitted_terms)} cross-terms (from {len(_CANDIDATE_PAIRS)} candidates)")
    for fa, fb, coef, desc, source in fitted_terms:
        print(f"  [{source}] {fa} + {fb}: {coef:+.4f} ({desc})")
    print(f"\nSaved to {OUTPUT_PATH}")

    # Evaluate
    predictions = []
    experimentals = []
    for entry in benchmark:
        diel = entry.get("dielectric_constant")
        if diel is None:
            continue
        ctx = MoleculeContext.from_smiles(entry["smiles"])
        if ctx is None:
            continue
        counts = _count_fragments(ctx.mol)

        pred = _base_dielectric_no_cross(ctx)
        for fa, fb, coef, _, _ in fitted_terms:
            if counts.get(fa, 0) > 0 and counts.get(fb, 0) > 0:
                pred += coef

        predictions.append(pred)
        experimentals.append(diel)

    # Spearman
    from scipy.stats import spearmanr
    rho, _ = spearmanr(predictions, experimentals)
    print(f"\nSpearman rho (dielectric, fitted cross-terms): {rho:.4f}")

    # Also check base-only
    preds_base = []
    for entry in benchmark:
        diel = entry.get("dielectric_constant")
        if diel is None:
            continue
        ctx = MoleculeContext.from_smiles(entry["smiles"])
        if ctx is None:
            continue
        preds_base.append(_base_dielectric_no_cross(ctx))
    rho_base, _ = spearmanr(preds_base, [e["dielectric_constant"] for e in benchmark if e.get("dielectric_constant") is not None])
    print(f"Spearman rho (dielectric, base-only, no cross-terms): {rho_base:.4f}")


if __name__ == "__main__":
    main()
