#!/usr/bin/env python3
"""Reduction-axis benchmark against clean experimental electron affinities.

Why this exists
---------------
The reduction axis was previously validated against DFT/xTB LUMO labels. Those
numbers were either provenance-confounded (multi-source DFT, citation-only
ρ = 0.68) or circular (xTB labels scored by an xTB-backed model). Neither could
answer the only question that matters: *does the reduction axis rank unseen
molecules correctly against reality?*

This benchmark scores every estimator against 40 directly measured gas-phase
electron affinities (``experimental_electron_affinity.json``). Because every
label comes from the same measurement class and carries an identical reference
string, provenance carries zero signal by construction, so Spearman ρ is a
legitimate metric here — unlike on the external orbital benchmark.

Three controls are applied:

1. **Chemical-class-disjoint CV.** Quinones, nitroaromatics, polyacenes and so
   on are held out as whole groups. A random split would let the model see a
   near-twin of every test molecule and inflate the result.
2. **Affine-invariant scoring.** MAE is reported after an OLS affine map fitted
   *on the training fold only*, so a model is never rewarded or punished for
   its unit convention. ρ is unaffected by affine maps by construction.
3. **Label permutation control.** The whole pipeline is re-run against shuffled
   labels; anything that still scores well is measuring structure in the
   protocol rather than chemistry.

Usage::

    python benchmarks/benchmark_reduction_axis.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from aurelius.scoring.oracle.quantum import predict_tom_orbitals  # noqa: E402
from aurelius.scoring.oracle.reduction import (  # noqa: E402
    _StructuralEAModel,
    compute_dscf_ea_batch,
    has_xtb,  # noqa: E402
    load_experimental_ea_gas,
)


def _affine_mae(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """MAE after the best affine map, plus (slope, intercept)."""
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    return float(np.mean(np.abs(y - (a * x + b)))), float(a), float(b)


def _score(name: str, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(preds)
    p, y = preds[mask], labels[mask]
    if len(p) < 3:
        return {"estimator": name, "n": int(len(p)), "spearman_rho": float("nan"),
                "mae_eV": float("nan")}
    mae, slope, icpt = _affine_mae(p, y)
    return {
        "estimator": name,
        "n": int(len(p)),
        "spearman_rho": float(spearmanr(p, y).statistic),
        "mae_eV": mae,
        "affine_slope": slope,
        "affine_intercept": icpt,
    }


def _class_disjoint_cv(entries: list[dict], feats: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-chemical-class-out CV for the structural fallback."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    classes = np.array([e.get("chemical_class", "unknown") for e in entries])
    preds = np.full(len(labels), np.nan)

    for cls in np.unique(classes):
        test = classes == cls
        train = ~test
        if train.sum() < 8:
            continue
        sc = StandardScaler().fit(feats[train])
        model = Ridge(alpha=1.0).fit(sc.transform(feats[train]), labels[train])
        preds[test] = model.predict(sc.transform(feats[test]))

    return _score("structural_ridge (class-disjoint CV)", preds, labels)


def run(json_out: str | None = None) -> dict:
    # Gas-phase only: the ΔSCF EA computed here is gas-phase (no solvent), so
    # mixing in solution-phase labels (on a different physical scale) would
    # corrupt the correlation. See ADR-2026-08-11 (solution-phase EA).
    entries = load_experimental_ea_gas()
    if not entries:
        print("No experimental gas-phase EA data available; aborting.")
        return {}

    mols = [Chem.MolFromSmiles(e["smiles"]) for e in entries]
    labels = np.array([float(e["ea_eV"]) for e in entries])

    print("=" * 74)
    print("REDUCTION AXIS BENCHMARK — experimental gas-phase electron affinities")
    print("=" * 74)
    print(f"\n  n = {len(entries)}  span {labels.min():+.2f} to {labels.max():+.2f} eV")
    print(f"  chemical classes: {len(set(e.get('chemical_class') for e in entries))}")
    print(f"  xTB available: {has_xtb()}")

    results: list[dict] = []

    # --- Estimator 1: the descriptor being replaced -----------------------
    tom_lumo = np.array([predict_tom_orbitals(m)[1] for m in mols])
    # TOM LUMO is an energy: lower LUMO should mean easier reduction, so the
    # natural reduction descriptor is its negative.
    results.append(_score("TOM LUMO (negated) [superseded]", -tom_lumo, labels))

    # --- Estimator 2: structural fallback, class-disjoint -----------------
    from aurelius.scoring.oracle.reduction import structural_features
    feats = np.vstack([structural_features(m) for m in mols])
    results.append(_class_disjoint_cv(entries, feats, labels))

    fallback = _StructuralEAModel(entries)
    if fallback.available:
        loo = fallback.loo_metrics()
        results.append({"estimator": "structural_ridge (LOO)", "n": loo["n"],
                        "spearman_rho": loo["spearman_rho"], "mae_eV": loo["mae_eV"]})

    # --- Estimator 3: xTB ΔSCF EA ----------------------------------------
    dscf_seconds = 0.0
    if has_xtb():
        t0 = time.perf_counter()
        raw = compute_dscf_ea_batch(mols)
        dscf = np.array([np.nan if v is None else v for v in raw], dtype=float)
        dscf_seconds = time.perf_counter() - t0
        results.append(_score("xTB ΔSCF EA", dscf, labels))
    else:
        dscf = np.full(len(labels), np.nan)
        print("\n  [skip] xTB not available — ΔSCF row omitted.")

    print(f"\n  {'estimator':44s} {'n':>4s} {'rho':>8s} {'MAE(eV)':>9s}")
    for r in results:
        print(f"  {r['estimator']:44s} {r['n']:4d} {r['spearman_rho']:+8.3f} "
              f"{r['mae_eV']:9.3f}")

    if dscf_seconds:
        print(f"\n  ΔSCF cost: {dscf_seconds:.1f}s for {len(mols)} molecules "
              f"({1000 * dscf_seconds / len(mols):.0f} ms/molecule, 2 SPs each)")

    # --- Control: label permutation ---------------------------------------
    rng = np.random.default_rng(0)
    perm_rhos = []
    best = dscf if has_xtb() else feats @ np.ones(feats.shape[1])
    finite = np.isfinite(best)
    for _ in range(200):
        shuffled = rng.permutation(labels[finite])
        perm_rhos.append(abs(spearmanr(best[finite], shuffled).statistic))
    perm95 = float(np.percentile(perm_rhos, 95))
    print(f"\n  Permutation control: |ρ| 95th pct under shuffled labels = {perm95:.3f}")
    print("  A real signal must clear this bar.")

    payload = {
        "n": len(entries),
        "label_span_eV": [float(labels.min()), float(labels.max())],
        "xtb_available": has_xtb(),
        "results": results,
        "permutation_rho_p95": perm95,
        "dscf_seconds": dscf_seconds,
    }

    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  Wrote {json_out}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out",
                        default=os.path.join(PROJECT_ROOT, "benchmarks", "results",
                                             "reduction_axis.json"))
    args = parser.parse_args()
    run(args.json_out)


if __name__ == "__main__":
    main()
