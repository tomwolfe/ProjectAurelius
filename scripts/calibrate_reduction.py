#!/usr/bin/env python3
"""Fit the affine map from raw xTB ΔSCF EA onto the experimental EA scale.

An affine map cannot change Spearman ρ, so this sets *units* only — it is not
a ranking fit. It exists so that ``ea_eV`` is reported on a scale a chemist can
compare against a literature electron affinity, rather than on the raw xTB
ΔSCF scale which is offset by several eV.

Prints the coefficients to paste into ``reduction.py`` (_EA_CALIBRATION) along
with the raw span that defines ``_EA_CALIBRATED_SPAN_RAW``.

Usage::

    python scripts/calibrate_reduction.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from aurelius.scoring.oracle.reduction import (  # noqa: E402
    compute_dscf_ea,
    has_xtb,
    load_experimental_ea,
)


def main() -> None:
    if not has_xtb():
        print("xTB not available — cannot calibrate.")
        return

    entries = load_experimental_ea()
    raws, labels, names = [], [], []
    for e in entries:
        mol = Chem.MolFromSmiles(e["smiles"])
        raw = compute_dscf_ea(mol)
        if raw is None:
            print(f"  [fail] {e['name']}")
            continue
        raws.append(raw)
        labels.append(float(e["ea_eV"]))
        names.append(e["name"])

    x = np.asarray(raws)
    y = np.asarray(labels)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept

    print(f"n = {len(x)}")
    print(f"Spearman rho = {spearmanr(x, y).statistic:+.4f}  (affine-invariant)")
    print(f"MAE  = {np.mean(np.abs(pred - y)):.4f} eV")
    print(f"RMSE = {np.sqrt(np.mean((pred - y) ** 2)):.4f} eV")
    print(f"\n_EA_CALIBRATION: tuple[float, float] = ({slope:.4f}, {intercept:.4f})")
    print(f"_EA_CALIBRATED_SPAN_RAW: tuple[float, float] = ({x.min():.2f}, {x.max():.2f})")

    print("\nworst residuals:")
    order = np.argsort(-np.abs(pred - y))
    for i in order[:6]:
        print(f"  {names[i]:32s} exp={y[i]:+.2f} pred={pred[i]:+.2f} "
              f"err={pred[i] - y[i]:+.2f}")


if __name__ == "__main__":
    main()
