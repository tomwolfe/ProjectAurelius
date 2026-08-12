#!/usr/bin/env python3
"""Fit the affine maps for gas-phase and solution-phase ΔSCF EA calibration.

An affine map cannot change Spearman ρ, so these set *units* only — they are
not ranking fits. They exist so that ``ea_eV`` is reported on a scale a chemist
can compare against a literature electron affinity, rather than on the raw xTB
ΔSCF scale which is offset by several eV.

Default mode calibrates the gas-phase map against 40 experimental gas-phase EAs.
``--solution`` mode calibrates the solution-phase map against 10 CV onsets,
using the Born-correction approach (ADR-2026-08-11-05):

    EA_solution = calibrate_ea(raw_gas_ea) + born_correction(epsilon, r_cavity)

followed by an affine map that absorbs the residual systematic offset.

Usage::

    python scripts/calibrate_reduction.py
    python scripts/calibrate_reduction.py --solution
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from aurelius.scoring.oracle.reduction import (  # noqa: E402
    _born_solvation_correction,
    _cavity_radius,
    _SOLUTION_PHASE_EPSILON,
    compute_dscf_ea,
    calibrate_ea,
    has_xtb,
    load_experimental_ea_gas,
    load_experimental_ea_solution,
)


def calibrate_gas_phase() -> None:
    """Fit _EA_CALIBRATION against gas-phase experimental EAs (default)."""
    entries = load_experimental_ea_gas()
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


def calibrate_solution_phase() -> None:
    """Fit _EA_SOLUTION_CALIBRATION against CV onsets using Born correction.

    Computes gas-phase ΔSCF EA + Born solvation correction for each of the
    10 solution-phase entries, then fits an affine map onto the experimental
    CV onsets. The Born correction is fixed (physical); only the affine
    map is calibrated.
    """
    entries = load_experimental_ea_solution()
    if not entries:
        print("No solution-phase entries available.")
        return

    epsilon = _SOLUTION_PHASE_EPSILON
    gas_raw, gas_cal, born, labels, names = [], [], [], [], []
    for e in entries:
        mol = Chem.MolFromSmiles(e["smiles"])
        raw = compute_dscf_ea(mol)
        if raw is None:
            print(f"  [fail] {e['name']}")
            continue
        r = _cavity_radius(mol)
        b = _born_solvation_correction(epsilon, r)
        gc = calibrate_ea(raw)
        gas_raw.append(raw)
        gas_cal.append(gc)
        born.append(b)
        labels.append(float(e["ea_eV"]))
        names.append(e["name"])

    gas_raw = np.asarray(gas_raw)
    gas_cal = np.asarray(gas_cal)
    born = np.asarray(born)
    y = np.asarray(labels)
    x = gas_cal + born  # gas EA + Born correction

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept

    print("Solution-phase calibration (Born-corrected gas ΔSCF EA)")
    print(f"n = {len(x)}, ε = {epsilon}")
    print(f"Spearman rho (gas+Born vs exp) = {spearmanr(x, y).statistic:+.4f}")
    print(f"Spearman rho (gas-cal only)    = {spearmanr(gas_cal, y).statistic:+.4f}")
    print(f"Spearman rho (gas-raw only)    = {spearmanr(gas_raw, y).statistic:+.4f}")
    print(f"MAE  = {np.mean(np.abs(pred - y)):.4f} eV")
    print(f"RMSE = {np.sqrt(np.mean((pred - y) ** 2)):.4f} eV")
    print(f"\n_EA_SOLUTION_CALIBRATION: tuple[float, float] = ({slope:.4f}, {intercept:.4f})")
    print(f"_EA_SOLUTION_CALIBRATED_SPAN_RAW: tuple[float, float] = "
          f"({gas_raw.min():.2f}, {gas_raw.max():.2f})")

    print("\nworst residuals:")
    order = np.argsort(-np.abs(pred - y))
    for i in order[:6]:
        print(f"  {names[i]:32s} exp={y[i]:+.2f} pred={pred[i]:+.2f} "
              f"gas={gas_cal[i]:+.2f} born={born[i]:+.2f} "
              f"err={pred[i] - y[i]:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", action="store_true",
                        help="Calibrate solution-phase map (Born correction + CV onsets)")
    args = parser.parse_args()

    if not has_xtb():
        print("xTB not available — cannot calibrate.")
        return

    if args.solution:
        calibrate_solution_phase()
    else:
        calibrate_gas_phase()


if __name__ == "__main__":
    main()
