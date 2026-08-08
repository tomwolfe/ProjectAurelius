#!/usr/bin/env python3
"""Calibrate the Lone-Pair Orbital Model against experimental ionisation energies.

Fits the LPM parameter vector by physics-anchored ridge regression on
``src/aurelius/data/experimental_ionization.json`` (88 NIST gas-phase IPs)
and writes ``src/aurelius/data/lone_pair_params.json``.

Two properties make this fit trustworthy rather than a curve-fitting exercise:

1. **The target is experimental.** Gas-phase ionisation energies are directly
   measured (photoelectron spectroscopy / photoionisation), span 7.7-16.2 eV,
   and have 81 distinct values across 88 molecules. The previous orbital
   labels were 17 distinct values across 72 molecules with a 1.1 eV span,
   which cannot support a meaningful rank metric.

2. **The prior is literature chemistry.** Ridge shrinkage pulls each orbital
   class intercept toward its Hinze-Jaffe valence-state ionisation energy,
   not toward zero. A class with one supporting molecule therefore stays near
   its atomic value instead of taking an arbitrary fitted value. Removing
   this prior raises worst-case leave-one-out error from 2.4 eV to 11.9 eV.

The latent-orbital assignment (which lone pair is the HOMO) is unobserved, so
it is solved by iterative reassignment: fit weights, reassign each molecule to
its lowest-IP candidate orbital, repeat to convergence.

Usage:
    python scripts/calibrate_lone_pair.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.scoring.oracle.lone_pair import (  # noqa: E402
    FEATURE_NAMES,
    VOIE_PRIOR,
    clear_params_cache,
    orbital_candidates,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "data")
IP_PATH = os.path.join(DATA_DIR, "experimental_ionization.json")
PARAMS_PATH = os.path.join(DATA_DIR, "lone_pair_params.json")

RHO_ALK_GRID = (0.40, 0.55, 0.70, 0.80)
RHO_IND_GRID = (0.25, 0.35, 0.50, 0.65)
LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
MAX_ITERS = 30


def load_dataset() -> tuple[list[str], list[Chem.Mol], np.ndarray]:
    with open(IP_PATH) as fh:
        raw = json.load(fh)
    names, mols, ips = [], [], []
    for entry in raw:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            print(f"  WARNING: unparseable SMILES skipped: {entry['name']}")
            continue
        names.append(entry["name"])
        mols.append(mol)
        ips.append(entry["ip_eV"])
    return names, mols, np.asarray(ips, dtype=float)


def design_matrices(
    mols: list[Chem.Mol], rho_alk: float, rho_ind: float
) -> list[np.ndarray]:
    """One matrix per molecule: rows are candidate orbitals, columns features."""
    out = []
    for mol in mols:
        rows = [
            [feats[name] for name in FEATURE_NAMES]
            for _, feats in orbital_candidates(mol, rho_alk=rho_alk, rho_ind=rho_ind)
        ]
        out.append(np.asarray(rows, dtype=float))
    return out


def _prior_vector() -> np.ndarray:
    return np.asarray([VOIE_PRIOR[name] for name in FEATURE_NAMES], dtype=float)


def fit_weights(mats: list[np.ndarray], y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge fit with latent orbital assignment, shrunk toward the VOIE prior."""
    w0 = _prior_vector()
    eye = np.eye(len(FEATURE_NAMES))
    selection = [0] * len(mats)
    weights = w0
    for _ in range(MAX_ITERS):
        design = np.asarray([mats[i][selection[i]] for i in range(len(mats))])
        weights = np.linalg.solve(design.T @ design + lam * eye, design.T @ y + lam * w0)
        updated = [int(np.argmin(mats[i] @ weights)) for i in range(len(mats))]
        if updated == selection:
            break
        selection = updated
    design = np.asarray([mats[i][selection[i]] for i in range(len(mats))])
    return np.linalg.solve(design.T @ design + lam * eye, design.T @ y + lam * w0)


def predict(mats: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.asarray([float((m @ weights).min()) for m in mats])


def leave_one_out(mats: list[np.ndarray], y: np.ndarray, lam: float) -> np.ndarray:
    preds = np.empty(len(mats))
    for j in range(len(mats)):
        train = [mats[i] for i in range(len(mats)) if i != j]
        weights = fit_weights(train, np.delete(y, j), lam)
        preds[j] = float((mats[j] @ weights).min())
    return preds


def score(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = preds - y
    return {
        "spearman_rho": float(spearmanr(preds, y).statistic),
        "mae_eV": float(np.abs(err).mean()),
        "rmse_eV": float(np.sqrt((err**2).mean())),
        "max_abs_error_eV": float(np.abs(err).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="do not write params")
    args = parser.parse_args()

    names, mols, y = load_dataset()
    print(f"Loaded {len(y)} experimental ionisation energies "
          f"({y.min():.2f}-{y.max():.2f} eV, {len(set(y))} distinct)")

    best: tuple[float, float, float, float, dict[str, float], np.ndarray] | None = None
    print("\nGrid search (leave-one-out CV):")
    print(f"  {'rho_alk':>8} {'rho_ind':>8} {'lambda':>7} {'rho':>8} {'MAE':>7} {'maxerr':>7}")
    for rho_alk in RHO_ALK_GRID:
        for rho_ind in RHO_IND_GRID:
            # Attenuation factors enter feature construction, so re-derive.
            mats = design_matrices(mols, rho_alk, rho_ind)
            for lam in LAMBDA_GRID:
                metrics = score(leave_one_out(mats, y, lam), y)
                # Penalise worst-case error: a model used for ranking must not
                # produce wild outliers on unseen orbital classes.
                objective = metrics["spearman_rho"] - 0.02 * metrics["max_abs_error_eV"]
                print(f"  {rho_alk:8.2f} {rho_ind:8.2f} {lam:7.2f} "
                      f"{metrics['spearman_rho']:8.4f} {metrics['mae_eV']:7.3f} "
                      f"{metrics['max_abs_error_eV']:7.2f}")
                if best is None or objective > best[0]:
                    best = (objective, rho_alk, rho_ind, lam, metrics,
                            fit_weights(mats, y, lam))

    assert best is not None
    _, rho_alk, rho_ind, lam, metrics, weights = best
    print(f"\nBEST: rho_alk={rho_alk} rho_ind={rho_ind} lambda={lam}")
    print(f"  Spearman rho = {metrics['spearman_rho']:+.4f}")
    print(f"  MAE          = {metrics['mae_eV']:.3f} eV")
    print(f"  RMSE         = {metrics['rmse_eV']:.3f} eV")
    print(f"  max |error|  = {metrics['max_abs_error_eV']:.2f} eV")

    print("\nFitted parameters (VOIE prior in parentheses):")
    for name, value in zip(FEATURE_NAMES, weights, strict=True):
        print(f"  {name:12s} {value:+8.3f}   ({VOIE_PRIOR[name]:+.2f})")

    if args.dry_run:
        print("\n--dry-run: parameters not written")
        return 0

    _write_params(rho_alk, rho_ind, weights, metrics=metrics, lam=lam)
    clear_params_cache()
    print(f"\nWrote {PARAMS_PATH}")
    return 0


def _write_params(
    rho_alk: float,
    rho_ind: float,
    weights: np.ndarray,
    metrics: dict[str, float],
    lam: float,
) -> None:
    payload = {
        "_comment": (
            "Lone-Pair Orbital Model parameters. Fitted by "
            "scripts/calibrate_lone_pair.py against 88 NIST experimental "
            "gas-phase ionisation energies with a Hinze-Jaffe VOIE ridge "
            "prior. Do not hand-edit; re-run the script."
        ),
        "rho_alk": rho_alk,
        "rho_ind": rho_ind,
        "ridge_lambda": lam,
        "weights": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, weights, strict=True)},
        "validation": {k: round(v, 4) for k, v in metrics.items()},
    }
    tmp = PARAMS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, PARAMS_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
