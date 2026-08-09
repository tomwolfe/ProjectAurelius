#!/usr/bin/env python3
"""Leakage-aware orbital benchmark: report SEEN and UNSEEN molecules separately.

Why this exists
---------------
``benchmark_external_validation.py`` scores the orbital models over the whole
of ``external_property_benchmark.json``. 25 of the 68 parseable molecules in
that file also appear in ``orbital_calibration.json``, which is the set the
model constants were tuned on. Pooling them inflates the headline number:

    model   ALL      SEEN (n=27)   UNSEEN (n=45)
    TOM     0.204    0.542         0.170

A reported "rho = 0.5" for TOM was therefore largely a measurement of recall
on the calibration set, not of predictive power. This script always splits
the two, so the honest number is the one that gets printed.

It also evaluates against ``experimental_ionization.json`` — 88 NIST gas-phase
ionisation energies — which is a stronger target than the DFT-derived orbital
labels for three reasons:

1. the values are experimentally measured, not computed;
2. they span 7.7-16.2 eV with 81 distinct values across 88 molecules, whereas
   the orbital labels give 17 distinct HOMO values across 72 molecules over a
   1.1 eV range — too coarse to rank meaningfully;
3. no molecule in it was used to tune the TOM constants.

Usage:
    python benchmarks/benchmark_orbital_leakage.py [--json out.json]
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

DATA_DIR = os.path.join(PROJECT_ROOT, "src", "aurelius", "data")

from aurelius.scoring.oracle.lone_pair import (  # noqa: E402
    predict_ionization_energy,
    predict_lone_pair_homo,
)
from aurelius.scoring.oracle.quantum import predict_tom_orbitals_batch  # noqa: E402


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _load(name: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, name)) as fh:
        return json.load(fh)


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(ref)),
        "spearman_rho": float(spearmanr(pred, ref).statistic),
        "mae_eV": float(np.abs(pred - ref).mean()),
    }


def orbital_label_benchmark() -> dict:
    """Score TOM and LPM on the DFT orbital labels, split by calibration overlap."""
    calibration = {_canonical(e["smiles"]) for e in _load("orbital_calibration.json")}
    rows = []
    for entry in _load("external_property_benchmark.json"):
        if entry.get("homo_eV") is None:
            continue
        canonical = _canonical(entry["smiles"])
        if canonical is None:
            continue
        rows.append((canonical, entry["homo_eV"], canonical in calibration))

    mols = [Chem.MolFromSmiles(s) for s, _, _ in rows]
    ref = np.array([h for _, h, _ in rows])
    seen = np.array([s for _, _, s in rows])

    tom_homo, _ = predict_tom_orbitals_batch(mols)
    # Condensed-phase scale: the DFT labels are condensed-phase, so compare
    # like with like. The map is monotone, so rho is unchanged either way.
    lpm_homo = np.array([predict_lone_pair_homo(m) for m in mols])

    out: dict = {}
    for label, mask in (("all", np.ones(len(ref), bool)), ("seen", seen), ("unseen", ~seen)):
        if mask.sum() < 5:
            continue
        out[label] = {
            "tom": _metrics(tom_homo[mask], ref[mask]),
            "lpm": _metrics(lpm_homo[mask], ref[mask]),
        }
    return out


def lumo_label_benchmark() -> dict:
    """Score raw and Δ-corrected TOM LUMO, split by calibration overlap.

    LUMO was previously absent from this report, so the pooled "rho ~ 0.5"
    figure went unchallenged. Split honestly it is 0.94 on molecules whose
    labels are duplicated between the two files and 0.06 on everything else.

    Note the ranking numbers here are not trustworthy in either direction:
    ``audit_label_confound.py`` shows 69% of the unseen LUMO variance is
    between-source, and a citation-only predictor scores rho = 0.84. MAE is
    the defensible metric for this target (ADR-2026-08-08-07).
    """
    from aurelius.scoring.oracle.delta_correction import get_delta_correction

    calibration = {_canonical(e["smiles"]) for e in _load("orbital_calibration.json")}
    rows = []
    for entry in _load("external_property_benchmark.json"):
        if entry.get("lumo_eV") is None:
            continue
        canonical = _canonical(entry["smiles"])
        if canonical is None:
            continue
        rows.append((canonical, entry["lumo_eV"], canonical in calibration))

    mols = [Chem.MolFromSmiles(s) for s, _, _ in rows]
    ref = np.array([v for _, v, _ in rows])
    seen = np.array([s for _, _, s in rows])

    _, tom_lumo = predict_tom_orbitals_batch(mols)
    delta = get_delta_correction()
    corrected = np.array([delta.predict_corrected(m)[1] for m in mols])

    out: dict = {}
    for label, mask in (("all", np.ones(len(ref), bool)), ("seen", seen), ("unseen", ~seen)):
        if mask.sum() < 5:
            continue
        out[label] = {
            "tom": _metrics(tom_lumo[mask], ref[mask]),
            "delta": _metrics(corrected[mask], ref[mask]),
        }
    return out


def experimental_ip_benchmark() -> dict:
    """Score both models against experimental gas-phase ionisation energies."""
    entries = _load("experimental_ionization.json")
    mols, ref = [], []
    for entry in entries:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        ref.append(entry["ip_eV"])
    ref_arr = np.array(ref)

    start = time.perf_counter()
    lpm = np.array([predict_ionization_energy(m)[0] for m in mols])
    lpm_seconds = time.perf_counter() - start

    start = time.perf_counter()
    tom_homo, _ = predict_tom_orbitals_batch(mols)
    tom_seconds = time.perf_counter() - start

    return {
        "lpm": {**_metrics(lpm, ref_arr), "seconds": round(lpm_seconds, 4)},
        "tom": {**_metrics(-tom_homo, ref_arr), "seconds": round(tom_seconds, 4)},
        "label_span_eV": round(float(ref_arr.max() - ref_arr.min()), 2),
        "distinct_values": int(len(set(ref))),
    }


def _print_pair(title: str, block: dict, second: str = "lpm") -> None:
    name = second.upper()
    print(f"\n{title}")
    print(f"  {'split':10s} {'n':>4s}  {'TOM rho':>9s} {'TOM MAE':>9s}  "
          f"{name + ' rho':>9s} {name + ' MAE':>9s}")
    for split, models in block.items():
        tom, other = models["tom"], models[second]
        print(f"  {split:10s} {tom['n']:4d}  {tom['spearman_rho']:+9.3f} {tom['mae_eV']:9.3f}"
              f"  {other['spearman_rho']:+9.3f} {other['mae_eV']:9.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write results to this path")
    args = parser.parse_args()

    print("=" * 74)
    print("ORBITAL BENCHMARK (leakage-aware)")
    print("=" * 74)

    labels = orbital_label_benchmark()
    _print_pair("DFT orbital labels (external_property_benchmark.json)", labels)
    if "seen" in labels and "unseen" in labels:
        gap = labels["seen"]["tom"]["spearman_rho"] - labels["unseen"]["tom"]["spearman_rho"]
        print(f"\n  TOM leakage gap (seen - unseen): {gap:+.3f}")
        print("  Any headline computed over 'all' is inflated by this gap.")

    lumo = lumo_label_benchmark()
    _print_pair("DFT LUMO labels (raw TOM vs delta-corrected)", lumo, second="delta")
    if "seen" in lumo and "unseen" in lumo:
        print(
            f"\n  Delta-corrected LUMO: seen rho "
            f"{lumo['seen']['delta']['spearman_rho']:+.3f} vs unseen "
            f"{lumo['unseen']['delta']['spearman_rho']:+.3f}"
        )
        print("  26/27 'seen' molecules have byte-identical labels in both files,")
        print("  so the seen figure is recall of duplicated numbers, not skill.")
        print("  Unseen LUMO rho is provenance-confounded (citation-only rho = 0.84);")
        print("  see audit_label_confound.py. Use MAE, not rho, for this target.")

    ips = experimental_ip_benchmark()
    print("\nExperimental gas-phase ionisation energies (NIST, no leakage)")
    print(f"  n = {ips['lpm']['n']}, span {ips['label_span_eV']} eV, "
          f"{ips['distinct_values']} distinct values")
    print(f"  {'model':6s} {'rho':>9s} {'MAE (eV)':>10s} {'seconds':>9s}")
    for name in ("tom", "lpm"):
        m = ips[name]
        print(f"  {name.upper():6s} {m['spearman_rho']:+9.3f} {m['mae_eV']:10.3f} {m['seconds']:9.4f}")

    results = {"orbital_labels": labels, "lumo_labels": lumo, "experimental_ip": ips}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
            fh.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
