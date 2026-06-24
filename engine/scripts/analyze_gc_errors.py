#!/usr/bin/env python3
"""Analyze GC fragment error patterns to suggest new fragment parameters.

Loads external_property_benchmark.json, predicts dielectric/viscosity for all
entries using the current GC model, identifies high-error molecules (>20%
absolute error), performs BRICS decomposition on those molecules, and reports
recurring substructures missing from the current _GC_FRAGMENTS list.

Usage:
    python scripts/analyze_gc_errors.py

Output:
    Table of top 10 candidate fragments with frequency and mean residual error.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.scoring.oracle.gc import (
    _GC_FRAGMENTS,
    ElectrolytePack,
)
from aurelius.types import MoleculeContext


def _load_benchmark(path: str | None = None) -> list[dict]:
    if path is not None:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "external_property_benchmark.json"
    with open(p) as f:
        return json.load(f)


def _has_dielectric_or_viscosity(entry: dict) -> bool:
    return entry.get("dielectric_constant") is not None or entry.get("viscosity_cP") is not None


def _relative_error(predicted: float, experimental: float) -> float:
    if experimental == 0.0:
        return 0.0
    return abs(predicted - experimental) / abs(experimental)


def _get_existing_fragment_smarts() -> set[str]:
    existing: set[str] = set()
    for pattern, _name, *_rest in _GC_FRAGMENTS:
        if pattern is not None:
            existing.add(Chem.MolToSmarts(pattern))
    return existing


def _get_fragment_name_map() -> dict[str, str]:
    return {Chem.MolToSmarts(p): n for p, n, *_ in _GC_FRAGMENTS if p is not None}


def _brics_decompose(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        fragments = BRICS.BRICSDecompose(mol)
        return list(fragments)
    except Exception:
        return []


def _canonicalize_fragment_smiles(frag_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main() -> None:
    data = _load_benchmark()
    pack = ElectrolytePack()

    existing_smarts = _get_existing_fragment_smarts()

    high_error_entries: list[dict] = []
    for entry in data:
        if not _has_dielectric_or_viscosity(entry):
            continue
        smi = entry["smiles"]
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue

        props = pack.predict_all(ctx)

        has_high_error = False
        residual_sum = 0.0
        n_props = 0

        if entry.get("dielectric_constant") is not None:
            exp = entry["dielectric_constant"]
            pred = props.get("dielectric_proxy", 0.0)
            err = _relative_error(pred, exp)
            if err > 0.20:
                has_high_error = True
            residual_sum += pred - exp
            n_props += 1

        if entry.get("viscosity_cP") is not None:
            exp = entry["viscosity_cP"]
            pred = props.get("viscosity_proxy", 0.0)
            err = _relative_error(pred, exp)
            if err > 0.20:
                has_high_error = True
            residual_sum += pred - exp
            n_props += 1

        if has_high_error:
            avg_residual = residual_sum / max(n_props, 1)
            high_error_entries.append({
                "smiles": smi,
                "avg_residual": avg_residual,
                "dielectric_exp": entry.get("dielectric_constant"),
                "dielectric_pred": props.get("dielectric_proxy"),
                "viscosity_exp": entry.get("viscosity_cP"),
                "viscosity_pred": props.get("viscosity_proxy"),
            })

    fragment_counter: Counter = Counter()
    fragment_residuals: dict[str, list[float]] = {}

    for entry in high_error_entries:
        fragments = _brics_decompose(entry["smiles"])
        for frag in fragments:
            canon = _canonicalize_fragment_smiles(frag)
            if canon is None:
                continue
            frag_mol = Chem.MolFromSmiles(canon)
            if frag_mol is None:
                continue
            frag_smarts = Chem.MolToSmarts(frag_mol)
            if frag_smarts in existing_smarts:
                continue
            fragment_counter[canon] += 1
            if canon not in fragment_residuals:
                fragment_residuals[canon] = []
            fragment_residuals[canon].append(entry["avg_residual"])

    print("=" * 80)
    print("  GC Fragment Error Analysis — Top 10 Missing Fragment Candidates")
    print("=" * 80)
    print(f"\nAnalyzed {len(data)} benchmark entries.")
    print(f"High-error molecules (>20% relative error): {len(high_error_entries)}")
    print(f"Candidate fragments not in _GC_FRAGMENTS: {len(fragment_counter)}")
    print()
    print(f"{'Rank':<6} {'Fragment SMILES':<45} {'Count':<8} {'Avg Residual':<14} {'Freq in Errors':<16}")
    print("-" * 80)

    sorted_frags = fragment_counter.most_common(10)
    for rank, (frag_smiles, count) in enumerate(sorted_frags, 1):
        residuals = fragment_residuals.get(frag_smiles, [0.0])
        avg_res = sum(residuals) / len(residuals)
        freq_in_errors = count / max(len(high_error_entries), 1) * 100
        display_smi = frag_smiles if len(frag_smiles) <= 42 else frag_smiles[:39] + "..."
        print(
            f"{rank:<6} {display_smi:<45} {count:<8} {avg_res:+7.4f}      {freq_in_errors:>5.1f}%"
        )

    print()
    print("Legend:")
    print("  Count         = number of high-error molecules containing this fragment")
    print("  Avg Residual  = mean (predicted - experimental) across occurrences")
    print("  Freq in Errors = percentage of high-error molecules with this fragment")
    print()
    print("Tip: A positive Avg Residual means the GC model over-predicts for")
    print("molecules with this fragment. A negative value means under-prediction.")
    print("High-frequency fragments are strong candidates for new GC parameters.")


if __name__ == "__main__":
    main()
