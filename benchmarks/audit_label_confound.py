#!/usr/bin/env python3
"""Detect benchmark targets whose labels are predictable from provenance alone.

Why this exists
---------------
``audit_benchmark_integrity.py`` checks whether individual entries are
internally consistent (does the SMILES parse, does the structure match the
name, does a value contradict a verified source). That is necessary but not
sufficient: a file can pass every per-entry check and still be unusable as a
*ranking* target, because the labels encode where they came from rather than
what the molecule is.

This is the failure that hid behind the reported LUMO accuracy. The 45
"unseen" LUMO labels in ``external_property_benchmark.json`` were compiled
from roughly a dozen papers, each using its own DFT functional and basis set.
Within a paper the values cluster tightly; between papers they are offset.
The consequence:

    a "model" given only the citation string -- no molecular structure at
    all -- scores Spearman rho = 0.84 on unseen LUMO.

No physics model beat rho = 0.09 on the same split. Any effort spent pushing
that number up would have been spent learning to infer the journal, not the
chemistry. The dataset cannot support a ranking claim regardless of the model.

The two diagnostics
-------------------
``citation_rho``
    Spearman correlation between the per-source *mean* label and the true
    label. This is the score achievable with zero chemistry. If it exceeds
    the best real model, the target is measuring provenance.

``between_source_fraction``
    Var(source means) / (Var(source means) + mean within-source Var). The
    share of total spread that lives between papers rather than between
    molecules. Above ~0.5 the target is dominated by methodology.

A clean target looks like ``experimental_ionization.json``: one consistent
measurement technique, so the citation carries no signal and this audit
reports no confound.

Usage:
    python benchmarks/audit_label_confound.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DATA_DIR = os.path.join(PROJECT_ROOT, "src", "aurelius", "data")

# A target is flagged when provenance alone out-ranks this. Set at the level
# a physics model would need to reach to be considered useful; if the
# citation string already clears it, the metric is not about chemistry.
CITATION_RHO_LIMIT = 0.50

# Share of variance that may sit between sources before the target is
# considered methodology-dominated rather than chemistry-dominated.
BETWEEN_SOURCE_LIMIT = 0.50

MIN_ENTRIES = 10
MIN_SOURCES = 3


def _grouped(
    entries: list[dict[str, Any]], target: str
) -> tuple[list[float], list[str]]:
    """Extract (label, source) pairs for entries that carry the target."""
    values: list[float] = []
    sources: list[str] = []
    for entry in entries:
        value = entry.get(target)
        if value is None:
            continue
        values.append(float(value))
        sources.append(str(entry.get("reference") or entry.get("source") or "unknown"))
    return values, sources


def analyse(
    entries: list[dict[str, Any]], target: str
) -> dict[str, Any] | None:
    """Quantify how much of a target's signal is explained by provenance."""
    values, sources = _grouped(entries, target)
    if len(values) < MIN_ENTRIES:
        return None

    by_source: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        by_source[source].append(value)

    arr = np.asarray(values)

    # A single-source target is the *ideal* case, not an unanalysable one:
    # provenance is constant so it cannot carry any signal. Report it
    # explicitly rather than skipping, so the clean control stays visible
    # next to the confounded targets.
    if len(by_source) < MIN_SOURCES:
        return {
            "target": target,
            "n": len(values),
            "n_sources": len(by_source),
            "label_std": float(arr.std()),
            "citation_rho": 0.0,
            "citation_mae": round(float(np.abs(arr - arr.mean()).mean()), 4),
            "within_source_var": round(float(np.var(arr)), 5),
            "between_source_var": 0.0,
            "between_source_fraction": 0.0,
            "confounded": False,
            "note": f"single measurement source ({len(by_source)}); "
                    "provenance carries no signal",
        }
    # Prediction using nothing but the citation the value was taken from.
    citation_pred = np.asarray([float(np.mean(by_source[s])) for s in sources])
    rho = spearmanr(citation_pred, arr).correlation
    citation_rho = float(rho) if np.isfinite(rho) else 0.0

    multi = [v for v in by_source.values() if len(v) > 1]
    within = float(np.mean([np.var(v) for v in multi])) if multi else 0.0
    between = float(np.var([np.mean(v) for v in multi])) if multi else 0.0
    denom = between + within
    between_fraction = float(between / denom) if denom > 0 else 0.0

    confounded = (
        citation_rho > CITATION_RHO_LIMIT or between_fraction > BETWEEN_SOURCE_LIMIT
    )
    return {
        "target": target,
        "n": len(values),
        "n_sources": len(by_source),
        "label_std": float(arr.std()),
        "citation_rho": round(citation_rho, 4),
        "citation_mae": round(float(np.abs(citation_pred - arr).mean()), 4),
        "within_source_var": round(within, 5),
        "between_source_var": round(between, 5),
        "between_source_fraction": round(between_fraction, 4),
        "confounded": bool(confounded),
    }


def audit_file(path: str, targets: tuple[str, ...]) -> list[dict[str, Any]]:
    """Run the confound analysis over every requested target in a file."""
    with open(path) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("solvents", [])
    results = []
    for target in targets:
        result = analyse(entries, target)
        if result is not None:
            result["file"] = os.path.basename(path)
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Write results to this path")
    args = parser.parse_args()

    checks = [
        (
            os.path.join(DATA_DIR, "external_property_benchmark.json"),
            ("homo_eV", "lumo_eV", "dielectric_constant", "viscosity_cP"),
        ),
        (os.path.join(DATA_DIR, "orbital_calibration.json"), ("homo_eV", "lumo_eV")),
        (os.path.join(DATA_DIR, "experimental_ionization.json"), ("ip_eV",)),
    ]

    results: list[dict[str, Any]] = []
    for path, targets in checks:
        if os.path.exists(path):
            results.extend(audit_file(path, targets))

    print("=" * 78)
    print("  LABEL-PROVENANCE CONFOUND AUDIT")
    print("=" * 78)
    print("  Can the label be predicted from the citation alone, with no chemistry?")
    print()
    print(
        f"  {'file':<34s} {'target':<20s} {'n':>4s} {'src':>4s} "
        f"{'cite rho':>9s} {'btwn':>6s}"
    )
    for r in results:
        if r["confounded"]:
            flag = "  <-- CONFOUNDED"
        elif r["n_sources"] < MIN_SOURCES:
            flag = "  <-- clean (single source)"
        else:
            flag = "  <-- clean"
        print(
            f"  {r['file'][:33]:<34s} {r['target']:<20s} {r['n']:>4d} "
            f"{r['n_sources']:>4d} {r['citation_rho']:>+9.3f} "
            f"{r['between_source_fraction']:>6.2f}{flag}"
        )

    confounded = [r for r in results if r["confounded"]]
    print()
    print("=" * 78)
    if confounded:
        print(f"  {len(confounded)} target(s) are provenance-confounded:")
        for r in confounded:
            print(
                f"    {r['file']} :: {r['target']} — citation-only rho "
                f"{r['citation_rho']:+.3f}, {r['between_source_fraction']:.0%} "
                f"of variance is between-source"
            )
        print()
        print("  A ranking claim on these targets measures which paper a value")
        print("  came from, not the chemistry. Report MAE, or move to a target")
        print("  with a single consistent measurement method.")
    else:
        print("  No provenance confound detected.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\n  Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
