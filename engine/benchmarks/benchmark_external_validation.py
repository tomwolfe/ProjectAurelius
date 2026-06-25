#!/usr/bin/env python3
"""Benchmark: External Property Validation — rank correlation vs published data.

Loads external_property_benchmark.json (published experimental values for
common electrolyte solvents) and measures Spearman rank correlation between
Aurelius oracle predictions and experimental values for each property.

Usage:
    python -m benchmarks.benchmark_external_validation
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger("aurelius").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.pipeline import AureliusPipeline  # noqa: E402
from aurelius.types import MoleculeContext  # noqa: E402


def _load_benchmark() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "external_property_benchmark.json"
    with open(path) as f:
        return json.load(f)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 4:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(vals)
        ranks = [sorted_vals.index(v) + 1 for v in vals]
        tie_corrected = []
        for i, v in enumerate(vals):
            tied = sum(1 for sv in sorted_vals if sv == v)
            tie_corrected.append(sum(r for r, sv in zip(ranks, vals, strict=False) if sv == v) / tied if tied > 0 else ranks[i])
        return tie_corrected

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def main() -> None:
    print("=" * 72)
    print("  EXTERNAL PROPERTY VALIDATION — Spearman Rank Correlation")
    print("=" * 72)
    print()
    print("  Compares Aurelius oracle predictions against published experimental")
    print("  values for common electrolyte solvents.")
    print()

    benchmark = _load_benchmark()
    print(f"  Benchmark molecules: {len(benchmark)}")
    print()

    pipeline = AureliusPipeline()
    pipeline.initialize()

    results: dict[str, dict[str, list[float]]] = {
        "dielectric_constant": {"predicted": [], "experimental": [], "names": []},
        "viscosity_cP": {"predicted": [], "experimental": [], "names": []},
        "donor_number": {"predicted": [], "experimental": [], "names": []},
        "homo_eV": {"predicted": [], "experimental": [], "names": []},
        "lumo_eV": {"predicted": [], "experimental": [], "names": []},
    }

    property_map = {
        "dielectric_constant": "dielectric_proxy",
        "viscosity_cP": "viscosity_proxy",
        "donor_number": "li_solvation_proxy",
        "homo_eV": "homo_eV",
        "lumo_eV": "lumo_eV",
    }

    skipped = 0
    for entry in benchmark:
        smiles = entry["smiles"]
        name = entry["name"]
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            print(f"  SKIP {name}: invalid SMILES")
            skipped += 1
            continue

        try:
            oracle_result = pipeline.screen_molecule(ctx)
        except Exception as e:
            print(f"  SKIP {name}: oracle error — {e}")
            skipped += 1
            continue

        t2 = oracle_result.get("tier2")
        if t2 is None:
            skipped += 1
            continue

        for exp_key, pred_key in property_map.items():
            exp_val = entry.get(exp_key)
            if exp_val is None:
                continue
            pred_val = t2.get(pred_key)
            if pred_key == "li_solvation_proxy":
                pred_val = t2.get("li_solvation_proxy")
            if pred_val is None:
                continue
            results[exp_key]["predicted"].append(pred_val)
            results[exp_key]["experimental"].append(exp_val)
            results[exp_key]["names"].append(name)

    print(f"  Skipped: {skipped}")
    print()
    print("  ┌──────────────────────┬────────────┬──────────┬───────────┐")
    print("  │ Property             │     N      │    ρ     │   p-val   │")
    print("  ├──────────────────────┼────────────┼──────────┼───────────┤")

    for exp_key, label in [
        ("dielectric_constant", "Dielectric ε"),
        ("viscosity_cP", "Viscosity η"),
        ("donor_number", "Donor Number"),
        ("homo_eV", "HOMO"),
        ("lumo_eV", "LUMO"),
    ]:
        data = results[exp_key]
        n = len(data["predicted"])
        rho = 0.0
        p_val = 1.0
        if n >= 4:
            rho = _spearman_rho(data["predicted"], data["experimental"])
            t_stat = rho * math.sqrt((n - 2) / max(1 - rho * rho, 1e-10))
            try:
                from scipy.stats import t as t_dist
                p_val = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
            except ImportError:
                p_val = float("nan")

        sig = " ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else ""
        print(f"  │ {label:<20s} │ {n:4d}        │ {rho:+.4f}  │ {p_val:.4f}{sig} │")
        if n > 0 and rho > 0:
            high_name = data["names"][data["predicted"].index(max(data["predicted"]))]
            low_name = data["names"][data["predicted"].index(min(data["predicted"]))]
            exp_high = data["experimental"][data["experimental"].index(max(data["experimental"]))]
            exp_low = data["experimental"][data["experimental"].index(min(data["experimental"]))]
            print(f"  │   top: {high_name:<25s} pred={max(data['predicted']):.3f} exp={exp_high:.3f} │")
            print(f"  │   bot: {low_name:<25s} pred={min(data['predicted']):.3f} exp={exp_low:.3f} │")

    print("  └──────────────────────┴────────────┴──────────┴───────────┘")
    print()
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print()

    n_passing = sum(1 for k in results if len(results[k]["predicted"]) >= 4 and _spearman_rho(
        results[k]["predicted"], results[k]["experimental"]
    ) > 0)
    n_total = sum(1 for k in results if len(results[k]["predicted"]) >= 4)

    print(f"  Properties with ρ > 0: {n_passing}/{n_total}")
    print()

    all_pass = True
    for exp_key in results:
        data = results[exp_key]
        if len(data["predicted"]) >= 4:
            rho = _spearman_rho(data["predicted"], data["experimental"])
            if rho <= 0:
                print(f"  FAIL: {exp_key} ρ = {rho:.4f} (not above zero)")
                all_pass = False

    if all_pass:
        print("  EXTERNAL VALIDATION: ALL PROPERTIES SHOW POSITIVE RANK CORRELATION")
    else:
        print("  EXTERNAL VALIDATION: SOME PROPERTIES LACK POSITIVE RANK CORRELATION")

    print()
    print("__BENCHMARK_RESULTS__")
    print(json.dumps({
        exp_key: {
            "rho": round(_spearman_rho(data["predicted"], data["experimental"]), 4)
                if len(data["predicted"]) >= 4 else 0.0,
            "n": len(data["predicted"]),
        }
        for exp_key, data in results.items()
    }, indent=2))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
