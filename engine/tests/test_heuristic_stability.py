"""Heuristic stability test: GC prediction rank correlation vs published data.

Ensures the GC fragment-additivity models maintain minimum rank correlation
thresholds for dielectric and viscosity. If this test fails, the GC
parameters have drifted from their calibration.

This is a lightweight, fast test — it runs only GC predictions on the
benchmark data, not the full quantum oracle.

Thresholds:
  - Dielectric: Spearman ρ > 0.6
  - Viscosity:  Spearman ρ > 0.5
"""

from __future__ import annotations

import json
import os
from typing import Any

from aurelius.scoring.oracle.gc import ElectrolytePack
from aurelius.types import MoleculeContext

BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "aurelius",
    "data",
    "external_property_benchmark.json",
)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 4:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(vals)
        return [
            sum(1 + i for i, sv in enumerate(sorted_vals) if sv == v)
            / max(sum(1 for sv in sorted_vals if sv == v), 1)
            for v in vals
        ]

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _load_benchmark() -> list[dict[str, Any]]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def _collect_gc_predictions() -> dict[str, dict[str, Any]]:
    """Collect GC-only predictions vs experimental values for all benchmark entries."""
    benchmark = _load_benchmark()
    results: dict[str, dict[str, Any]] = {
        "dielectric_constant": {"predicted": [], "experimental": [], "names": []},
        "viscosity_cP": {"predicted": [], "experimental": [], "names": []},
    }
    pack = ElectrolytePack()

    for entry in benchmark:
        smiles = entry["smiles"]
        name = entry["name"]
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            continue

        try:
            props = pack.predict_all(ctx)
        except Exception:
            continue

        diel_exp = entry.get("dielectric_constant")
        visc_exp = entry.get("viscosity_cP")
        diel_pred = props.get("dielectric_proxy")
        visc_pred = props.get("viscosity_proxy")

        if diel_exp is not None and diel_pred is not None:
            results["dielectric_constant"]["predicted"].append(diel_pred)
            results["dielectric_constant"]["experimental"].append(diel_exp)
            results["dielectric_constant"]["names"].append(name)

        if visc_exp is not None and visc_pred is not None:
            results["viscosity_cP"]["predicted"].append(visc_pred)
            results["viscosity_cP"]["experimental"].append(visc_exp)
            results["viscosity_cP"]["names"].append(name)

    return results


def test_gc_dielectric_rank_correlation() -> None:
    """GC dielectric proxy must maintain Spearman ρ > 0.6 against benchmark."""
    data = _collect_gc_predictions()["dielectric_constant"]
    n = len(data["predicted"])
    assert n >= 10, f"Only {n} molecules with dielectric data (need >= 10)"
    rho = _spearman_rho(data["predicted"], data["experimental"])
    assert rho > 0.6, (
        f"GC Dielectric ρ={rho:.4f} < 0.6 (n={n}). "
        "GC fragment-additivity dielectric predictions have drifted from "
        "their calibration. Run `python scripts/update_benchmark_docs.py` "
        "to reassess."
    )


def test_gc_viscosity_rank_correlation() -> None:
    """GC viscosity proxy must maintain Spearman ρ > 0.5 against benchmark."""
    data = _collect_gc_predictions()["viscosity_cP"]
    n = len(data["predicted"])
    assert n >= 10, f"Only {n} molecules with viscosity data (need >= 10)"
    rho = _spearman_rho(data["predicted"], data["experimental"])
    assert rho > 0.5, (
        f"GC Viscosity ρ={rho:.4f} < 0.5 (n={n}). "
        "GC fragment-additivity viscosity predictions have drifted from "
        "their calibration. Run `python scripts/update_benchmark_docs.py` "
        "to reassess."
    )
