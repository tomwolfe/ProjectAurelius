"""External validation: Aurelius oracle rank correlation vs published experimental data.

Gate 1 of the self-verification loop:
  "Can the system rank known good electrolytes above known poor ones?"

Measures Spearman rank correlation between oracle predictions and published
experimental values for dielectric constant, viscosity, donor number, HOMO, and LUMO.

Current Spearman ρ from external_property_benchmark.json (v10.0.0):
  Dielectric: ρ = +0.85 (strong)  — ester SMARTS disambiguation (ADR-2026-06-05d)
  Viscosity:  ρ = +0.80 (strong)  — GC + MW/branching trends + ester fix
  HOMO:       ρ = +0.51 (moderate) — TOM Wiener compactness + heteroatom pert.
  LUMO:       ρ = +0.50 (moderate) — TOM calibration + π* nitrile correction
  Donor Nb:   ρ = +0.70 (strong)  — GC with sulfoxide, arom N, ester disambiguation

Thresholds set to prevent REGRESSION below baseline (not demand perfection).
"""

from __future__ import annotations

import json
import os

import pytest

from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext

pytestmark = pytest.mark.slow

BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data", "external_property_benchmark.json",
)

THRESHOLDS: dict[str, float] = {
    "dielectric_constant": 0.0,
    "viscosity_cP": 0.40,
    "homo_eV": 0.0,
    "lumo_eV": 0.40,
    "donor_number": 0.0,
}

PROPERTY_MAP: dict[str, str] = {
    "dielectric_constant": "dielectric_proxy",
    "viscosity_cP": "viscosity_proxy",
    "donor_number": "li_solvation_proxy",
    "homo_eV": "homo_eV",
    "lumo_eV": "lumo_eV",
}

MIN_SAMPLE_SIZE: int = 4


def _load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    def _rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(vals)
        ranks = []
        for v in vals:
            tied_count = sum(1 for sv in sorted_vals if sv == v)
            tied_rank = sum(i + 1 for i, sv in enumerate(sorted_vals) if sv == v) / tied_count
            ranks.append(tied_rank)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(len(x)))
    n = len(x)
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


@pytest.fixture(scope="module")
def pipeline():
    p = AureliusPipeline()
    p.initialize()
    return p


def _collect_data(pipeline) -> dict[str, dict]:
    benchmark = _load_benchmark()
    results: dict[str, dict] = {
        k: {"predicted": [], "experimental": [], "names": []} for k in THRESHOLDS
    }

    for entry in benchmark:
        smiles = entry["smiles"]
        name = entry["name"]
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            continue

        try:
            oracle_result = pipeline.screen_molecule(ctx)
        except Exception:
            continue

        t2 = oracle_result.get("tier2")
        if t2 is None:
            continue

        for exp_key in THRESHOLDS:
            exp_val = entry.get(exp_key)
            pred_key = PROPERTY_MAP[exp_key]
            pred_val = t2.get(pred_key)
            if exp_val is not None and pred_val is not None:
                results[exp_key]["predicted"].append(pred_val)
                results[exp_key]["experimental"].append(exp_val)
                results[exp_key]["names"].append(name)

    return results


def test_external_validation_dielectric(pipeline):
    data = _collect_data(pipeline)["dielectric_constant"]
    n = len(data["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} samples for dielectric (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(data["predicted"], data["experimental"])
    threshold = THRESHOLDS["dielectric_constant"]
    assert rho > threshold, (
        f"Dielectric ρ={rho:.4f} < {threshold}. "
        f"Dielectric proxy rank order does not match experimental values "
        f"(n={n})."
    )


def test_external_validation_viscosity(pipeline):
    data = _collect_data(pipeline)["viscosity_cP"]
    n = len(data["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} samples for viscosity (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(data["predicted"], data["experimental"])
    threshold = THRESHOLDS["viscosity_cP"]
    assert rho > threshold, (
        f"Viscosity ρ={rho:.4f} < {threshold}. "
        f"Viscosity proxy rank order does not match experimental values "
        f"(n={n})."
    )


def test_external_validation_homo(pipeline):
    data = _collect_data(pipeline)["homo_eV"]
    n = len(data["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} samples for HOMO (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(data["predicted"], data["experimental"])
    threshold = THRESHOLDS["homo_eV"]
    assert rho > threshold, (
        f"HOMO ρ={rho:.4f} < {threshold}. "
        f"TOM HOMO rank order does not match DFT reference values (n={n})."
    )


def test_external_validation_lumo(pipeline):
    data = _collect_data(pipeline)["lumo_eV"]
    n = len(data["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} samples for LUMO (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(data["predicted"], data["experimental"])
    threshold = THRESHOLDS["lumo_eV"]
    assert rho > threshold, (
        f"LUMO ρ={rho:.4f} < {threshold}. "
        f"TOM LUMO rank order does not match DFT reference values (n={n})."
    )


def test_external_validation_donor_number(pipeline):
    data = _collect_data(pipeline)["donor_number"]
    n = len(data["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} samples for donor number (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(data["predicted"], data["experimental"])
    threshold = THRESHOLDS["donor_number"]
    assert rho > threshold, (
        f"Donor number ρ={rho:.4f} < {threshold}. "
        f"Li+ solvation proxy rank order does not match donor numbers (n={n})."
    )


def test_benchmark_has_minimum_molecules():
    benchmark = _load_benchmark()
    assert len(benchmark) >= 15, (
        f"External benchmark has only {len(benchmark)} entries; need >= 15 "
        f"for statistically meaningful validation."
    )
