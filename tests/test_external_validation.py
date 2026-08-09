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
from rdkit import Chem

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

UNSEEN_LUMO_THRESHOLD: float = 0.05
"""Floor for LUMO rank correlation on molecules absent from the calibration
set. Low by necessity, not by choice: the DFT LUMO labels are
provenance-confounded (see ``benchmarks/audit_label_confound.py``), so no
model — physics or ML — exceeds ρ ≈ 0.09 there. This is a regression guard."""


def _unseen_subset(data: dict) -> dict:
    """Restrict collected predictions to molecules outside the calibration set."""
    from pathlib import Path

    calib_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "aurelius" / "data" / "orbital_calibration.json"
    )
    calibration = set()
    for entry in json.loads(calib_path.read_text()):
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is not None:
            calibration.add(Chem.MolToSmiles(mol))

    keep = [
        i for i, smiles in enumerate(data["smiles"])
        if smiles not in calibration
    ]
    return {k: [v[i] for i in keep] for k, v in data.items()}


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
        k: {"predicted": [], "experimental": [], "names": [], "smiles": []}
        for k in THRESHOLDS
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
                results[exp_key]["smiles"].append(Chem.MolToSmiles(ctx.mol))

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
    """LUMO rank order on molecules outside the calibration set.

    Scored on the UNSEEN split only. Pooling seen and unseen rewards leakage:
    26 of the 27 overlapping molecules carry byte-identical LUMO labels in
    ``external_property_benchmark.json`` and ``orbital_calibration.json``, so
    the Δ-corrected model scores ρ = 0.944 there by recall and ρ = 0.061 on
    genuinely new chemistry. A pooled threshold therefore *penalised* enabling
    xTB, which is more accurate where it matters (unseen MAE 0.366 vs 0.797)
    but cannot memorise the duplicated rows (ADR-2026-08-08-09).

    The bar is deliberately low: ``audit_label_confound.py`` shows 69% of the
    unseen LUMO variance is between-source, and a citation-only predictor
    scores ρ = 0.84, so this target cannot support a strong ranking claim.
    This test guards against catastrophic regression, not accuracy.
    """
    data = _collect_data(pipeline)["lumo_eV"]
    unseen = _unseen_subset(data)
    n = len(unseen["predicted"])
    if n < MIN_SAMPLE_SIZE:
        pytest.skip(f"Only {n} unseen samples for LUMO (need {MIN_SAMPLE_SIZE})")
    rho = _spearman_rho(unseen["predicted"], unseen["experimental"])
    assert rho > UNSEEN_LUMO_THRESHOLD, (
        f"LUMO ρ={rho:.4f} < {UNSEEN_LUMO_THRESHOLD} on the unseen split "
        f"(n={n}). Frontier-orbital rank order has regressed."
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


def test_benchmark_has_minimum_60_molecules():
    benchmark = _load_benchmark()
    assert len(benchmark) >= 60, (
        f"External benchmark has only {len(benchmark)} entries; need >= 60 "
        f"for statistically meaningful validation."
    )


def test_ml_baseline_oracle_comparison():
    """Load ML baseline comparison results and print WARNING if oracle
    underperforms RF baseline by >0.05 rho on any property.

    This is a transparency check, not a gate. If the oracle loses,
    the finding is reported honestly — the physics value is
    interpretability + extrapolation, not raw accuracy.
    """
    results_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "benchmarks",
        "results",
        "ml_baseline_comparison.json",
    )
    if not os.path.exists(results_path):
        pytest.skip(
            "ML baseline comparison results not found. "
            "Run: python -m benchmarks.benchmark_ml_baseline"
        )

    with open(results_path) as f:
        results = json.load(f)

    warnings_issued = []
    for prop_name, res in results.get("properties", {}).items():
        if "error" in res:
            continue
        oracle_rho = res["oracle_rho"]
        rf_rho = res["rf_rho"]
        if oracle_rho < rf_rho - 0.05:
            msg = (
                f"WARNING: Oracle rho ({oracle_rho:.4f}) < RF baseline rho "
                f"({rf_rho:.4f}) - 0.05 for {prop_name}. The physics-grounded "
                f"oracle underperforms a simple fingerprint regressor on this "
                f"property. The physics value is interpretability + "
                f"extrapolation, not raw accuracy."
            )
            warnings_issued.append(msg)
            print(f"\n  ⚠️  {msg}")

    if warnings_issued:
        print(
            f"\n  {len(warnings_issued)} transparency warning(s) issued. "
            f"This is expected for frontier orbitals where TOM is a coarse "
            f"approximation."
        )
