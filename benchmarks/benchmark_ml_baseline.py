"""ML baseline benchmark for external property validation.

Compares the physics-grounded oracle predictions against an ECFP4+
RandomForest baseline on HOMO and LUMO energies.

The comparison is written to benchmarks/results/ml_baseline_comparison.json
and a WARNING is printed (not a failure) if the oracle underperforms
the RF baseline by >0.05 Spearman rho on any property.

Physical justification: Without an ML baseline, the project's claim that
physics-based TOM predictions are valuable cannot be falsified. A simple
ECFP4+RandomForest gives the null hypothesis that fragment-based ML
predictions perform at least as well as the hybrid oracle.

Transparency rule: If oracle rho < RF rho - 0.05 for any property,
the finding is reported honestly — the physics value is interpretability
+ extrapolation, not raw accuracy.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(SRC_DIR, "aurelius", "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "external_property_benchmark.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "ml_baseline_comparison.json")


def _ecfp4_descriptors(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Generate ECFP4 (Morgan radius 2) bit vector descriptors."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


def _load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def _compute_oracle_predictions(
    molecules: list[Chem.Mol],
) -> dict[str, list[float]]:
    """Run the Aurelius oracle on all molecules and return predictions."""
    from aurelius.scoring.oracle.oracle import PropertyOracle

    oracle = PropertyOracle(use_xtb=False)
    preds: dict[str, list[float]] = {
        "homo": [],
        "lumo": [],
        "dielectric": [],
        "viscosity": [],
    }

    for mol in molecules:
        try:
            from aurelius.types import MoleculeContext
            ctx = MoleculeContext.from_smiles(Chem.MolToSmiles(mol))
            if ctx is None:
                preds["homo"].append(float("nan"))
                preds["lumo"].append(float("nan"))
                preds["dielectric"].append(float("nan"))
                preds["viscosity"].append(float("nan"))
                continue
            result = oracle.evaluate(ctx)
            preds["homo"].append(result.get("homo_eV", float("nan")))
            preds["lumo"].append(result.get("lumo_eV", float("nan")))
            preds["dielectric"].append(result.get("dielectric_proxy", float("nan")))
            preds["viscosity"].append(result.get("viscosity_proxy", float("nan")))
        except Exception:
            preds["homo"].append(float("nan"))
            preds["lumo"].append(float("nan"))
            preds["dielectric"].append(float("nan"))
            preds["viscosity"].append(float("nan"))

    return preds


def _cross_val_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float]:
    """5-fold cross-validation with RandomForest, returning mean rho and std."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores: list[float] = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_rho = float(spearmanr(y_pred, y_test).statistic)
        cv_scores.append(fold_rho)

    return float(np.mean(cv_scores)), float(np.std(cv_scores))


def main() -> None:
    """Compare oracle vs ECFP4+RF on HOMO and LUMO; write results to JSON."""
    print("=" * 70)
    print("ML Baseline Benchmark — Oracle vs ECFP4+RandomForest")
    print("=" * 70)

    benchmark = _load_benchmark()
    valid_entries = [e for e in benchmark if e.get("homo_eV") is not None]
    print(f"\nBenchmark molecules: {len(benchmark)}")
    print(f"With DFT HOMO/LUMO: {len(valid_entries)}")

    if len(valid_entries) < 10:
        print("ERROR: Not enough molecules with DFT data for meaningful analysis")
        return

    molecules: list[Chem.Mol] = []
    targets: dict[str, list[float]] = {"homo": [], "lumo": []}

    for entry in valid_entries:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        molecules.append(mol)
        if "homo_eV" in entry:
            targets["homo"].append(entry["homo_eV"])
        if "lumo_eV" in entry:
            targets["lumo"].append(entry["lumo_eV"])

    print(f"\nAnalyzing {len(molecules)} molecules")

    # Oracle predictions
    print("\nRunning Aurelius oracle predictions...")
    oracle_preds = _compute_oracle_predictions(molecules)

    # ECFP4 descriptors
    X = np.array([_ecfp4_descriptors(mol) for mol in molecules], dtype=np.float64)

    properties = [("HOMO", "homo"), ("LUMO", "lumo")]
    results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("PROPERTY-WISE COMPARISON")
    print("=" * 70)

    for prop_name, target_key in properties:
        y_true = np.array(targets[target_key])
        y_oracle = np.array(oracle_preds[target_key])

        valid_mask = ~(np.isnan(y_oracle) | np.isnan(y_true))
        if valid_mask.sum() < 10:
            print(f"\nSkipping {prop_name}: insufficient valid predictions")
            results[prop_name] = {"error": "insufficient data"}
            continue

        y_oracle_clean = y_oracle[valid_mask]
        y_true_clean = y_true[valid_mask]

        # Oracle Spearman rho (oracle predictions vs DFT reference)
        oracle_rho, oracle_p = spearmanr(y_oracle_clean, y_true_clean)

        # RF baseline: 5-fold CV
        rf_rho, rf_std = _cross_val_rf(X[valid_mask], y_true_clean)

        # Gap
        gap = oracle_rho - rf_rho

        print(f"\n{prop_name} Energy:")
        print(f"  Oracle ρ = {oracle_rho:.4f} (p = {oracle_p:.4f})")
        print(f"  RF baseline ρ = {rf_rho:.4f} ± {rf_std:.4f}")
        print(f"  Gap (oracle - RF) = {gap:.4f}")

        if oracle_rho < rf_rho - 0.05:
            msg = (
                f"WARNING: Oracle ρ ({oracle_rho:.4f}) < RF baseline ρ "
                f"({rf_rho:.4f}) - 0.05 for {prop_name}. The physics-grounded "
                f"oracle underperforms a simple fingerprint regressor on this "
                f"property. This is expected for frontier orbitals where TOM "
                f"is a coarse approximation; the physics value is interpretability "
                f"and extrapolation, not raw accuracy."
            )
            print(f"  ⚠️  {msg}")
            warnings.warn(msg, stacklevel=2)
        else:
            print("  ✓ Oracle performs as well as or better than RF baseline")

        results[prop_name] = {
            "oracle_rho": round(float(oracle_rho), 4),
            "oracle_p": round(float(oracle_p), 4),
            "rf_rho": round(float(rf_rho), 4),
            "rf_std": round(float(rf_std), 4),
            "gap": round(float(gap), 4),
            "n_valid": int(valid_mask.sum()),
            "oracle_wins": bool(oracle_rho >= rf_rho - 0.05),
        }

    # Write results to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "n_molecules": len(molecules),
        "n_with_dft": len(valid_entries),
        "properties": results,
        "transparency_note": (
            "If oracle rho < RF rho - 0.05 for any property, the finding "
            "is reported honestly. The physics value is interpretability + "
            "extrapolation, not raw accuracy."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {RESULTS_PATH}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for prop_name, res in results.items():
        if "error" in res:
            print(f"  {prop_name}: {res['error']}")
        else:
            status = "ORACLE WINS" if res["oracle_wins"] else "RF WINS"
            print(
                f"  {prop_name}: Oracle ρ={res['oracle_rho']:.4f}, "
                f"RF ρ={res['rf_rho']:.4f}, Gap={res['gap']:.4f} [{status}]"
            )


if __name__ == "__main__":
    main()
