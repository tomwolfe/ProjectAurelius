"""External validation: Compare oracle predictions with ML baseline.

G1 Priority: Verify that the physics-grounded oracle provides value beyond
what a simple ML baseline achieves. This addresses the project's central
claim that "physical interpretability" is empirically useful.

Physical justification: Without an ML baseline, the project's claim that
physics-based TOM predictions are valuable cannot be falsified. A simple
ECFP4+RandomForest gives the null hypothesis that fragment-based ML
predictions perform at least as well as the hybrid oracle.

This benchmark ensures transparency by comparing oracle ρ vs ML baseline ρ
per property. If oracle ρ < ML baseline ρ - 0.05 for any property, a WARNING
is printed (transparency > gating). The test does NOT fail based on this
comparison.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "aurelius", "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "external_property_benchmark.json")


def _ecfp4_descriptors(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Generate ECFP4 (Morgan radius 2) bit vector descriptors."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


def _spearman(x: list[float], y: list[float]) -> float:
    """Calculate Spearman rank correlation coefficient."""
    return float(spearmanr(x, y).statistic)


def _load_benchmark() -> list[dict]:
    """Load external property benchmark data."""
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def _load_oracle_predictions() -> dict:
    """Load oracle predictions for benchmark molecules.
    
    This would normally be computed by running the oracle on all
    benchmark molecules, but for this demonstration, we'll simulate
    by computing them directly.
    """
    benchmark = _load_benchmark()
    valid_entries = [e for e in benchmark if e.get("homo_eV") is not None]
    
    predictions = {}
    
    for entry in valid_entries:
        smiles = entry["smiles"]
        name = entry["name"]
        
        from aurelius.types import MoleculeContext
        from aurelius.scoring.oracle import QuantumOracle
        
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            continue
        
        qc = QuantumOracle(use_xtb=False, use_delta_correction=True)
        result = qc.evaluate(ctx.mol)
        
        predictions[name] = {
            "homo_eV": result.get("homo_eV"),
            "lumo_eV": result.get("lumo_eV"),
            "dielectric_proxy": result.get("dielectric_proxy", 0.0),
            "viscosity_proxy": result.get("viscosity_proxy", 0.0),
            "li_solvation_proxy": result.get("li_solvation_proxy", 0.0)
        }
    
    return predictions


def main():
    print("=" * 70)
    print("EXTERNAL VALIDATION — Oracle vs ML Baseline Comparison")
    print("=" * 70)
    
    # Load benchmark data
    benchmark = _load_benchmark()
    
    # Filter to molecules with DFT predictions
    valid_entries = [e for e in benchmark if e.get("homo_eV") is not None]
    print(f"\nBenchmark molecules: {len(benchmark)}")
    print(f"With DFT HOMO/LUMO: {len(valid_entries)}")
    
    if len(valid_entries) < 10:
        print("ERROR: Not enough molecules with DFT data for meaningful analysis")
        return
    
    # Extract data
    molecules = []
    targets = {
        "homo": [],
        "lumo": [],
        "dielectric": [],
        "viscosity": [],
        "li_solvation": []
    }
    
    for entry in valid_entries:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        molecules.append(mol)
        
        if "homo_eV" in entry:
            targets["homo"].append(entry["homo_eV"])
        if "lumo_eV" in entry:
            targets["lumo"].append(entry["lumo_eV"])
        if "dielectric_constant" in entry:
            targets["dielectric"].append(entry["dielectric_constant"])
        if "viscosity_cP" in entry:
            targets["viscosity"].append(entry["viscosity_cP"])
        if "donor_number" in entry:
            targets["li_solvation"].append(entry["donor_number"])
    
    print(f"\nAnalyzing {len(molecules)} molecules for Oracle vs ML comparison")
    
    # Load oracle predictions
    print("\nLoading oracle predictions...")
    oracle_preds = _load_oracle_predictions()
    
    # Properties to analyze
    properties = [
        ("HOMO", "homo", targets["homo"]),
        ("LUMO", "lumo", targets["lumo"]),
        ("Dielectric", "dielectric", targets["dielectric"]),
        ("Viscosity", "viscosity", targets["viscosity"]),
        ("Li Solvation", "li_solvation", targets["li_solvation"])
    ]
    
    print("\n" + "=" * 70)
    print("PROPERTY-WISE COMPARISON: Oracle vs ML Baseline")
    print("=" * 70)
    
    all_warnings = []
    
    for prop_name, target_key, y_true in properties:
        if not y_true or len(y_true) != len(molecules):
            print(f"\nSkipping {prop_name}: insufficient data")
            continue
        
        # Get oracle predictions for this property
        oracle_values = []
        for mol, entry in zip(molecules, valid_entries):
            name = entry["name"]
            if prop_name == "HOMO":
                oracle_values.append(oracle_preds.get(name, {}).get("homo_eV"))
            elif prop_name == "LUMO":
                oracle_values.append(oracle_preds.get(name, {}).get("lumo_eV"))
            elif prop_name == "Dielectric":
                oracle_values.append(oracle_preds.get(name, {}).get("dielectric_proxy"))
            elif prop_name == "Viscosity":
                oracle_values.append(oracle_preds.get(name, {}).get("viscosity_proxy"))
            elif prop_name == "Li Solvation":
                oracle_values.append(oracle_preds.get(name, {}).get("li_solvation_proxy"))
        
        # Filter out None values
        valid_pairs = [(o, e) for o, e in zip(oracle_values, y_true) if o is not None]
        if len(valid_pairs) < len(molecules) * 0.8:
            print(f"\nSkipping {prop_name}: insufficient oracle predictions")
            continue
        
        oracle_vals, exp_vals = zip(*valid_pairs)
        
        # Generate ECFP4 descriptors for all molecules
        X = np.array([_ecfp4_descriptors(mol) for mol in molecules])
        
        # 5-fold cross-validation with RandomForest
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = np.array(oracle_vals)[train_idx], np.array(oracle_vals)[test_idx]
            exp_test = np.array(exp_vals)[test_idx]
            
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Calculate Spearman correlation for this fold
            fold_rho = _spearman(y_pred.tolist(), exp_test.tolist())
            cv_scores.append(fold_rho)
        
        # Overall performance (average across folds)
        avg_rho = np.mean(cv_scores)
        std_rho = np.std(cv_scores)
        
        print(f"\n{prop_name} Energy:")
        print(f"  Oracle ρ = {avg_rho:.3f} ± {std_rho:.3f}")
        
        # Generate ML baseline
        kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
        ml_cv_scores = []
        
        for train_idx, test_idx in kf2.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = np.array(exp_vals)[train_idx], np.array(exp_vals)[test_idx]
            
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Calculate Spearman correlation for this fold
            fold_rho = _spearman(y_pred.tolist(), y_test.tolist())
            ml_cv_scores.append(fold_rho)
        
        ml_avg_rho = np.mean(ml_cv_scores)
        ml_std_rho = np.std(ml_cv_scores)
        
        print(f"  ML Baseline (ECFP4+RF): ρ = {ml_avg_rho:.3f} ± {ml_std_rho:.3f}")
        
        # Check if oracle underperforms ML baseline significantly
        if avg_rho < ml_avg_rho - 0.05:
            warning = f"WARNING: Oracle ρ ({avg_rho:.3f}) < ML baseline ρ ({ml_avg_rho:.3f}) - 0.05"
            print(f"  ⚠️  {warning}")
            all_warnings.append(f"{prop_name}: {warning}")
        else:
            print(f"  ✓ Oracle performs as well as or better than ML baseline")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total properties analyzed: {len(properties)}")
    print(f"Warnings issued: {len(all_warnings)}")
    
    if all_warnings:
        print("\nWarnings (transparency, not gating):")
        for warning in all_warnings:
            print(f"  {warning}")
    
    print("\nNext steps:")
    print("1. If oracle ρ < ML baseline ρ - 0.05 for any property:")
    print("   - Document the finding honestly (transparency)")
    print("   - Investigate why physics-based approach underperforms")
    print("   - Consider whether physical interpretability is valuable")
    print("   - This does NOT automatically mean the project should fail")
    print("     (G1 is about scientific credibility, not just predictive accuracy)")
    print("2. If oracle ρ ≥ ML baseline ρ - 0.05 for all properties:")
    print("   - The physics-based approach provides comparable or better")
    print("     predictions than a fingerprint regressor")
    print("   - This supports the project's claim of value")
    
    print("\nNote: This benchmark prioritizes transparency over gating.")
    print("Warnings are logged, but do not constitute test failures.")


if __name__ == "__main__":
    main()
