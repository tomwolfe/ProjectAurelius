"""ML baseline benchmark for external property validation.

Compares the physics-grounded oracle predictions against a simple ML baseline
to test whether the "physical interpretability" claim is testable against
the null hypothesis of fingerprint regressors.

Physical justification: Without an ML baseline, the project's claim that
physics-based TOM predictions are valuable cannot be falsified. A simple
ECFP4+RandomForest gives the null hypothesis that fragment-based ML
predictions perform at least as well as the hybrid oracle.
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
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def _load_benchmark() -> list[dict]:
    """Load external property benchmark data."""
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("ML Baseline Benchmark for Project Aurelius")
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
        "viscosity": []
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
    
    print(f"\nAnalyzing {len(molecules)} molecules for ML baseline comparison")
    
    # Properties to analyze (prioritize HOMO/LUMO which are the focus of G1, G2, G3)
    properties = [("HOMO", "homo"), ("LUMO", "lumo")]
    
    print("\n" + "=" * 70)
    print("PROPERTY-WISE COMPARISON: Oracle vs ML Baseline")
    print("=" * 70)
    
    for prop_name, target_key in properties:
        if target_key not in targets or len(targets[target_key]) != len(molecules):
            print(f"\nSkipping {prop_name}: insufficient data")
            continue
            
        y_true = np.array(targets[target_key])
        
        # Generate ECFP4 descriptors for all molecules
        X = np.array([_ecfp4_descriptors(mol) for mol in molecules])
        
        # 5-fold cross-validation with RandomForest
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_true[train_idx], y_true[test_idx]
            
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Calculate Spearman correlation for this fold
            fold_rho = _spearman(y_pred.tolist(), y_test.tolist())
            cv_scores.append(fold_rho)
        
        # Overall performance (average across folds)
        avg_rho = np.mean(cv_scores)
        std_rho = np.std(cv_scores)
        
        print(f"\n{prop_name} Energy:")
        print(f"  ML Baseline (ECFP4+RF): ρ = {avg_rho:.3f} ± {std_rho:.3f}")
        
        # Note: In a real implementation, we would compare against actual oracle predictions
        # For now, we output the ML baseline as a reference point
        print(f"  (Oracle comparison would be done separately)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The ML baseline provides the null hypothesis that fingerprint-based")
    print("predictions perform at least as well as the physics-grounded oracle.")
    print("If oracle ρ < ML baseline ρ - 0.05 for any property, transparency is")
    print("maintained that the physical interpretability claim may not hold.")
    
    print("\nNext steps:")
    print("1. Run the actual oracle predictions on these molecules")
    print("2. Compare oracle ρ vs ML baseline ρ per property")
    print("3. If oracle ρ < ML baseline ρ - 0.05, print WARNING (transparency)")
    print("4. Do NOT fail based on this comparison (transparency > gating)")


if __name__ == "__main__":
    main()
