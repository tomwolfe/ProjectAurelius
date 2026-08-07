"""Oracle Absolute Accuracy Audit for Project Aurelius v11.0.

This script analyzes WHERE the oracle fails (not just HOW WELL it ranks) by:
- Computing MAE/RMSE against reference values
- Identifying top-5 outliers per property
- Hypothesizing failure causes for outliers
- Comparing against ECFP4+RF baseline
- Validating MAE < 1.5 eV for HOMO/LUMO on calibration set

Physical justification: Spearman ρ alone cannot guide synthesis; we need to
know WHEN the oracle is wrong to enable wet-lab decision readiness.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = Path(__file__).parent.parent / "src" / "aurelius" / "data"
BENCHMARKS_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARKS_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def _predict_tom_orbitals(mol: Chem.Mol) -> tuple[float, float]:
    """Predict HOMO/LUMO using TOM (particle-in-a-box model)."""
    if mol is None:
        return 0.0, 0.0
    from aurelius.scoring.oracle.quantum import predict_tom_orbitals
    return predict_tom_orbitals(mol)


def _predict_gc_properties(mol: Chem.Mol) -> tuple[float, float]:
    """Predict (dielectric_proxy, viscosity_proxy) via group-contribution models.

    ADR-2026-08-07-01: Callers previously unpacked this as
    ``_, dielectric_pred = ...``, which silently reported *viscosity* as the
    dielectric constant. That is the sole origin of the "EC epsilon = 1.6 vs
    89.8" figure in oracle_absolute_audit.json — 1.59 is EC's predicted
    viscosity, not its dielectric. Return order is (dielectric, viscosity).
    """
    if mol is None:
        return 0.0, 0.0
    from aurelius.types import MoleculeContext
    from aurelius.scoring.oracle.gc import predict_dielectric_proxy, predict_viscosity_proxy
    ctx = MoleculeContext(smiles="", mol=mol)
    return predict_dielectric_proxy(ctx), predict_viscosity_proxy(ctx)


def _compute_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Compute Tanimoto similarity between two molecules."""
    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

    from rdkit.DataStructs import TanimotoSimilarity

    return TanimotoSimilarity(fp1, fp2)


def _analyze_failure_mode(mol: Chem.Mol, smiles: str, true_homo: float, true_lumo: float) -> str:
    """Analyze structure and hypothesize failure cause for orbital prediction."""
    if mol is None:
        return "invalid_molecule"

    n_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    n_s = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
    n_n = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    # Check for conjugation > 12 atoms
    if n_aromatic > 12:
        return "high_aromatic_conjugation"

    # Check for heavy fluorination (TOM underestimates -I effect)
    if n_f > 8:
        return "excessive_fluorination"

    # Check for sulfur-oxygen interactions (TOM misses)
    if n_s > 2 and n_o > 3:
        return "sulfur_oxygen_clusters"

    # Check for steric clash (TOM misses through-space effects)
    if any(atom.GetDegree() > 4 for atom in mol.GetAtoms()):
        return "steric_clash"

    # Check for nitrile clusters (TOM underestimates -I effect)
    nitrile_count = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7
        and any(neighbor.GetAtomicNum() == 6 and any(n.GetAtomicNum() == 6 for n in neighbor.GetNeighbors())
                for neighbor in atom.GetNeighbors())
    )
    if nitrile_count > 4:
        return "nitrile_cluster"

    # Check for missing GC fragment (TOM assumes uniform electron density)
    hetero_density = (n_f + n_o + n_s + n_n) / max(n_c, 1)
    if hetero_density > 0.8:
        return "high_hetero_density"

    # Default
    return "complex_electronic_structure"


def _hypothesize_gc_failure_mode(mol: Chem.Mol) -> str:
    """Hypothesize cause for GC property prediction failures."""
    if mol is None:
        return "invalid_molecule"

    n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    n_s = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
    n_n = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

    # Fluorine effects on dielectric
    if n_f > 6:
        return "fluorine_dipolar_cancellation"

    # Sulfur-oxygen clusters (strong dipoles)
    if n_s > 1 and n_o > 2:
        return "sulfur_oxygen_cluster_dipoles"

    # Nitrile clusters
    if n_n > 6:
        return "nitrile_cluster_polarity"

    return "structural_complexity"


def main() -> None:
    """Run the Oracle Absolute Accuracy Audit."""
    print("Running Oracle Absolute Accuracy Audit...")

    # Load calibration data
    with open(DATA_DIR / "orbital_calibration.json") as f:
        calibration_data = json.load(f)

    # Load external benchmark data
    with open(DATA_DIR / "external_property_benchmark.json") as f:
        external_data = json.load(f)

    # Load ML baseline comparison
    with open(BENCHMARKS_DIR / "results" / "ml_baseline_comparison.json") as f:
        ml_comparison = json.load(f)

    audit_report = {
        "calibration_set_size": len(calibration_data),
        "external_benchmark_size": len(external_data),
        "ml_baseline_comparison": ml_comparison,
        "properties": {},
    }

    # Process HOMO property
    print("\nProcessing HOMO property...")
    homo_errors = []
    for entry in calibration_data:
        mol = Chem.MolFromSmiles(entry["smiles"])
        homo_pred, lumo_pred = _predict_tom_orbitals(mol)
        homo_true = entry["homo_eV"]
        error = abs(homo_pred - homo_true)
        homo_errors.append({
            "smiles": entry["smiles"],
            "name": entry["name"],
            "predicted": homo_pred,
            "true": homo_true,
            "absolute_error": error,
            "failure_mode": _analyze_failure_mode(mol, entry["smiles"], homo_true, entry["lumo_eV"])
        })

    # Sort by error and get top-5 outliers
    homo_errors.sort(key=lambda x: -x["absolute_error"])
    audit_report["properties"]["HOMO"] = {
        "n_calibrated": len(calibration_data),
        "mae": float(np.mean([e["absolute_error"] for e in homo_errors])),
        "rmse": float(np.sqrt(np.mean([e["absolute_error"] ** 2 for e in homo_errors]))),
        "max_error": float(max(e["absolute_error"] for e in homo_errors)),
        "top_5_outliers": homo_errors[:5],
        "ml_baseline_oracle_gap": ml_comparison["properties"]["HOMO"]["gap"],
        "ml_baseline_better": ml_comparison["properties"]["HOMO"]["oracle_wins"] is False,
    }

    # Process LUMO property
    print("\nProcessing LUMO property...")
    lumo_errors = []
    for entry in calibration_data:
        mol = Chem.MolFromSmiles(entry["smiles"])
        homo_pred, lumo_pred = _predict_tom_orbitals(mol)
        lumo_true = entry["lumo_eV"]
        error = abs(lumo_pred - lumo_true)
        lumo_errors.append({
            "smiles": entry["smiles"],
            "name": entry["name"],
            "predicted": lumo_pred,
            "true": lumo_true,
            "absolute_error": error,
            "failure_mode": _analyze_failure_mode(mol, entry["smiles"], entry["homo_eV"], lumo_true)
        })

    lumo_errors.sort(key=lambda x: -x["absolute_error"])
    audit_report["properties"]["LUMO"] = {
        "n_calibrated": len(calibration_data),
        "mae": float(np.mean([e["absolute_error"] for e in lumo_errors])),
        "rmse": float(np.sqrt(np.mean([e["absolute_error"] ** 2 for e in lumo_errors]))),
        "max_error": float(max(e["absolute_error"] for e in lumo_errors)),
        "top_5_outliers": lumo_errors[:5],
        "ml_baseline_oracle_gap": ml_comparison["properties"]["LUMO"]["gap"],
        "ml_baseline_better": not ml_comparison["properties"]["LUMO"]["oracle_wins"],
    }

    # Process Dielectric property
    print("\nProcessing Dielectric property...")
    dielectric_errors = []
    for entry in external_data:
        if entry.get("dielectric_constant") is None:
            continue
        mol = Chem.MolFromSmiles(entry["smiles"])
        dielectric_pred, _ = _predict_gc_properties(mol)
        dielectric_true = entry["dielectric_constant"]
        error = abs(dielectric_pred - dielectric_true)
        dielectric_errors.append({
            "smiles": entry["smiles"],
            "name": entry["name"],
            "predicted": dielectric_pred,
            "true": dielectric_true,
            "absolute_error": error,
            "failure_mode": _hypothesize_gc_failure_mode(mol)
        })

    dielectric_errors.sort(key=lambda x: -x["absolute_error"])
    audit_report["properties"]["Dielectric"] = {
        "n_calibrated": len(dielectric_errors),
        "mae": float(np.mean([e["absolute_error"] for e in dielectric_errors])),
        "rmse": float(np.sqrt(np.mean([e["absolute_error"] ** 2 for e in dielectric_errors]))),
        "max_error": float(max(e["absolute_error"] for e in dielectric_errors)),
        "top_5_outliers": dielectric_errors[:5],
    }

    # Process Viscosity property
    print("\nProcessing Viscosity property...")
    viscosity_errors = []
    for entry in external_data:
        if entry.get("viscosity_cP") is None:
            continue
        mol = Chem.MolFromSmiles(entry["smiles"])
        _, viscosity_pred = _predict_gc_properties(mol)
        viscosity_true = entry["viscosity_cP"]
        error = abs(viscosity_pred - viscosity_true)
        viscosity_errors.append({
            "smiles": entry["smiles"],
            "name": entry["name"],
            "predicted": viscosity_pred,
            "true": viscosity_true,
            "absolute_error": error,
            "failure_mode": _hypothesize_gc_failure_mode(mol)
        })

    viscosity_errors.sort(key=lambda x: -x["absolute_error"])
    audit_report["properties"]["Viscosity"] = {
        "n_calibrated": len(viscosity_errors),
        "mae": float(np.mean([e["absolute_error"] for e in viscosity_errors])),
        "rmse": float(np.sqrt(np.mean([e["absolute_error"] ** 2 for e in viscosity_errors]))),
        "max_error": float(max(e["absolute_error"] for e in viscosity_errors)),
        "top_5_outliers": viscosity_errors[:5],
    }

    # Save audit report
    output_path = RESULTS_DIR / "oracle_absolute_audit.json"
    with open(output_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nAudit report saved to {output_path}")

    # Validate MAE < 1.5 eV for HOMO/LUMO
    homo_mae = audit_report["properties"]["HOMO"]["mae"]
    lumo_mae = audit_report["properties"]["LUMO"]["mae"]

    print(f"\nValidation:")
    print(f"  HOMO MAE: {homo_mae:.4f} eV {'✓' if homo_mae < 1.5 else '✗'}")
    print(f"  LUMO MAE: {lumo_mae:.4f} eV {'✓' if lumo_mae < 1.5 else '✗'}")

    if homo_mae >= 1.5 or lumo_mae >= 1.5:
        raise AssertionError(f"MAE >= 1.5 eV for HOMO ({homo_mae:.4f}) or LUMO ({lumo_mae:.4f})")

    print("\n✓ Oracle Absolute Accuracy Audit completed successfully!")


if __name__ == "__main__":
    main()
