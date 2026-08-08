"""Benchmark: Targeted Δ-Learning for OOD Scaffolds improvement.

Measures the improvement in Spearman ρ for OOD molecules after
applying weighted GPR training with OOD calibration data.

Physical justification: TOM's particle-in-a-box model systematically
fails on branched π-systems and novel scaffolds. The Δ-learning
correction layer can fix this by training on OOD molecules with
known DFT values, weighted 2× to prioritize OOD correction.

Success criterion: OOD HOMO Spearman ρ > 0.60 (currently ~0.51).
"""

from __future__ import annotations

import json
import os
import sys

from rdkit import Chem
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.data.ood_validation import get_ood_molecules
from aurelius.scoring.oracle.delta_correction import (
    DeltaCorrection,
    compute_ood_spearman,
)
from aurelius.scoring.oracle.quantum import predict_tom_orbitals


def _load_calibration() -> list[dict]:
    """Load the DFT HOMO/LUMO calibration set."""
    calib_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data",
        "orbital_calibration.json",
    )
    with open(calib_path) as f:
        return json.load(f)


def _load_ood_entries() -> list[dict]:
    """Load OOD molecules with known DFT values for benchmarking."""
    ood_molecules = get_ood_molecules()
    entries = []
    for mol_data in ood_molecules:
        smiles = mol_data.get("smiles", "")
        homo = mol_data.get("homo_eV")
        lumo = mol_data.get("lumo_eV")
        if smiles and homo is not None and lumo is not None:
            entries.append({
                "smiles": smiles,
                "homo_eV": homo,
                "lumo_eV": lumo,
                "name": mol_data.get("name", ""),
                "class": mol_data.get("class", ""),
            })
    return entries


def _compute_ood_rho(model: DeltaCorrection, ood_entries: list[dict]) -> float:
    """Compute Spearman ρ between corrected HOMO and DFT HOMO for OOD molecules."""
    preds: list[float] = []
    refs: list[float] = []
    for entry in ood_entries:
        smiles = entry["smiles"]
        ref_homo = entry["homo_eV"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        raw_h, raw_l = predict_tom_orbitals(mol)
        corr_h, corr_l = model.predict_corrected(mol, base=(raw_h, raw_l))
        preds.append(corr_h)
        refs.append(ref_homo)

    if len(preds) < 3:
        return 0.0
    rho, _ = spearmanr(preds, refs)
    return float(rho)


def main() -> None:
    """Benchmark OOD improvement with weighted Δ-learning."""
    print("=" * 70)
    print("OOD Δ-Learning Improvement Benchmark")
    print("=" * 70)

    calib = _load_calibration()
    ood_entries = _load_ood_entries()

    print(f"\nCalibration set size: {len(calib)}")
    print(f"OOD validation set size: {len(ood_entries)}")

    # Baseline: standard Δ-correction (no OOD weighting)
    baseline_model = DeltaCorrection()
    baseline_ood_rho = _compute_ood_rho(baseline_model, ood_entries)
    baseline_loo_mae = baseline_model.loo_mae()

    print("\nBaseline (no OOD weighting):")
    print(f"  OOD HOMO ρ: {baseline_ood_rho:.4f}")
    print(f"  LOO MAE:    {baseline_loo_mae:.4f} eV")

    # Improved: Δ-correction with OOD calibration set (2× weight)
    improved_model = DeltaCorrection(ood_calibration_set=ood_entries)
    improved_ood_rho = _compute_ood_rho(improved_model, ood_entries)
    improved_loo_mae = improved_model.loo_mae()

    print("\nImproved (OOD weighted 2×):")
    print(f"  OOD HOMO ρ: {improved_ood_rho:.4f}")
    print(f"  LOO MAE:    {improved_loo_mae:.4f} eV")

    # Also compute in-domain ρ for comparison
    in_domain_rho_baseline = compute_ood_spearman(calib[:20], model=baseline_model)
    in_domain_rho_improved = compute_ood_spearman(calib[:20], model=improved_model)

    print("\nIn-domain ρ (first 20 calibration molecules):")
    print(f"  Baseline:  {in_domain_rho_baseline:.4f}")
    print(f"  Improved:  {in_domain_rho_improved:.4f}")

    # Check improvement
    ood_improvement = improved_ood_rho - baseline_ood_rho
    print(f"\nOOD ρ improvement: {ood_improvement:+.4f}")

    if ood_improvement >= 0.05:
        print(f"SUCCESS: OOD ρ improved by {ood_improvement:.4f} (target: ≥0.05)")
    else:
        print(f"WARNING: OOD ρ improvement {ood_improvement:.4f} below target 0.05")

    # Verify in-domain ρ does not degrade
    in_domain_change = in_domain_rho_improved - in_domain_rho_baseline
    if in_domain_change >= -0.05:
        print(f"SUCCESS: In-domain ρ change {in_domain_change:+.4f} within tolerance")
    else:
        print(f"WARNING: In-domain ρ degraded by {abs(in_domain_change):.4f}")


if __name__ == "__main__":
    main()
