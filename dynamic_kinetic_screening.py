#!/usr/bin/env python3
"""
Phase 5: Dynamic Kinetic Calibration.
Addresses the 'Blind Tier 3' issue by mapping RDKit electronic descriptors
to effective Activation Energy (Ea) shifts for the kMC simulator.

Strategy:
1. Calculate Electronegativity/Index of Refraction/MolLogP as proxies for LUMO/HOMO.
2. Apply heuristic shifts to Ea_SOLVENT, Ea_SALT, Ea_POLYMER.
3. Patch the GCMDigitalTwin instance in memory before simulation.
"""

import json
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig
from aurelius.types import GCMDTwinResult


def calculate_electronic_descriptors(smiles: str) -> dict:
    """Calculate RDKit descriptors that correlate with reduction potential."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # 1. MolLogP: Proxy for hydrophobicity/electron density distribution
    logp = Descriptors.MolLogP(mol)

    # 2. NumFluorineAtoms: Strong electron withdrawer, lowers LUMO
    num_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)

    # 3. NumAromaticRings / Unsaturation: Proxy for polymerization potential
    num_aromatic = Lipinski.NumAromaticRings(mol)
    num_double_bonds = sum(
        1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
    )
    num_triple_bonds = sum(
        1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE
    )

    # 4. MaxPartialCharge: Proxy for local reactivity
    # Note: Gasteiger charges are approximate but fast
    AllChem.ComputeGasteigerCharges(mol)
    max_charge = max(
        [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    )

    return {
        "logp": logp,
        "num_f": num_f,
        "num_aromatic": num_aromatic,
        "num_unsat": num_double_bonds + num_triple_bonds,
        "max_charge": max_charge,
    }


def compute_ea_shifts(descriptors: dict) -> dict:
    """
    Map descriptors to Activation Energy (Ea) shifts (eV).
    Base Ea values from force_field_params.json:
    - Solvent (EC): 0.65 eV
    - Salt (PF6): 1.20 eV
    - Polymer: 0.40 eV

    Heuristics:
    - High F count -> Lowers Solvent Ea (easier reduction to LiF/NaF)
    - High Unsaturation -> Lowers Polymer Ea (easier polymerization)
    - High Max Charge -> Lowers Salt Ea (stronger ion pairing/decomp)
    """
    delta_ea_solvent = 0.0
    delta_ea_salt = 0.0
    delta_ea_poly = 0.0

    # Fluorine effect: Each F reduces Solvent Ea by ~0.05 eV (cap at 0.3 eV)
    delta_ea_solvent -= min(descriptors["num_f"] * 0.05, 0.3)

    # Unsaturation effect: Each double/triple bond reduces Polymer Ea by ~0.1 eV
    delta_ea_poly -= min(descriptors["num_unsat"] * 0.1, 0.4)

    # Aromaticity: Stabilizes radicals, slightly increases Polymer Ea (harder to break ring)
    # But promotes pi-stacking SEI. Let's assume it helps homogeneity by slowing polymerization slightly?
    # Actually, aromatic rings often form stable SEI components. Let's lower Salt Ea slightly if aromatic?
    # Let's keep it simple: Aromatics reduce Solvent Ea slightly (stable SEI)
    delta_ea_solvent -= descriptors["num_aromatic"] * 0.02

    # Max Charge: High positive charge on P/B/S centers facilitates salt decomp
    if descriptors["max_charge"] > 0.5:
        delta_ea_salt -= 0.2

    return {
        "delta_solvent": delta_ea_solvent,
        "delta_salt": delta_ea_salt,
        "delta_poly": delta_ea_poly,
    }


def screen_with_dynamic_kinetics(smiles: str, base_twin: GCMDigitalTwin) -> GCMDTwinResult:
    """
    Screen a single molecule by patching the Twin's Ea values in memory.
    """
    descriptors = calculate_electronic_descriptors(smiles)
    if not descriptors:
        raise ValueError(f"Invalid SMILES: {smiles}")

    shifts = compute_ea_shifts(descriptors)

    # Clone the twin to avoid thread safety issues if batching later
    # For now, we just modify the instance attributes directly for this call
    original_ea_solvent = base_twin._Ea_SOLVENT_EC
    original_ea_salt = base_twin._Ea_SALT_PF6
    original_ea_poly = base_twin._activation_energies.get("polymerization", 0.40)

    # Apply Shifts
    base_twin._Ea_SOLVENT_EC = original_ea_solvent + shifts["delta_solvent"]
    base_twin._Ea_SALT_PF6 = original_ea_salt + shifts["delta_salt"]
    base_twin._activation_energies["polymerization"] = (
        original_ea_poly + shifts["delta_poly"]
    )

    try:
        result = base_twin.simulate_sei_evolution(
            smiles=smiles,
            solvent_type="ec:dmc",
            salt_type="NaPF6",
            voltage_cutoff=0.05,
            max_time_ps=1000.0,
        )
    finally:
        # Restore original values
        base_twin._Ea_SOLVENT_EC = original_ea_solvent
        base_twin._Ea_SALT_PF6 = original_ea_salt
        base_twin._activation_energies["polymerization"] = original_ea_poly

    return result


def main():
    print("=" * 60)
    print("  Phase 5: Dynamic Kinetic Calibration")
    print("=" * 60)

    # Load candidates from previous phase
    candidates = []
    with open("homogeneity_targeted_candidates.smi") as f:
        candidates.extend(
            [line.strip() for line in f if line.strip() and not line.startswith("#")]
        )
    with open("refined_candidates.smi") as f:
        candidates.extend(
            [line.strip() for line in f if line.strip() and not line.startswith("#")]
        )

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    candidates = unique_candidates

    print(f"\nScreening {len(candidates)} candidates with dynamic kinetics...\n")

    # Initialize Base Twin (5000 steps for better homogeneity resolution)
    twin = GCMDigitalTwin(
        gcmtwin_config=GCMDTConfig(max_simulation_steps=5000)
    )

    results = []
    for smiles in candidates:
        try:
            res = screen_with_dynamic_kinetics(smiles, twin)
            results.append(
                {
                    "smiles": smiles,
                    "sei_homogeneity_raw": res.sei_evolution.homogeneity_score,
                    "sei_homogeneity_scaled": res.sei_evolution.homogeneity_score * 100,
                    "thickness": res.sei_evolution.thickness_angstrom,
                    "components": res.sei_evolution.components,
                }
            )
            print(
                f"  {smiles}: Homog={res.sei_evolution.homogeneity_score:.3f} ({res.sei_evolution.homogeneity_score*100:.1f}/100)"
            )
        except Exception as e:
            print(f"  ERROR {smiles}: {e}")

    # Save results
    with open("dynamic_kinetic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to dynamic_kinetic_results.json")

    # Analyze
    if results:
        best_homog = max(results, key=lambda x: x["sei_homogeneity_raw"])
        print(f"\n{'=' * 60}")
        print("  Best Candidate for Homogeneity:")
        print(f"    SMILES:    {best_homog['smiles']}")
        print(f"    Homogeneity: {best_homog['sei_homogeneity_scaled']:.1f}/100 (raw {best_homog['sei_homogeneity_raw']:.4f})")
        print(f"    SEI Components: {best_homog['components']}")
        print(f"    Thickness: {best_homog['thickness']:.1f} A")

        # Compare with baseline
        baseline_homog = 0.122
        print(f"\n  Improvement over baseline (raw 0.122): +{best_homog['sei_homogeneity_raw'] - baseline_homog:.4f}")

        # Show top 5 by homogeneity
        sorted_results = sorted(results, key=lambda x: x["sei_homogeneity_raw"], reverse=True)
        print(f"\n  Top 5 by Homogeneity:")
        for r in sorted_results[:5]:
            print(
                f"    {r['smiles']:40s}  raw={r['sei_homogeneity_raw']:.4f}  scaled={r['sei_homogeneity_scaled']:.1f}  components={r['components']}"
            )


if __name__ == "__main__":
    main()
