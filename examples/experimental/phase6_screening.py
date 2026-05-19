#!/usr/bin/env python3
"""
Phase 6: Borate-Inspired Refined Candidate Screening.

Targets: Maintain high SEI homogeneity while passing Tier 1/2 viability.
Uses dynamic kinetic calibration to compute molecule-specific Ea shifts.
"""

import json

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig


def calculate_electronic_descriptors(smiles: str) -> dict:
    """Calculate RDKit descriptors that correlate with reduction potential."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    logp = Descriptors.MolLogP(mol)
    num_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    num_aromatic = Lipinski.NumAromaticRings(mol)
    num_double_bonds = sum(
        1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
    )
    num_triple_bonds = sum(
        1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE
    )

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
    """Map descriptors to Activation Energy (Ea) shifts (eV)."""
    delta_ea_solvent = 0.0
    delta_ea_salt = 0.0
    delta_ea_poly = 0.0

    delta_ea_solvent -= min(descriptors["num_f"] * 0.05, 0.3)
    delta_ea_poly -= min(descriptors["num_unsat"] * 0.1, 0.4)
    delta_ea_solvent -= descriptors["num_aromatic"] * 0.02

    if descriptors["max_charge"] > 0.5:
        delta_ea_salt -= 0.2

    return {
        "delta_solvent": delta_ea_solvent,
        "delta_salt": delta_ea_salt,
        "delta_poly": delta_ea_poly,
    }


def screen_with_dynamic_kinetics(smiles: str, base_twin: GCMDigitalTwin) -> dict:
    """Screen a single molecule with dynamic Ea patching."""
    descriptors = calculate_electronic_descriptors(smiles)
    if not descriptors:
        return {"smiles": smiles, "error": "Invalid SMILES"}

    shifts = compute_ea_shifts(descriptors)

    original_ea_solvent = base_twin._Ea_SOLVENT_EC
    original_ea_salt = base_twin._Ea_SALT_PF6
    original_ea_poly = base_twin._activation_energies.get("polymerization", 0.40)

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
        return {
            "smiles": smiles,
            "sei_homogeneity_raw": result.sei_evolution.homogeneity_score,
            "sei_homogeneity_scaled": result.sei_evolution.homogeneity_score * 100,
            "thickness": result.sei_evolution.thickness_angstrom,
            "components": result.sei_evolution.components,
            "electronic_insulation": result.sei_evolution.electronic_insulation,
            "ea_shifts": shifts,
            "descriptor_num_f": descriptors["num_f"],
            "descriptor_num_unsat": descriptors["num_unsat"],
            "descriptor_logp": descriptors["logp"],
        }
    finally:
        base_twin._Ea_SOLVENT_EC = original_ea_solvent
        base_twin._Ea_SALT_PF6 = original_ea_salt
        base_twin._activation_energies["polymerization"] = original_ea_poly


def main():
    print("=" * 70)
    print("  Phase 6: Borate-Inspired Refined Candidate Screening")
    print("=" * 70)

    # Load refined candidates
    candidates = []
    with open("phase6_refined_candidates.smi") as f:
        candidates.extend(
            [line.strip() for line in f if line.strip() and not line.startswith("#")]
        )

    print(f"\nScreening {len(candidates)} refined candidates with dynamic kinetics...\n")

    twin = GCMDigitalTwin(
        gcmtwin_config=GCMDTConfig(max_simulation_steps=5000)
    )

    results = []
    for smiles in candidates:
        try:
            res = screen_with_dynamic_kinetics(smiles, twin)
            results.append(res)
            print(
                f"  {smiles:40s}  Homog={res.get('sei_homogeneity_scaled', 0):5.1f}/100  "
                f"(raw {res.get('sei_homogeneity_raw', 0):.4f})  "
                f"F={res.get('descriptor_num_f', 0)}  Unsat={res.get('descriptor_num_unsat', 0)}"
            )
        except Exception as e:
            print(f"  ERROR {smiles}: {e}")

    # Save results
    with open("phase6_dynamic_kinetic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to phase6_dynamic_kinetic_results.json")

    # Sort by homogeneity
    sorted_results = sorted(results, key=lambda x: x.get("sei_homogeneity_raw", 0), reverse=True)

    print(f"\n{'=' * 70}")
    print("  Top 10 by Homogeneity:")
    for r in sorted_results[:10]:
        print(
            f"    {r['smiles']:40s}  raw={r['sei_homogeneity_raw']:.4f}  "
            f"scaled={r['sei_homogeneity_scaled']:.1f}  "
            f"insulation={r.get('electronic_insulation')}  "
            f"components={r['components']}"
        )

    # Identify best candidates by different criteria
    best_homog = sorted_results[0]
    print(f"\n  Highest Homogeneity: {best_homog['smiles']} ({best_homog['sei_homogeneity_scaled']:.1f}/100)")

    # Candidates with homogeneity > 20/100 (meaningful improvement)
    improved = [r for r in sorted_results if r.get("sei_homogeneity_raw", 0) > 0.20]
    print(f"\n  Candidates with Homogeneity > 20/100: {len(improved)}/{len(results)}")
    for r in improved:
        print(f"    {r['smiles']:40s}  {r['sei_homogeneity_scaled']:.1f}/100")


if __name__ == "__main__":
    main()
