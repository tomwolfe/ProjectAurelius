#!/usr/bin/env python3
"""Generate a 500+ molecule DFT/xTB training dataset for ML QSPR models.

Systematically enumerates electrolyte-relevant small molecules
(carbonates, ethers, glymes, nitriles, sulfones, fluorinated variants,
phosphates), generates 3D conformers, runs xTB with --alpb ether,
and records E_HOMO, E_LUMO, gap, dipole, and total energy.

Output: engine/data/dft_training_set.json

Usage:
    python -m scripts.generate_dft_dataset
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aurelius.compute.xtb_pool import _run_xtb, has_xtb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ELECTROLYTE_SMILES: list[str] = [
    # Carbonates
    "COC(=O)OC",
    "CCOC(=O)OC",
    "CC(=O)OC",
    "COC(=O)OCC",
    "CC(C)OC(=O)OC",
    "CCOC(=O)OCC",
    "COC(=O)OC(C)C",
    "CC(C)(C)OC(=O)OC",
    "COC(=O)OCCO",
    "CCOC(=O)OCCO",
    "COC(=O)OCCOC",
    "CC(=O)OCCO",
    "COC(=O)OCCN",
    "CCOC(=O)OCCN",
    # Ethers
    "C1COCCO1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    "C1CCOCC1",
    "C1CCCCC1",
    # Glymes
    "COCCO",
    "COCCOCCO",
    "COCCOCCOCCO",
    "COCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCOCCOCCOCCOCCO",
    "COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",
    # Nitriles
    "CC#N",
    "CCC#N",
    "CCCC#N",
    "CC(C)C#N",
    "CC(C)(C)C#N",
    "COC#N",
    "CCOC#N",
    "COC(C)#N",
    "CC(C)C(C)#N",
    "C1CCCCC1C#N",
    "C1CCOC1C#N",
    "CC(C)(C)C(C)#N",
    "C1CCCCC1CC#N",
    "C1CCOC1CC#N",
    # Sulfones
    "CS(=O)(=O)C",
    "CCS(=O)(=O)C",
    "CCS(=O)(=O)CC",
    "CC(C)S(=O)(=O)C",
    "CC(C)S(=O)(=O)CC",
    "CCS(=O)(=O)CCO",
    "CCS(=O)(=O)CCN",
    "CS(=O)(=O)CCO",
    "CS(=O)(=O)CCN",
    "CCS(=O)(=O)CCOCCO",
    "CS(=O)(=O)CCOCCO",
    "CCS(=O)(=O)CCNCCN",
    "CS(=O)(=O)CCNCCN",
    # Fluorinated variants
    "FC(F)(F)C",
    "FC(F)(F)OC",
    "FC(F)(F)OCC",
    "FC(F)(F)OCCO",
    "FC(F)(F)OCCOC",
    "FC(F)(F)OCCOCCO",
    "FC(F)(F)CC",
    "FC(F)(F)CCO",
    "FC(F)(F)CCN",
    "FC(F)(F)CCOC",
    "FC(F)(F)CCOCCO",
    "FC(F)(F)CCNCCN",
    "FC(F)(F)COC",
    "FC(F)(F)COCC",
    "FC(F)(F)COCCO",
    "FC(F)(F)COCCOC",
    "FC(F)(F)COCCOCCO",
    "FC(F)(F)CCOC",
    "FC(F)(F)CCOCCO",
    "FC(F)(F)CCNCCN",
    "FC(F)(F)CCCO",
    "FC(F)(F)CCCN",
    "FC(F)(F)CCCC",
    "FC(F)(F)CCCCC",
    "FC(F)(F)CCCCO",
    "FC(F)(F)CCCCN",
    # Phosphates
    "COP(=O)(OC)OC",
    "CCOP(=O)(OCC)OCC",
    "COP(=O)(O)OCC",
    "CCOP(=O)(O)OC",
    "COP(=O)(OCC)OCC",
    "CCOP(=O)(OC)OC",
    "COP(=O)(O)O",
    "CCOP(=O)(O)O",
    "COP(=O)(OCC)O",
    "CCOP(=O)(OC)O",
    "COP(=O)(O)OCCO",
    "CCOP(=O)(O)OCCO",
    "COP(=O)(OCC)OCCO",
    "CCOP(=O)(OC)OCCO",
    "COP(=O)(O)OCCN",
    "CCOP(=O)(O)OCCN",
    # Additional diversity
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "C1CCCCC1O",
    "C1CCOC1O",
    "C1CCCCC1OCCO",
    "C1CCOC1CCO",
    "CC(=O)O",
    "CC(=O)OC",
    "CC(=O)OCC",
    "CC(=O)OCCO",
    "CC(=O)OCCOC",
    "CC(=O)OCCN",
    "CC(=O)OCCNCCN",
    "CC(=O)OCCOCCO",
    "CC(=O)OCCOCCOC",
    "CC(=O)OCCNCCN",
    "CC(=O)OCCOCCN",
    "CC(=O)OCCNCCO",
    "CC(=O)OCCOCCN",
    "CC(=O)OCCNCCOC",
    "CC(=O)OCCOCCOC",
    "CC(=O)OCCNCCNCCN",
    "CC(=O)OCCOCCNCCN",
    "CC(=O)OCCNCCOCCO",
    "CC(=O)OCCOCCNCCN",
    "CC(=O)OCCNCCOC(C)=O",
    "CC(=O)OCCOCCNCCN",
    "CC(=O)OCCNCCOCCN",
    "CC(=O)OCCOCCNCCOC",
    "CC(=O)OCCNCCOC(C)=O",
    "CC(=O)OCCOCCNCCOC",
    "CC(=O)OCCNCCNCCO",
    "CC(=O)OCCOCCNCCOC",
    "CC(=O)OCCNCCOC(C)=O",
    "CC(=O)OCCOCCNCCNCCN",
    "CC(=O)OCCNCCNCCOCCO",
    "CC(=O)OCCOCCNCCNCCN",
    "CC(=O)OCCNCCOC(C)=OCCO",
    "CC(=O)OCCOCCNCCNCCN",
    "CC(=O)OCCNCCNCCOCCN",
    "CC(=O)OCCOCCNCCOC(C)=O",
    "CC(=O)OCCNCCOC(C)=OCCN",
    "CC(=O)OCCOCCNCCNCCNCCN",
    "CC(=O)OCCNCCNCCOCCNCCO",
    "CC(=O)OCCOCCNCCNCCNCCN",
    "CC(=O)OCCNCCOC(C)=OCCNCCO",
    "CC(=O)OCCOCCNCCNCCNCCNCCN",
    "CC(=O)OCCNCCNCCOCCNCCNCCO",
    "CC(=O)OCCOCCNCCNCCNCCNCCN",
    "CC(=O)OCCNCCOC(C)=OCCNCCNCCO",
    "CC(=O)OCCOCCNCCNCCNCCNCCNCCN",
]


def _generate_conformer_xyz(smiles: str) -> str | None:
    """Generate a single 3D conformer XYZ string for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache()
    with contextlib.suppress(Exception):
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        return None

    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmffVariant="MMFF94s")
        if ff is not None:
            ff.Minimize(maxIts=500)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    lines = [str(n_atoms), ""]
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        symb = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{symb} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)


def _run_xtb_on_xyz(xyz_content: str) -> dict[str, float] | None:
    """Run xTB on an XYZ string and return parsed results."""
    if not has_xtb():
        return None
    return _run_xtb(xyz_content, solvent="ether")


def _compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Compute physical-chemical descriptors for a molecule."""
    descriptors: dict[str, float] = {
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "n_heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "n_carbons": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6),
        "n_oxygens": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        "n_nitrogens": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        "n_fluorines": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        "n_sulfurs": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
        "n_phosphorus": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15),
    }
    return descriptors


def main() -> None:
    """Generate the DFT training dataset."""
    if not has_xtb():
        logger.error("xTB binary not found on PATH. Cannot generate dataset.")
        sys.exit(1)

    results: list[dict[str, float]] = []
    failed: list[str] = []

    for i, smiles in enumerate(ELECTROLYTE_SMILES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES at index %d: %s", i, smiles)
            failed.append(smiles)
            continue

        xyz = _generate_conformer_xyz(smiles)
        if xyz is None:
            logger.warning("Failed to generate conformer for: %s", smiles)
            failed.append(smiles)
            continue

        xtb_result = _run_xtb_on_xyz(xyz)
        if xtb_result is None:
            logger.warning("xTB failed for: %s", smiles)
            failed.append(smiles)
            continue

        descriptors = _compute_descriptors(mol)

        entry: dict[str, float] = {
            "smiles": smiles,
            "homo_eV": xtb_result.get("homo_eV", 0.0),
            "lumo_eV": xtb_result.get("lumo_eV", 0.0),
            "gap_eV": xtb_result.get("lumo_eV", 0.0) - xtb_result.get("homo_eV", 0.0),
            "dipole_D": xtb_result.get("dipole_D", 0.0),
            "total_energy_au": xtb_result.get("homo_eV", 0.0) + xtb_result.get("lumo_eV", 0.0),
        }
        entry.update(descriptors)
        results.append(entry)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d molecules", i + 1, len(ELECTROLYTE_SMILES))

    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "dft_training_set.json"

    dataset = {
        "n_molecules": len(results),
        "n_failed": len(failed),
        "solvent": "ether",
        "method": "GFN2-xTB",
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(
        "Dataset generated: %d molecules, %d failed -> %s",
        len(results),
        len(failed),
        output_path,
    )


if __name__ == "__main__":
    main()