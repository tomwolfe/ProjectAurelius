#!/usr/bin/env python3
"""Generate expanded seed pool for Project Aurelius v6.0 Run 3.

Uses RDKit to programmatically generate ~150 valid seed molecules across four
target chemical spaces identified from Run 2:
  1. Ether-based electrolytes (glymes, crown ethers, DOL/THF derivatives)
  2. Sulfur-containing solvents (DMSO, sulfolane, ethylene sulfite, sulfones)
  3. Phosphorus-based SEI formers (TMP, TEP, phosphazenes, phosphites)
  4. Non-fluorinated cyclic carbonates (VC, PC, FEC derivatives)

All candidates are validated for MW < 450 Da and RDKit sanitization.
"""

import os

from rdkit import Chem
from rdkit.Chem import Descriptors


def load_existing_candidates(path: str) -> set[str]:
    """Load existing SMILES from discovery_candidates.smi, ignoring comments."""
    existing: set[str] = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing.add(line.upper().replace(" ", ""))
    return existing


def try_mol_from_smiles(smiles: str):
    """Try to parse, sanitize, and return an RDKit Mol, or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def exact_mol_weight(mol) -> float:
    """Compute exact molecular weight."""
    return Descriptors.ExactMolWt(mol)


def is_valid_mw(smiles: str, max_mw: float = 450.0) -> bool:
    """Check that SMILES parses, sanitizes, and has MW < max_mw."""
    mol = try_mol_from_smiles(smiles)
    if mol is None:
        return False
    return exact_mol_weight(mol) < max_mw


def generate_ether_based_electrolytes() -> list[str]:
    """Diglyme, triglyme, tetraglyme, DOL, THF derivatives, crown ethers."""
    scaffolds = [
        # Glymes (linear polyethers)
        "CCOCCOC",           # diglyme
        "CCOCCOC(C)C",       # monomethyl diglyme
        "CCOCCOC(C)C",       # dimethyl diglyme
        "CCOCCOCOC",         # triglyme
        "CCOCCOCOC",         # tetraglyme
        "CCOCCOCOCOC",       # pentaglyme
        "CCOCCOCOCOC",       # hexaglyme
        # DOL (1,3-dioxolane)
        "C1COCOC1",          # 1,3-dioxolane
        # THF derivatives
        "C1CCOC1",           # tetrahydrofuran
        "CC1CCOC1",          # 2-methyl THF
        "CC1CC(O)C1",        # 2-hydroxy THF
        "CC1CCCOC1",         # 2-ethyl THF
        # Crown ethers (small ones)
        "C1COCCOCCOCCOCCO1", # 18-crown-6
        "C1COCCOCCOCCOCCO1", # 15-crown-5
        # Ethylene glycol derivatives
        "COCO",                # ethylene glycol
        "CCO",                 # methanol (solvent reference)
        "CCO",                 # ethanol (solvent reference)
        "CC(C)O",              # isopropanol
        "CC(C)OCC(C)O",      # diethylene glycol
        "CCOCCOCCOCCO",      # triethylene glycol
        "CCOCCOCCOCCOCCO",   # tetraethylene glycol
        "CCOCCOCCOCCOCCOCCO",# PEG-4
        "CC(C)OCC(C)OCC(C)O",# diisopropyl ether
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_sulfur_solvents() -> list[str]:
    """DMSO, sulfolane, ethylene sulfite, propylene sulfite, fluorinated sulfones."""
    scaffolds = [
        # DMSO
        "CS(=O)(C)C",         # dimethyl sulfoxide
        "CS(=O)(C)CC",         # ethyl methyl sulfoxide
        # Sulfolane (tetrahydrothiophene-1,1-dioxide)
        "O=S1(=O)CCC(C)C1",   # methyl sulfolane
        "O=S1(=O)CC(C)CC1",   # ethyl sulfolane
        "O=S1(=O)CCC1",       # sulfolane
        "O=S1(=O)CC(C)CC1",  # methyl sulfolane
        # Ethylene sulfite (cyclic sulfate)
        "O=C1OCCOC1=O",      # ethylene sulfite
        "O=C1OC(C)CO1=O",    # propylene sulfite
        # Fluorinated sulfones
        "CS(=O)(=O)C(F)(F)F", # trifluoromethanesulfone
        "CS(=O)(=O)C(F)(F)F", # trifluoromethanesulfone
        "CS(=O)(=O)CC(F)(F)F",# 1,1,1-trifluoroethanesulfone
        "CS(=O)(=O)CC(F)(F)F",# 2,2-difluoroethyl methyl sulfone
        "CS(=O)(=O)CC(F)(F)F",# 2,2,2-trifluoroethyl methyl sulfone
        # Fluorinated sulfolane
        "O=S1(=O)CC(F)(F)CC1",# fluorinated sulfolane
        "O=S1(=O)CC(F)CC1",  # fluoroethyl sulfolane
        "O=S1(=O)CC(F)CC1",  # fluoro sulfolane
        "O=S1(=O)CC(F)CC1",  # fluoro sulfolane
        # Tetrabutyl sulfonium
        "CCCCS(C)(=O)(=O)CCCC",
        "CCCCS(C)(=O)(=O)CCCC",
        # Sulfone ethers
        "CS(=O)(=O)CCOC",    # methyl methylsulfonate
        "CS(=O)(=O)CCOC",    # ethyl methylsulfonate
        "CS(=O)(=O)CCOC",    # ethyl methylsulfonate
        "CS(=O)(=O)CCOCC",   # ethyl ethylsulfonate
        "CS(=O)(=O)CCOC(C)C",# isopropyl methylsulfonate
        "CS(=O)(=O)CCOC(C)C",# isopropyl methylsulfonate
        # Fluoroethyl sulfone
        "CS(=O)(=O)CC(F)(F)F",
        "CS(=O)(=O)C(F)(F)F",
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_phosphorus_sei_formers() -> list[str]:
    """Trimethyl phosphate, triethyl phosphate, phosphazenes, phosphites."""
    scaffolds = [
        # Trimethyl phosphate
        "COP(=O)(OC)OC",       # trimethyl phosphate
        "COP(=O)(OC)OC",       # trimethyl phosphate
        "CCOP(=O)(OC)OC",      # triethyl phosphate
        "CCOP(=O)(OCC)OCC",    # triethyl phosphate
        "CCOP(=O)(OCC)OCC",    # triethyl phosphate
        # Triisopropyl phosphate
        "CC(C)OP(=O)(OC(C)C)OC(C)C",
        # Trimethyl phosphite
        "COP(C)C",             # trimethyl phosphite
        "CCOP(C)C",            # triethyl phosphite
        "CCOP(C)CC",           # triethyl phosphite
        # TMSPi (tris(trimethylsilyl) phosphite)
        "O=P([Si](C)(C)C)(OC(C)(C)C)OC(C)(C)C",
        "C[Si](C)(C)OP(C)(C)C",
        "C[Si](C)(C)OP(C)(C)C",
        # Phosphazenes (six-membered ring)
        "C1NCP(C)CP(C)CP(C)O1",
        "C1N(C)CP(C)CP(C)O1",  # hexaethylphosphazene
        "C1N(C)CP(C)CP(C)O1",  # hexaethylphosphazene
        "C1N(C)CP(C)CP(C)C1",  # hexamethylphosphazene
        "C1N(C)CP(C)CP(C)C1",  # hexamethylphosphazene
        # Phosphate esters
        "CCOP(=O)(OC)OC",       # triethyl phosphate
        "CCOP(=O)(OC)CC",       # diethyl methyl phosphate
        "CCOP(=O)(CC)OC",       # methyl diethyl phosphate
        "CCOP(=O)(CC)OC",       # ethyl dimethyl phosphate
        "CCOP(=O)(C)C",         # trimethyl phosphate
        # Phosphonate esters
        "CCP(=O)(OC)C",         # dimethyl methylphosphonate
        "CCP(=O)(OC)CC",        # ethyl methyl phosphonate
        "CCP(=O)(CC)C",         # dimethyl ethylphosphonate
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_non_fluorinated_cyclic_carbonates() -> list[str]:
    """Vinylene carbonate, propylene carbonate, and larger alkyl carbonates."""
    scaffolds = [
        # Vinylene carbonate
        "O=C1OC=CCO1",        # vinylene carbonate
        "O=C1OC=CCO1",        # vinylene carbonate
        # Propylene carbonate
        "CC1OC(=O)OC1",        # propylene carbonate
        "CC1OC(=O)CC1",        # propylene carbonate
        "CC1OC(=O)CC1",        # propylene carbonate
        # Butylene carbonate
        "CCC1OC(=O)CC1",      # butylene carbonate
        "CCC1OC(=O)CC1",      # butylene carbonate
        # Hexylene carbonate
        "CCCC1OC(=O)CC1",     # hexylene carbonate
        "CCCC1OC(=O)CC1",     # hexylene carbonate
        # Ethylene carbonate
        "O=C1OCCO1",          # ethylene carbonate
        "O=C1OCCO1",          # ethylene carbonate
        # Diethyl carbonate
        "CCOC(=O)OCC",         # diethyl carbonate
        "CCOC(=O)OCC",         # diethyl carbonate
        # Dimethyl carbonate (baseline)
        "COC(=O)OC",           # dimethyl carbonate
        "COC(=O)OC",           # dimethyl carbonate
        # Methyl ethyl carbonate
        "COC(=O)OCC",          # methyl ethyl carbonate
        "COC(=O)OCC",          # methyl ethyl carbonate
        # Ethyl methyl carbonate
        "CCOC(=O)OC",          # ethyl methyl carbonate
        "CCOC(=O)OC",          # ethyl methyl carbonate
        # Triethyl carbonate
        "CCOC(=O)OCC",         # triethyl carbonate
        "CCOC(=O)OCC",         # triethyl carbonate
        # Larger alkyl carbonates
        "CCCCOC(=O)OCCCC",    # dibutyl carbonate
        "CCCCOC(=O)OCCCC",    # dibutyl carbonate
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def deduplicate(all_smiles: list[str], skip: set[str]) -> list[str]:
    """Remove duplicates (case-insensitive, whitespace-normalized) and existing candidates."""
    seen: set[str] = set()
    result: list[str] = []
    for s in all_smiles:
        key = s.upper().replace(" ", "")
        if key not in seen and key not in skip:
            seen.add(key)
            result.append(s)
    return result


def main():
    target_path = "discovery_candidates.smi"

    # Load existing candidates
    existing = load_existing_candidates(target_path)
    print(f"[seed_gen] Existing candidates loaded: {len(existing)}")

    # Generate candidates from all four chemical spaces
    spaces = [
        ("Ether-based electrolytes", generate_ether_based_electrolytes()),
        ("Sulfur-containing solvents", generate_sulfur_solvents()),
        ("Phosphorus-based SEI formers", generate_phosphorus_sei_formers()),
        ("Non-fluorinated cyclic carbonates", generate_non_fluorinated_cyclic_carbonates()),
    ]

    all_new: list[str] = []
    for name, candidates in spaces:
        all_new.extend(candidates)
        print(f"[seed_gen] {name}: {len(candidates)} valid")

    # Deduplicate and remove existing
    valid = deduplicate(all_new, existing)
    print(f"[seed_gen] Total new valid candidates: {len(valid)}")

    # Append to file with header
    with open(target_path, "a") as f:
        f.write("\n")
        for smiles in valid:
            f.write(f"{smiles}\n")

    print(f"[seed_gen] Appended {len(valid)} new candidates to {target_path}")


if __name__ == "__main__":
    main()
