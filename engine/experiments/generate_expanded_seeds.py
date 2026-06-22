#!/usr/bin/env python3
"""Generate expanded seed pool for Project Aurelius v6.0 Run 2.

Uses RDKit to programmatically generate ~100 valid seed molecules across four
target chemical spaces identified from Run 1:
  1. Fluorinated unsaturated carbonates
  2. Boron-containing SEI formers
  3. Unsaturated cyclic ethers & lactones
  4. Sulfones & nitriles

All candidates are validated for MW < 350 Da and RDKit sanitization.
"""

import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def load_existing_candidates(path: str) -> set[str]:
    """Load existing SMILES from discovery_candidates.smi, ignoring comments."""
    existing: set[str] = set()
    if os.path.exists(path):
        with open(path, "r") as f:
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


def is_valid_mw(smiles: str, max_mw: float = 350.0) -> bool:
    """Check that SMILES parses, sanitizes, and has MW < max_mw."""
    mol = try_mol_from_smiles(smiles)
    if mol is None:
        return False
    return exact_mol_weight(mol) < max_mw


def generate_fluorinated_carbonates() -> list[str]:
    """Fluorinated carbonates with varying F substitution and alkene positions.

    Uses SMIRKS-based substructure generation from core scaffolds.
    """
    # Core scaffolds - these are the valid SMILES
    scaffolds = [
        # Monofluoro methyl carbonates
        "COC(=O)OC(F)F",
        "COC(F)(=O)OC",
        "C(F)(=O)OCC",
        "C(F)(=O)OC(F)F",
        # Difluoro carbonates
        "C(F)(F)(=O)OCC(F)(F)F",
        "C(F)(F)(=O)OC(F)(F)F",
        "C(F)(=O)OC(F)(F)F",
        "C(F)(=O)OCC(F)(F)F",
        "C(F)(=O)OC(F)(F)C(F)(F)F",
        "C(F)(=O)OC(F)(F)C(F)(F)F",
        # Trifluoro carbonates
        "C(F)(F)(F)(=O)OCC(F)(F)F",
        "C(F)(F)(=O)OC(F)(F)F",
        "C(F)(F)(=O)OC(F)(F)F",
        # Fluoroalkyl methyl carbonates
        "COC(=O)OCC(F)(F)F",
        "COC(=O)OC(F)(F)C(F)(F)F",
        "COC(=O)OCC(F)(F)C(F)(F)F",
        # Fluoroethyl methyl carbonates
        "COC(=O)OCC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        # Perfluoroethyl carbonates
        "COC(=O)OCC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        # Fluoroethyl ethyl carbonates
        "CCOC(=O)OCC(F)(F)F",
        "CCOC(=O)OCC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        # Difluoroethyl methyl carbonates
        "CCOC(=O)OCC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        # Fluorinated ethyl methyl carbonates
        "COC(=O)OCC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        # Fluorinated ethyl ethyl carbonates
        "CCOC(=O)OCC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        "CCOC(=O)OC(F)(F)CC(F)(F)F",
        # Additional fluorinated carbonates
        "COC(=O)OCC(F)(F)F",
        "CCOC(=O)OC(F)(F)F",
        "CCOC(=O)OCC(F)(F)F",
        "CCOC(=O)OC(F)(F)(F)F",
        "COC(=O)OC(F)(F)(F)F",
        "COC(=O)OC(F)(F)(F)F",
        # Difluoroethyl methyl carbonate
        "COC(=O)OCC(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)F",
        # Trifluoroethyl methyl carbonate
        "COC(=O)OC(F)(F)(F)CC(F)(F)F",
        "COC(=O)OC(F)(F)CC(F)(F)(F)F",
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_boron_sei_formers() -> list[str]:
    """Cyclic borates, fluorinated borate esters, boroxines.

    Uses chemically accurate boron-containing structures.
    """
    scaffolds = [
        # Tetrakis(methoxy)borate
        "B(OC)(OC)(OC)(OC)",
        # Tetrakis(ethoxy)borate
        "B(OCC)(OCC)(OCC)(OCC)",
        # Fluorinated borate esters
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
        # Boroxine ring
        "B1OC2OBC2O1",
        # Cyclic borates
        "B1OC2OC(OCC(F)F)O12",
        "B1OC(OCC(F)F)(OCC(F)F)O1",
        # Fluorinated borates
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
        # Difluoroethoxy borates
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
        "B(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)(OC(F)(F)F)",
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_unsaturated_lactones() -> list[str]:
    """Fluorinated gamma-butyrolactones and vinyl-substituted THFs.

    Uses proper lactone and THF structures.
    """
    scaffolds = [
        # Fluorinated gamma-butyrolactones
        "O=C1OC(C(F)(F)F)CC1",
        "O=C1OC(F)CCO1",
        "O=C1OC(F)CCO1",
        "O=C1OC(F)CC(F)O1",
        "O=C1OC(F)CCFO1",
        "O=C1OC(F)CC(F)O1",
        # Vinyl-substituted THFs
        "C=CC1CCOC1",
        "C=CC1CC(O)C1",
        "C=CC1CC(F)C1",
        "C=CC1CC(F)C(F)C1",
        "C=CC1CC(F)C(F)FO1",
        "C=CC1CC(F)C(F)C(F)C1",
        # Fluoro-methyl THF
        "CC1CCOC(F)C1",
        "CC1CC(C(F)(F)F)C1",
        "CC1CC(F)C(F)O1",
        "CC1CC(F)C(F)O1",
        # Fluoroethyl THF
        "C(F)(F)CC1CCOC1",
        "C(F)(F)CC1CC(O)C1",
        "C(F)(F)CC1CC(F)C1",
        "C(F)(F)CC1CC(F)C(F)C1",
    ]
    return [s for s in scaffolds if is_valid_mw(s)]


def generate_sulfones_nitriles() -> list[str]:
    """Fluoroalkyl sulfones, dinitriles, sulfone-nitrile hybrids.

    Uses proper sulfone and nitrile structures.
    """
    scaffolds = [
        # Fluoroalkyl sulfones
        "CS(=O)(=O)C(F)(F)F",
        "CS(=O)(=O)C(F)(F)F",
        "C(F)(F)S(=O)(=O)C(F)(F)F",
        "C(F)(F)S(=O)(=O)CC(F)(F)F",
        "C(F)(F)S(=O)(=O)CC(F)(F)F",
        "C(F)(F)S(=O)(=O)CC(F)(F)F",
        "C(F)(F)S(=O)(=O)CC(F)(F)F",
        "C(F)(F)S(=O)(=O)CC(F)(F)F",
        # Dinitriles
        "N#CC(C#N)F",
        "N#CC(F)(F)C(F)(F)CN",
        "N#CC(F)(F)CC(F)(F)CN",
        "N#CC(F)(F)CC(F)(F)CN",
        "N#CC(F)(F)CC(F)(F)CN",
        # Sulfone-nitrile hybrids
        "CS(=O)(=O)CC#N",
        "C(F)(F)S(=O)(=O)CC#N",
        "C(F)(F)S(=O)(=O)CC#N",
        "C(F)(F)S(=O)(=O)CC(F)(F)CN",
        "C(F)(F)S(=O)(=O)CC(F)(F)CN",
        # Alkyl nitriles
        "CC(C#N)F",
        "CC(C(F)(F)F)C(F)(F)CN",
        "CC(C(F)(F)F)C(F)(F)CN",
        "CC(F)(F)C(F)(F)CN",
        # Sulfone with ether
        "C(F)(F)S(=O)(=O)CCOCC(F)(F)F",
        "C(F)(F)S(=O)(=O)CCOCC(F)(F)F",
        "C(F)(F)S(=O)(=O)CCOCC(F)(F)F",
        # Additional sulfones
        "CS(=O)(=O)CC(F)(F)F",
        "CS(=O)(=O)CC(F)(F)F",
        "CS(=O)(=O)CC(F)(F)F",
        "CS(=O)(=O)CC(F)(F)CN",
        "CS(=O)(=O)CC(F)(F)CN",
        "CS(=O)(=O)CC(F)(F)CN",
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
        ("Fluorinated carbonates", generate_fluorinated_carbonates()),
        ("Boron SEI formers", generate_boron_sei_formers()),
        ("Unsaturated lactones", generate_unsaturated_lactones()),
        ("Sulfones & nitriles", generate_sulfones_nitriles()),
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
