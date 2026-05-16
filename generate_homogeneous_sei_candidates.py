"""Phase 1: Generate homogeneity-targeted SEI candidates.

Strategy: Create novel molecules designed to promote multi-pathway decomposition
(solvent + salt + polymerization) for improved SEI homogeneity.

Does NOT modify force_field_params.json - physics engine remains calibrated
to literature values. Instead, generates molecules with structural features
known to promote hybrid SEI formation.

Scaffold categories:
1. Dual-functional additives (fluorinated groups + polymerizable unsaturation)
2. Borate esters (stable B-O centers, cyclic borates)
3. Sulfone-nitrile hybrids (high-voltage stability + strong adsorption)
4. Asymmetric carbonates (altered decomposition kinetics)
"""

import os
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORCE_FIELD_PATH = os.path.join(
    BASE_DIR, "src", "aurelius", "data", "force_field_params.json"
)
DISCOVERY_CANDIDATES_SMI = os.path.join(BASE_DIR, "discovery_candidates.smi")
OUTPUT_SMI = os.path.join(BASE_DIR, "homogeneity_targeted_candidates.smi")
RATIONALE_MD = os.path.join(BASE_DIR, "generation_rationale_v2.md")


def load_existing_candidates(path: str) -> list[str]:
    """Load existing SMILES from a .smi file, skipping comments."""
    candidates = []
    if not os.path.isfile(path):
        return candidates
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if parts:
                candidates.append(parts[0])
    return candidates


def compute_ecfp4_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute ECFP4 (radius=2) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two bit-vector fingerprints."""
    if fp1 is None or fp2 is None:
        return 1.0  # Treat invalid as maximally similar (fail novelty)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def molecular_weight_ok(smiles: str) -> bool:
    """Check MW < 350 g/mol."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Descriptors.ExactMolWt(mol) < 350.0


def generate_dual_functional_additives() -> list[str]:
    """Category 1: Dual-functional additives with fluorinated groups AND polymerizable unsaturated bonds."""
    candidates = [
        # Fluoro-allyl methyl carbonate - F for salt reduction, C=C for polymerization
        "COC(=O)OCC=C",
        "COC(=O)OC(F)C=C",
        "COC(=O)OCC(F)=CF2",
        # Difluoro-vinyl carbonate with ether linkage
        "COC(=O)OC(F)=CF",
        # Trifluoro-allyl ethyl carbonate
        "CCOC(=O)OCC(F)=CF2",
        # Fluoro-acrylate ester
        "COC(=O)C(F)=C(F)F",
        # Vinyl fluorosulfonate analog
        "FCC(F)(F)S(=O)(=O)OC=C",
        # Perfluoro-vinyl ether with carbonate
        "COC(=O)OC(F)=C(F)C(F)(F)F",
    ]
    return candidates


def generate_borate_esters() -> list[str]:
    """Category 2: Cyclic borate esters (stable B-O centers, not direct C-B)."""
    candidates = [
        # Cyclic borate - 5-membered ring with two fluorinated ethyl groups
        "O1B(OCC(F)F)(OCC(F)F)OC1",
        # Cyclic borate with one fluorinated and one regular chain
        "O1B(OCC(F)F)(OCC)OC1",
        # Boroxine derivative (B3O3 ring) with fluorinated substituents
        "B1OB(OB(OCC(F)F)(OCC(F)F))O1",
        # Cyclic borate with trifluoroethyl groups
        "O1B(OCC(F)(F)F)(OCC(F)(F)F)OC1",
        # Borate ester with methoxy and fluoromethyl groups
        "COC(=O)OB1OC(C(F)F)(OCC(F)F)O1",
    ]
    return candidates


def generate_sulfone_nitrile_hybrids() -> list[str]:
    """Category 3: Sulfone-nitrile hybrids combining high-voltage stability with strong adsorption."""
    candidates = [
        # DMSO with nitrile substituent
        "N#CCS(=O)(=O)C",
        # Bis(nitrile) sulfone
        "N#CCS(=O)(=O)CC#N",
        # Fluorinated nitrile-sulfone
        "N#CC(F)S(=O)(=O)C",
        # Carbonate-linked sulfone-nitrile
        "N#CCOC(=O)OC(F)S(=O)(=O)C",
        # Trifluoromethyl sulfone with nitrile
        "N#CCS(=O)(=O)C(F)(F)F",
        # Nitrile-substituted sulfone with ether linkage
        "N#CCOCCS(=O)(=O)CC#N",
    ]
    return candidates


def generate_asymmetric_carbonates() -> list[str]:
    """Category 4: Asymmetric carbonates with fluorinated substituents."""
    candidates = [
        # Fluoro-ethyl methyl carbonate
        "COC(=O)OCC(F)F",
        # Trifluoro-isopropyl methyl carbonate
        "COC(=O)OC(C)(F)F",
        # Difluoro-ethyl ethyl carbonate
        "CCOC(=O)OCC(F)F",
        # Fluoro-methyl ethyl carbonate
        "CCOC(=O)OC(F)F",
        # Perfluoro-tert-butyl methyl carbonate
        "COC(=O)OC(F)(F)C(F)(F)F",
        # Fluoro-propyl carbonate with methyl
        "COC(=O)OCC(F)(F)C(F)F",
        # Vinyl fluoride carbonate
        "COC(=O)OC=C(F)F",
        # Fluoro-ethyl ethyl carbonate (different asymmetry)
        "CCOC(=O)OCC(F)F",
    ]
    return candidates


def main():

    print("=" * 60)
    print("  AURELIUS v5.2 - Homogeneity-Targeted Candidate Generation")
    print("=" * 60)

    # Load existing discovery candidates for novelty check
    existing_smiles = load_existing_candidates(DISCOVERY_CANDIDATES_SMI)
    print(f"\nLoading {len(existing_smiles)} existing discovery candidates for novelty check.")

    # Build fingerprints of existing candidates
    existing_fps = {}
    for smi in existing_smiles:
        fp = compute_ecfp4_fingerprint(smi, radius=2, n_bits=2048)
        existing_fps[smi] = fp

    # Generate candidates from all categories
    all_generated = []
    rationale_entries = []

    categories = {
        "Dual-Functional Additives (F + C=C)": generate_dual_functional_additives(),
        "Cyclic Borate Esters (B-O centers)": generate_borate_esters(),
        "Sulfone-Nitrile Hybrids": generate_sulfone_nitrile_hybrids(),
        "Asymmetric Fluoro-Carbonates": generate_asymmetric_carbonates(),
    }

    for category_name, raw_smiles in categories.items():
        print(f"\n--- {category_name} ---")
        for smi in raw_smiles:
            # RDKit sanitization check
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"  SKIP (invalid SMILES): {smi}")
                continue

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                print(f"  SKIP (sanitization failed): {smi}")
                continue

            # MW check
            if not molecular_weight_ok(smi):
                mw = Descriptors.ExactMolWt(mol)
                print(f"  SKIP (MW={mw:.1f} >= 350): {smi}")
                continue

            # Novelty check: Tanimoto < 0.75 against all existing candidates
            new_fp = compute_ecfp4_fingerprint(smi, radius=2, n_bits=2048)
            max_sim = 0.0
            for _existing_smi, existing_fp in existing_fps.items():
                sim = tanimoto_similarity(new_fp, existing_fp)
                if sim > max_sim:
                    max_sim = sim

            if max_sim >= 0.75:
                print(f"  SKIP (similarity {max_sim:.3f} >= 0.75): {smi}")
                continue

            # Passed all filters
            all_generated.append(smi)
            rationale_entries.append({
                "smiles": smi,
                "category": category_name,
                "max_similarity": max_sim,
                "mw": float(Descriptors.ExactMolWt(mol)),
            })
            print(f"  ACCEPT: {smi} (MW={Descriptors.ExactMolWt(mol):.1f}, max_sim={max_sim:.3f})")

    # Save to SMILES file
    with open(OUTPUT_SMI, "w") as f:
        f.write("# Homogeneity-targeted candidates - Aurelius v5.2\n")
        f.write("# Generated: multi-pathway decomposition strategy\n")
        f.write(f"# Total candidates: {len(all_generated)}\n\n")
        for smi in all_generated:
            f.write(f"{smi}\n")

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(all_generated)} candidates to {OUTPUT_SMI}")
    print(f"{'=' * 60}")

    # Generate rationale markdown
    with open(RATIONALE_MD, "w") as f:
        f.write("# Generation Rationale v2 - Homogeneity-Targeted SEI Candidates\n\n")
        f.write("## Strategy Overview\n\n")
        f.write("The baseline screen of 8 standard candidates showed SEI homogeneity\n")
        f.write("scores of ~12.2/100 (raw ~0.12), indicating that solvent decomposition\n")
        f.write("overwhelmingly dominates the kMC simulation. This produces a brittle,\n")
        f.write("heterogeneous SEI layer.\n\n")
        f.write("This generation targets **multi-pathway decomposition** by designing\n")
        f.write("molecules with structural features that promote balanced reaction\n")
        f.write("rates across all three kMC pathways:\n\n")
        f.write("1. **Solvent Decomposition** (EC/DMC reduction at anode)\n")
        f.write("2. **Salt Reduction** (PF6- decomposition)\n")
        f.write("3. **Polymerization** (organic SEI formation)\n\n")
        f.write("## Scaffold Categories\n\n")

        for entry in rationale_entries:
            f.write(f"### {entry['smiles']}\n\n")
            f.write(f"- **Category:** {entry['category']}\n")
            f.write(f"- **Molecular Weight:** {entry['mw']:.1f} g/mol\n")
            f.write(f"- **Max Tanimoto Similarity:** {entry['max_similarity']:.3f}\n\n")

            if "Dual-Functional" in entry["category"]:
                f.write("**Multi-pathway rationale:** This molecule contains both\n")
                f.write("a fluorinated group (lowering the effective barrier for salt\n")
                f.write("reduction by stabilizing F- intermediates) AND a C=C double bond\n")
                f.write("(providing a low-barrier pathway for polymerization). The dual\n")
                f.write("functional design ensures that salt reduction and polymerization\n")
                f.write("rates increase relative to pure solvent decomposition, pushing\n")
                f.write("the reaction distribution closer to the ideal 1/3:1/3:1/3 split.\n\n")

            elif "Borate" in entry["category"]:
                f.write("**Multi-pathway rationale:** Cyclic borate esters decompose via\n")
                f.write("B-O bond cleavage (lower Ea than C-B direct bonds), producing\n")
                f.write("fluorinated boron species that interact with PF6- salt anions.\n")
                f.write("The fluorinated alkyl chains provide additional pathways for\n")
                f.write("salt reduction. The ring-opening mechanism creates reactive\n")
                f.write("intermediates that can participate in polymerization, increasing\n")
                f.write("the relative rate of the polymerization pathway.\n\n")

            elif "Sulfone" in entry["category"]:
                f.write("**Multi-pathway rationale:** The sulfone group provides high\n")
                f.write("voltage stability (resisting early decomposition), while the\n")
                f.write("nitrile group provides strong adsorption to the anode surface.\n")
                f.write("This dual nature means the molecule contributes to both solvent\n")
                f.write("decomposition (via the sulfone framework) AND salt interaction\n")
                f.write("(via nitrile-anion complexation). Fluorinated variants further\n")
                f.write("enhance salt reduction events through F-stabilized intermediates.\n\n")

            elif "Asymmetric" in entry["category"]:
                f.write("**Multi-pathway rationale:** Breaking symmetry in carbonates\n")
                f.write("alters the decomposition kinetics compared to symmetric analogs.\n")
                f.write("Fluorinated asymmetric carbonates have lower activation barriers\n")
                f.write("for salt reduction (due to F stabilization of transition states)\n")
                f.write("while the carbonate backbone still supports solvent decomposition.\n")
                f.write("The asymmetry also creates multiple distinct decomposition\n")
                f.write("pathways, increasing the probability of salt and polymerization\n")
                f.write("events relative to the baseline symmetric carbonates.\n\n")

        f.write("## Novelty Validation\n\n")
        f.write(f"All {len(all_generated)} candidates passed Tanimoto similarity < 0.75\n")
        f.write("against the 8 baseline discovery candidates using ECFP4 fingerprints\n")
        f.write("(radius=2, 2048 bits), ensuring structural diversity.\n\n")
        f.write("## Filters Applied\n\n")
        f.write("- Molecular Weight < 350 g/mol\n")
        f.write("- RDKit sanitization (valid valence, aromaticity)\n")
        f.write("- Tanimoto similarity < 0.75 vs. baseline candidates (ECFP4)\n\n")

    print(f"Saved rationale to {RATIONALE_MD}")

    return all_generated


if __name__ == "__main__":
    generated = main()
    if not generated:
        print("\nERROR: No candidates passed filters. Check generation logic.")
        sys.exit(1)
    print(f"\nPhase 1 complete: {len(generated)} homogeneity-targeted candidates ready for screening.")
