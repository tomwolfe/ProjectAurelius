"""DFT re-rank benchmark for de-circularizing discovery validation.

Re-scores the top-20 EA discoveries using an independent extended-Hückel
method (per-atom Coulomb integrals + Wolfsberg-Helmholz resonance integrals)
that does NOT reuse the TOM particle-in-a-box assumptions.

Computes Spearman rho between the Aurelius composite score and the
independent EHT re-score to verify that the physics-grounded oracle
produces chemically meaningful rankings beyond its own internal scoring.

Physical justification:
  Extended Hückel Theory (EHT) is a semi-empirical quantum chemistry
  method that uses experimentally derived ionization potentials as
  Coulomb integrals (a_X) and the Wolfsberg-Helmholz approximation
  for resonance integrals. It is fundamentally different from the TOM
  particle-in-a-box model, which treats pi-electrons as confined in
  a 1-D box. EHT captures through-bond and through-space orbital
  interactions via the overlap matrix S, providing an independent
  physical model for frontier orbital energies.

  The Coulomb integrals a_X are the valence-state ionization potentials
  for each element (in eV):
    C: -11.4, N: -13.9, O: -17.3, F: -20.0
  The Wolfsberg-Helmholz constant K = 1.75 scales the resonance
  integral relative to the average of the two Coulomb integrals.

  The overlap integral S_ij for bonded atoms is approximated as 0.25,
  a standard EHT approximation for sigma bonds.

Success criterion: Spearman rho > 0.30 (weak-but-nontrivial agreement
between two physically distinct models).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

sys.path.insert(0, SRC_DIR)

from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext

_EHT_COULOMB: dict[int, float] = {
    6: -11.4,
    7: -13.9,
    8: -17.3,
    9: -20.0,
    15: -10.0,
    16: -10.5,
}

_K_WOLFBERG_HELMHOLTZ: float = 1.75
_S_OVERLAP_BONDED: float = 0.25
_S_OVERLAP_NONBONDED: float = 0.0


def _build_eht_hamiltonian(mol: Chem.Mol) -> np.ndarray:
    """Build the extended-Hückel Hamiltonian matrix for a molecule.

    Uses per-atom Coulomb integrals (a_X) and Wolfsberg-Helmholz
    resonance integrals (K * (a_i + a_j) / 2 * S_ij).

    Returns:
        H matrix (n_atoms x n_atoms)
    """
    n = mol.GetNumAtoms()
    H = np.zeros((n, n), dtype=np.float64)
    S = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        atom_i = mol.GetAtomWithIdx(i)
        z_i = atom_i.GetAtomicNum()
        h_ii = _EHT_COULOMB.get(z_i, -11.0)
        H[i, i] = h_ii
        S[i, i] = 1.0

    bond = mol.GetBonds()
    for b in bond:
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        z_i = atom_i.GetAtomicNum()
        z_j = atom_j.GetAtomicNum()
        h_ii = _EHT_COULOMB.get(z_i, -11.0)
        h_jj = _EHT_COULOMB.get(z_j, -11.0)
        h_ij = _K_WOLFBERG_HELMHOLTZ * (h_ii + h_jj) / 2.0 * _S_OVERLAP_BONDED
        H[i, j] = h_ij
        H[j, i] = h_ij
        S[i, j] = _S_OVERLAP_BONDED
        S[j, i] = _S_OVERLAP_BONDED

    return H, S


def _compute_eht_orbitals(mol: Chem.Mol) -> tuple[float, float]:
    """Compute HOMO/LUMO energies using extended-Hückel theory.

    Solves the generalized eigenvalue problem H c = E S c and returns
    the HOMO and LUMO eigenvalues (in eV, relative to vacuum).

    Returns:
        (homo_eV, lumo_eV)
    """
    H, S = _build_eht_hamiltonian(mol)

    try:
        eigenvalues = np.linalg.eigvals(np.linalg.solve(S, H))
    except np.linalg.LinAlgError:
        eigenvalues = np.linalg.eigvalsh(H)

    eigenvalues = np.sort(eigenvalues.real)

    n_electrons = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 1:
            n_electrons += 1
        elif z == 6:
            n_electrons += 4
        elif z == 7:
            n_electrons += 5
        elif z == 8:
            n_electrons += 6
        elif z == 9:
            n_electrons += 7
        elif z == 15:
            n_electrons += 5
        elif z == 16:
            n_electrons += 6
        else:
            n_electrons += 2

    n_occ = n_electrons // 2
    homo = eigenvalues[n_occ - 1]
    lumo = eigenvalues[n_occ] if n_occ < len(eigenvalues) else homo + 3.0

    return float(homo), float(lumo)


def _load_top_discoveries(n: int = 20) -> list[dict]:
    """Load top-N discoveries from the run summary or seed pool.

    Falls back to generating candidates from seed molecules if
    no run summary with discoveries is available.
    """
    summary_path = os.path.join(PROJECT_ROOT, "run_summary.json")
    discoveries: list[dict] = []

    try:
        with open(summary_path) as f:
            summary = json.load(f)
        raw = summary.get("discoveries", [])
        for d in raw[:n]:
            if isinstance(d, dict) and "smiles" in d:
                discoveries.append(d)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if len(discoveries) >= n:
        return discoveries[:n]

    seed_path = os.path.join(
        PROJECT_ROOT, "src", "aurelius", "data", "tier0_seed_smiles.json"
    )
    try:
        with open(seed_path) as f:
            seeds = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        seeds = ["COC(=O)OC", "C1COCCO1", "CS(=O)(=O)C", "CC#N"]

    from aurelius.agent.mutation import MutationEngine
    from aurelius.pipeline import AureliusPipeline

    engine = MutationEngine(seed_smiles=seeds[:4])
    candidates = engine.propose_candidates(n_candidates=n * 3, batch_size=50)

    pipeline = AureliusPipeline()
    pipeline.initialize()

    scored: list[dict] = []
    for smi in candidates:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        try:
            result = pipeline.screen_molecule(ctx)
            score = result.get("score", {}).get("total_score", 0.0)
            scored.append({
                "smiles": smi,
                "total_score": score,
                "homo_eV": result.get("tier2", {}).get("homo_eV"),
                "lumo_eV": result.get("tier2", {}).get("lumo_eV"),
            })
        except Exception:
            continue

    scored.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
    return scored[:n]


def main() -> None:
    """Re-score top discoveries with independent EHT and compute Spearman rho."""
    print("=" * 70)
    print("DFT Re-Rank Benchmark — De-circularizing Discovery Validation")
    print("=" * 70)

    discoveries = _load_top_discoveries(n=20)
    print(f"\nTop discoveries loaded: {len(discoveries)}")

    if len(discoveries) < 5:
        print("ERROR: Need at least 5 discoveries for meaningful correlation.")
        sys.exit(1)

    pipeline = AureliusPipeline()
    pipeline.initialize()

    aurelius_scores: list[float] = []
    eht_homos: list[float] = []
    eht_lumos: list[float] = []
    eht_composites: list[float] = []
    smiles_list: list[str] = []

    for d in discoveries:
        smi = d.get("smiles", "")
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue

        try:
            pipeline_result = pipeline.screen_molecule(ctx)
            oracle_score = pipeline_result.get("score", {}).get("total_score", 0.0)
        except Exception:
            continue

        homo_eht, lumo_eht = _compute_eht_orbitals(ctx.mol)
        eht_composite = -(homo_eht + lumo_eht) / 2.0

        smiles_list.append(smi)
        aurelius_scores.append(oracle_score)
        eht_homos.append(homo_eht)
        eht_lumos.append(lumo_eht)
        eht_composites.append(eht_composite)

    n_valid = len(smiles_list)
    print(f"Valid molecules scored: {n_valid}")

    if n_valid < 5:
        print("ERROR: Need at least 5 valid molecules for correlation.")
        sys.exit(1)

    rho_composite, p_composite = spearmanr(aurelius_scores, eht_composites)
    rho_homo, p_homo = spearmanr(aurelius_scores, eht_homos)
    rho_lumo, p_lumo = spearmanr(aurelius_scores, eht_lumos)

    print(f"\n{'Metric':<30} {'Spearman rho':>14} {'p-value':>10}")
    print("-" * 56)
    print(f"{'Aurelius vs EHT composite':<30} {rho_composite:>14.4f} {p_composite:>10.4f}")
    print(f"{'Aurelius vs EHT HOMO':<30} {rho_homo:>14.4f} {p_homo:>10.4f}")
    print(f"{'Aurelius vs EHT LUMO':<30} {rho_lumo:>14.4f} {p_lumo:>10.4f}")

    print("\n" + "=" * 70)
    print("Comparison Table: Aurelius Score vs EHT Re-score")
    print("=" * 70)
    print(f"{'SMILES':<20} {'Aurelius':>10} {'EHT HOMO':>10} {'EHT LUMO':>10} {'EHT Comp':>10}")
    print("-" * 62)
    for i in range(n_valid):
        print(f"{smiles_list[i]:<20} {aurelius_scores[i]:>10.2f} {eht_homos[i]:>10.3f} {eht_lumos[i]:>10.3f} {eht_composites[i]:>10.3f}")

    assert rho_composite > 0.30, (
        f"Spearman rho between Aurelius composite and EHT re-score is "
        f"{rho_composite:.3f} (threshold: 0.30). The two models do not "
        f"agree beyond noise. This suggests the Aurelius oracle may be "
        f"ranking molecules on a dimension that the independent EHT model "
        f"does not capture, or the oracle's scoring is not physically "
        f"grounded in frontier orbital energetics."
    )

    print(f"\nSUCCESS: rho = {rho_composite:.3f} > 0.30 — independent EHT model "
          f"validates Aurelius ranking.")


if __name__ == "__main__":
    main()
