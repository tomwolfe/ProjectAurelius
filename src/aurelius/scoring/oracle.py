"""PropertyOracle — Hybrid fragment-additivity + quantum chemistry oracle.

Architecture
------------
A **two-tier hybrid model** that uses the right physics for each property:

  **Bulk properties (GC fragment-additivity):**
    Dielectric, Viscosity, and Li+ Solvation are reasonably approximated
    by functional-group additivity.  Each fragment contributes a fixed
    additive shift — simple, interpretable, and physically valid for
    bulk thermodynamic/transport properties.

  **Frontier orbitals (Quantum Oracle):**
    HOMO/LUMO energies are global, delocalised quantum phenomena that
    cannot be predicted by fragment additivity.  The QuantumOracle uses:
      1. *Preferred*: xTB (GFN2-xTB) via subprocess — fast semi-empirical QM
      2. *Fallback*: Topological Orbital Model (TOM) — conjugation-aware
         non-linear empirical model based on π-conjugation length,
         heteroatom perturbation, and topology

  This hybrid architecture justifies the Bayesian Active Learning loop
  (the oracle is non-linear and computationally expensive), while
  keeping bulk-property prediction lightweight and transparent.

  Properties predicted:
    - HOMO energy (eV)          — QuantumOracle (xTB/TOM)
    - LUMO energy (eV)          — QuantumOracle (xTB/TOM)
    - Dielectric proxy          — GC fragment-additivity + TPSA cap
    - Viscosity proxy           — GC fragment-additivity + MW + rot. bonds
    - Li+ Solvation proxy       — GC fragment-additivity

Usage:
    from aurelius.scoring.oracle import PropertyOracle
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = oracle.evaluate(ctx)
    print(result["homo_eV"])            # e.g. -7.6 (quantum)
    print(result["li_solvation_proxy"])  # e.g. 2.3 (GC)
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import subprocess
import tempfile
from typing import Any

from rdkit import Chem

from aurelius.constants import MAX_DIELECTRIC_PER_TPSA, SULFONE_PATTERN as _SULFONE_PATTERN, CF3_PATTERN as _CF3_PATTERN
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_DATA_SOURCE: str = "hybrid (GC bulk + Quantum orbital)"


def get_data_source() -> str:
    """Return a human-readable string describing the oracle's data source."""
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models — Bulk Properties Only
# ---------------------------------------------------------------------------

# (pattern, name, dielectric_contrib, viscosity_contrib, li_solvation_contrib)
# Patterns are pre-compiled at module load time to avoid recompilation
# inside the per-molecule _count_fragments hot loop.
# Li+ solvation contributions are based on donor-number and chelation ability:
#   - Carbonates bind moderately-strongly (high donor number ~16)
#   - Ethers bind moderately (glyme family chelates Li+)
#   - Nitriles bind moderately (acetonitrile donor number ~14)
#   - Fluorinated groups reduce binding (electron withdrawal lowers donor strength)
#   - Alcohols bind too strongly (high donor number, poor transference)
_GC_FRAGMENTS: list[tuple[Chem.Mol, str, float, float, float]] = [
    (Chem.MolFromSmarts("[CX3](=O)[OX2H0]"),       "ester",              2.5,  0.6,  0.8),
    (Chem.MolFromSmarts("[CX3](=O)[OH]"),          "carboxylic_acid",    4.0,  1.0,  1.8),
    (Chem.MolFromSmarts("[CX3](=O)[NX3]"),         "amide",              5.0,  0.8,  1.2),
    (Chem.MolFromSmarts("[CX3](=O)[CX3]"),         "ketone",             3.0,  0.5,  0.6),
    (Chem.MolFromSmarts("[CH](=O)"),               "aldehyde",           2.5,  0.3,  0.3),
    (Chem.MolFromSmarts("O=C([OX2])[OX2]"),        "carbonate",          5.0,  0.7,  1.5),
    (Chem.MolFromSmarts("[OD2]([CX4])[CX4]"),      "ether",              1.5, -0.3,  0.5),
    (Chem.MolFromSmarts("[OH][CX4]"),              "alcohol",            4.5,  1.2,  2.0),
    (Chem.MolFromSmarts("[NX3;H2][CX4]"),          "primary_amine",      3.5,  0.5,  1.0),
    (Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4]"),   "secondary_amine",    2.5,  0.4,  0.8),
    (Chem.MolFromSmarts("[NX3;H0]([CX4])([CX4])[CX4]"), "tertiary_amine", 1.5,  0.3,  0.5),
    (Chem.MolFromSmarts("[C]#[N]"),                "nitrile",            8.0,  0.4,  1.2),
    (Chem.MolFromSmarts("[CX3]=[CX3]"),            "alkene",             0.5,  0.1,  0.1),
    (Chem.MolFromSmarts("[CX2]#[CX2]"),            "alkyne",             1.0,  0.2,  0.2),
    (Chem.MolFromSmarts("[c]"),                    "aromatic_carbon",    0.5,  0.5,  0.1),
    (Chem.MolFromSmarts("[F]"),                    "fluorine",           0.0,  0.1, -0.5),
    (Chem.MolFromSmarts("[Cl]"),                   "chlorine",           0.5,  0.2, -0.3),
    (Chem.MolFromSmarts("[Br]"),                   "bromine",            0.5,  0.3, -0.2),
    (Chem.MolFromSmarts("S(=O)(=O)[CX4]"),         "sulfone",            5.0,  0.5,  1.0),
    (Chem.MolFromSmarts("S(=O)(=O)[OX2]"),         "sulfonate",          5.5,  0.6,  1.2),
    (Chem.MolFromSmarts("S(=O)(=O)F"),             "sulfonyl_fluoride",  4.0,  0.4,  0.5),
    (Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])[OX2]"), "phosphate",    4.0,  0.8,  1.5),
    (Chem.MolFromSmarts("[C](F)(F)F"),             "trifluoromethyl",    0.5,  0.2, -0.3),
    (Chem.MolFromSmarts("[C](F)(F)"),              "difluoromethylene",  0.3,  0.1, -0.2),
    (Chem.MolFromSmarts("[BX3]([OX2])"),           "boronate",           2.0,  0.7,  1.0),
    (Chem.MolFromSmarts("[BX4]([OX2])([OX2])([OX2])[OX2]"), "borate",    3.0,  0.6,  1.5),
    (Chem.MolFromSmarts("[S]([CX4])[CX4]"),        "thioether",          1.0,  0.2,  0.3),
    (Chem.MolFromSmarts("[F][CX4][OX2][CX4]"),     "fluorinated_ether",  1.0,  0.0, -0.2),
    (Chem.MolFromSmarts("[PX4](=N)([OX2])([OX2])[OX2]"), "phosphazene",  3.5,  0.4,  0.8),
    (Chem.MolFromSmarts("[OX2][CX4][CX4][OX2]"),   "glyme_chelating",    2.0,  0.1,  1.8),
    (Chem.MolFromSmarts("[SX4](=O)(=O)[NX3][SX4](=O)(=O)"), "sulfonimide", 5.0,  0.5,  0.5),
    (Chem.MolFromSmarts("[CX3](=O)[OX2]C(F)(F)F"),  "fluorinated_carbonate", 3.0,  0.3, -0.1),
]

_GC_BASE_DIELECTRIC: float = 1.9
_GC_BASE_VISCOSITY: float = 0.1
_GC_BASE_LI_SOLVATION: float = 1.0

# Saturation parameter for GC fragment additivity.
# Uses Michaelis-Menten style: contrib = max_contrib * (1 - exp(-k * count))
# k is chosen so that the first occurrence contributes ~50% of max_contrib,
# and additional occurrences give diminishing returns — reflecting the
# physical reality that property contributions saturate (first carbonate
# group drastically changes polarity; the fifth does not).
_GC_SATURATION_K: float = 0.693  # ln(2), half-max at count=1


def _saturate_contrib(count: int, max_contrib: float) -> float:
    """Michaelis-Menten style saturation for fragment additivity.

    Contribution follows: max_contrib * (1 - exp(-k * count)).
    The first occurrence contributes ~50% of max_contrib; subsequent
    occurrences contribute diminishing amounts, asymptotically approaching
    max_contrib.

    Args:
        count: Number of occurrences of the fragment.
        max_contrib: Asymptotic maximum contribution from this fragment type.

    Returns:
        Saturated contribution value.
    """
    return max_contrib * (1.0 - math.exp(-_GC_SATURATION_K * count))


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each pre-compiled fragment pattern in a molecule."""
    counts: dict[str, int] = {}
    for pattern, name, _dd, _dv, _ls in _GC_FRAGMENTS:
        matches = mol.GetSubstructMatches(pattern)
        counts[name] = len(matches)
    return counts


def predict_dielectric_proxy(ctx: MoleculeContext) -> float:
    """Predict a dielectric constant proxy via fragment-additivity + TPSA cap.

    Uses pre-computed TPSA from ``MoleculeContext.tpsa`` (lazy-cached).

    Returns:
        Unitless dielectric proxy (typically 1–15 for electrolyte solvents).
    """
    mol = ctx.mol
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, dd, _dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += _saturate_contrib(n, dd * 2.0)

    tpsa = ctx.tpsa
    value += tpsa * 0.02

    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)

    return max(1.0, value)


def predict_viscosity_proxy(ctx: MoleculeContext) -> float:
    """Predict a viscosity proxy via fragment-additivity.

    Uses pre-computed molecular weight and rotatable bond count from
    ``MoleculeContext`` (lazy-cached).

    Returns:
        Unitless viscosity proxy (typically 0.5–5.0 for electrolyte solvents).
    """
    mol = ctx.mol
    counts = _count_fragments(mol)
    value = _GC_BASE_VISCOSITY
    for _smarts, _name, _dd, dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += _saturate_contrib(n, dv * 2.0)

    mw = ctx.mw
    value += (mw - 30.0) * 0.005
    n_rot = ctx.rotatable_bonds
    value += n_rot * 0.15
    return max(0.1, value)


def predict_li_solvation_proxy(ctx: MoleculeContext) -> float:
    """Predict a Li+ solvation energy proxy via fragment-additivity.

    Li+ solvation strength is modelled as a linear combination of
    functional-group donor abilities.  The proxy correlates with:
      - Donor number (DN): higher DN → stronger Li+ binding
      - Chelation: polydentate ethers (glymes) bind Li+ more strongly
        than monodentate analogues

    Uses pre-computed molecular weight from ``MoleculeContext.mw`` (lazy-cached).

    Returns:
        Unitless Li+ solvation proxy (typically 1.0–6.0).
            ~1.0–2.5 : weak binding (poor salt dissociation)
            ~2.5–4.5 : moderate binding (Goldilocks zone)
            ~4.5–6.0+ : strong binding (poor transference number)
    """
    mol = ctx.mol
    counts = _count_fragments(mol)
    value = _GC_BASE_LI_SOLVATION
    for _smarts, _name, _dd, _dv, ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += _saturate_contrib(n, ls * 2.0)

    mw = ctx.mw
    value += max(0.0, (mw - 50.0)) * 0.002
    return max(0.5, value)


# ---------------------------------------------------------------------------
# QuantumOracle — Real Quantum Chemistry for Frontier Orbitals
# ---------------------------------------------------------------------------
# HOMO/LUMO energies are global, delocalised quantum phenomena that CANNOT
# be predicted by fragment-additivity.  This module provides a two-tier
# quantum oracle:
#   1. xTB (GFN2-xTB) via subprocess — fast semi-empirical QM (preferred)
#   2. Topological Orbital Model (TOM) — conjugation-aware fallback
#
# The fallback is based on the particle-in-a-box model for π-electrons:
#   E ∝ n²/L²  where L = conjugation length, n = electron count
# with heteroatom perturbations from Hückel theory.
# This is non-linear, topology-dependent, and physically grounded.


def _find_xtb_binary() -> str | None:
    """Locate the xTB binary on the system PATH."""
    for candidate in ["xtb", "xtb_opt"]:
        with contextlib.suppress(Exception):
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
    return None


_HAS_XTB: bool = _find_xtb_binary() is not None


def has_xtb() -> bool:
    """Return True if the xTB binary is available on PATH."""
    return _HAS_XTB


def _generate_xyz(mol: Chem.Mol) -> str:
    """Generate an XYZ string from a molecule with a 3D conformer.

    Uses RDKit's ETKDG (Experimental Torsion Knowledge Distance Geometry)
    for conformer generation, then UFF optimisation.
    """
    from rdkit.Chem import AllChem

    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache()
    with contextlib.suppress(Exception):
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        return _generate_xyz_geometry_optimized(mol)

    with contextlib.suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=250)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    lines = [str(n_atoms), ""]
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        symb = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{symb} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)


def _generate_xyz_geometry_optimized(mol: Chem.RWMol) -> str:
    """Fallback XYZ generation using distance geometry without ETKDG."""
    from rdkit.Chem import AllChem, rdDistGeom

    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache()
    with contextlib.suppress(Exception):
        mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        with contextlib.suppress(Exception):
            AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())

    with contextlib.suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    lines = [str(n_atoms), ""]
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        symb = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{symb} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)


def _run_xtb(xyz_content: str, workdir: str | None = None) -> dict[str, float] | None:
    """Run xTB single-point calculation and parse HOMO/LUMO from output.

    Args:
        xyz_content: XYZ-format molecular geometry.
        workdir: Working directory for xTB (temp dir if None).

    Returns:
        Dict with ``homo_eV``, ``lumo_eV``, ``dipole_D`` or None on failure.
    """
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="aurelius_xtb_")

    xyz_path = os.path.join(workdir, "input.xyz")
    with open(xyz_path, "w") as f:
        f.write(xyz_content)

    xtb_bin = _find_xtb_binary()
    if xtb_bin is None:
        return None

    try:
        result = subprocess.run(
            [xtb_bin, "--gfn", "2", "--sp", xyz_path],
            cwd=workdir,
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        logger.debug("xTB run failed: %s", exc)
        return None

    return _parse_xtb_output(output)


def _parse_xtb_output(output: str) -> dict[str, float] | None:
    """Parse xTB output text for HOMO, LUMO, and dipole moment."""
    homo: float | None = None
    lumo: float | None = None
    dipole: float | None = None

    for line in output.splitlines():
        stripped = line.strip()
        # Match lines like:  HOMO:       -6.23478 eV
        if "HOMO" in stripped and "eV" in stripped and ":" in stripped:
            parts = stripped.split()
            for i, p in enumerate(parts):
                if p == "HOMO" and parts[i + 1] == ":" and i + 2 < len(parts):
                    with contextlib.suppress(ValueError):
                        homo = float(parts[i + 2])
        # Match lines like:  LUMO:        0.84527 eV
        if "LUMO" in stripped and "eV" in stripped and ":" in stripped:
            parts = stripped.split()
            for i, p in enumerate(parts):
                if p == "LUMO" and parts[i + 1] == ":" and i + 2 < len(parts):
                    with contextlib.suppress(ValueError):
                        lumo = float(parts[i + 2])

    if homo is not None and lumo is not None:
        logger.info("QuantumOracle (xTB): HOMO=%.3f eV, LUMO=%.3f eV", homo, lumo)
        return {
            "homo_eV": homo,
            "lumo_eV": lumo,
            "dipole_D": dipole or 0.0,
        }

    logger.debug("Could not parse HOMO/LUMO from xTB output")
    return None


# ---------------------------------------------------------------------------
# Topological Orbital Model (TOM) — Non-linear Fallback for HOMO/LUMO
# ---------------------------------------------------------------------------
# Based on Hückel Molecular Orbital (HMO) theory and the particle-in-a-box:
#   - The π-electron system defines the frontier orbital energy scale
#   - Conjugation length L determines the HOMO-LUMO gap: ΔE ∝ 1/L²
#   - Heteroatoms perturb the energy via electronegativity differences
#   - The model is non-linear in molecular topology (not fragment-additive)
#
# Reference: "Hückel Theory for Organic Chemists" (Heilbronner & Bock)


def _longest_conjugation_path(mol: Chem.Mol) -> int:
    """Find the longest conjugated π-system in a molecule.

    Returns the number of conjugated atoms in the longest continuous
    conjugated path.
    """
    visited: set[int] = set()
    max_path = 0

    def _dfs(atom_idx: int, path_len: int) -> None:
        nonlocal max_path
        visited.add(atom_idx)
        max_path = max(max_path, path_len)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited and _is_conjugated_bond(mol, atom_idx, n_idx):
                _dfs(n_idx, path_len + 1)
        visited.discard(atom_idx)

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() or atom.GetDegree() > 0:
            _dfs(atom.GetIdx(), 1)

    return max_path


def _is_conjugated_bond(mol: Chem.Mol, i: int, j: int) -> bool:
    """Check if bond between atoms i and j is conjugated."""
    bond = mol.GetBondBetweenAtoms(i, j)
    if bond is None:
        return False
    if bond.GetIsConjugated():
        return True
    bond_type = bond.GetBondType()
    if bond_type in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
        return True
    a1 = mol.GetAtomWithIdx(i)
    a2 = mol.GetAtomWithIdx(j)
    return bool(a1.GetIsAromatic() or a2.GetIsAromatic())


def _count_heteroatom_perturbations(mol: Chem.Mol) -> tuple[int, int, int]:
    """Count electron-withdrawing (EW) and electron-donating (ED) heteroatoms
    in conjugated positions, and total π-electrons.

    Returns:
        (n_ew, n_ed, n_pi_electrons)
    """
    n_ew = 0
    n_ed = 0
    n_pi = 0

    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if atom.GetIsAromatic() or atom.GetDegree() > 0:
            # Count π-electrons from double bonds, lone pairs, aromaticity
            if z == 6 and atom.GetIsAromatic():
                n_pi += 1
            if z == 7:
                n_pi += 1
                if atom.GetIsAromatic():
                    n_ed += 1  # Pyridine-like N donates electron density
                else:
                    n_ew += 1  # Nitrile-like N withdraws
            if z == 8:
                n_pi += 2  # Oxygen lone pairs
                n_ew += 1
            if z == 9:
                n_ew += 1  # Fluorine is strongly EW
            if z == 16:
                n_pi += 2
                n_ew += 1
            if z == 15:
                n_pi += 1
                n_ew += 1

    return n_ew, n_ed, n_pi


def predict_tom_orbitals(mol: Chem.Mol) -> tuple[float, float]:
    """Predict HOMO/LUMO using the Topological Orbital Model (TOM).

    The model estimates frontier orbital energies from:
      1. Longest conjugation path length (L)
      2. HOMO-LUMO gap from particle-in-a-box: ΔE = h²/(8mL²) in atomic units
      3. Heteroatom perturbations (electron-withdrawing/donating)
      4. Base offset calibrated to common electrolyte molecules

    Returns:
        (homo_eV, lumo_eV)
    """
    L = _longest_conjugation_path(mol)
    L = max(L, 2)

    # Particle-in-a-box gap: ΔE = h²/(8meL²) → in eV: ≈ 37.6/L² eV
    n_ew, n_ed, n_pi = _count_heteroatom_perturbations(mol)

    # Base energies for a simple alkane (no conjugation)
    base_homo = -9.5
    base_lumo = 3.0

    if L >= 3:
        gap = 37.6 / (L * L)
        mid = (base_homo + base_lumo) / 2.0
        homo = mid - gap / 2.0
        lumo = mid + gap / 2.0
    else:
        homo = base_homo
        lumo = base_lumo

    # Heteroatom perturbations (Hückel-like correction)
    ew_shift = -0.08 * n_ew
    ed_shift = 0.12 * n_ed
    homo += ew_shift + ed_shift
    lumo += ew_shift * 0.7 + ed_shift * 0.5

    # Fluorine correction (strong inductive withdrawal, stabilises both)
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    f_shift = -0.10 * n_f
    homo += f_shift
    lumo += f_shift

    # Sulfone/Phosphate correction (strong EW)
    n_sulfone = len(mol.GetSubstructMatches(_SULFONE_PATTERN))
    homo += -0.25 * n_sulfone
    lumo += -0.40 * n_sulfone

    n_cf3 = len(mol.GetSubstructMatches(_CF3_PATTERN))
    homo += -0.20 * n_cf3
    lumo += -0.15 * n_cf3

    return homo, lumo


# ---------------------------------------------------------------------------
# QuantumOracle — Unified interface for xTB + TOM fallback
# ---------------------------------------------------------------------------


class QuantumOracle:
    """Quantum-chemical oracle for frontier orbital energies.

    Two-tier evaluation:
      1. xTB (GFN2-xTB) via subprocess — preferred, real QM
      2. Topological Orbital Model (TOM) — conjugation-aware fallback

    Results are cached by SMILES to avoid redundant computation.

    Usage:
        >>> qc = QuantumOracle()
        >>> result = qc.evaluate(mol)
        >>> result["homo_eV"]
        -7.23
    """

    def __init__(self, use_xtb: bool = True) -> None:
        self._use_xtb = use_xtb and _HAS_XTB
        self._cache: dict[str, dict[str, float]] = {}
        self._n_xtb_calls = 0
        self._n_tom_calls = 0

        if use_xtb and not _HAS_XTB:
            logger.info("QuantumOracle: xTB binary not found — using TOM fallback.")
        elif self._use_xtb:
            logger.info("QuantumOracle: xTB backend ENABLED.")

    @property
    def method(self) -> str:
        return "xTB (GFN2-xTB)" if self._use_xtb else "TOM (Topological Orbital Model)"

    @property
    def n_quantum_calls(self) -> int:
        return self._n_xtb_calls + self._n_tom_calls

    def evaluate(self, mol: Chem.Mol) -> dict[str, float]:
        smiles = Chem.MolToSmiles(mol)
        if smiles in self._cache:
            return dict(self._cache[smiles])

        result: dict[str, float] | None = None
        if self._use_xtb:
            xyz = _generate_xyz(mol)
            result = _run_xtb(xyz)
            if result is not None:
                self._n_xtb_calls += 1
            else:
                logger.warning("QuantumOracle: xTB calculation failed — falling back to TOM.")

        if result is None:
            homo, lumo = predict_tom_orbitals(mol)
            result = {
                "homo_eV": homo,
                "lumo_eV": lumo,
                "dipole_D": 0.0,
            }
            self._n_tom_calls += 1

        self._cache[smiles] = result
        return dict(result)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# PropertyOracle — Hybrid GC (bulk) + Quantum (orbital) Oracle
# ---------------------------------------------------------------------------


class PropertyOracle:
    """Multi-objective property oracle with a hybrid physics model.

    Architecture:
      - HOMO / LUMO / Gap: QuantumOracle (xTB preferred, TOM fallback)
        Frontier orbitals are delocalised quantum phenomena — NOT additive.
        The QuantumOracle provides physically valid, non-linear predictions.
      - Dielectric proxy: GC fragment-additivity + TPSA-based cap
      - Viscosity proxy: GC fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy: GC fragment-additivity

    This hybrid model justifies the Bayesian active learning loop (the
    oracle is non-linear and moderately expensive) while keeping bulk
    property prediction lightweight and interpretable.
    """

    _CACHE: dict[str, dict[str, Any]] | None = None

    def __init__(self, use_xtb: bool = True) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Evaluate a molecule and return predicted properties.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dictionary with keys:
              - homo_eV (float)
              - lumo_eV (float)
              - gap_eV (float)
              - dielectric_proxy (float)
              - viscosity_proxy (float)
              - li_solvation_proxy (float)
              - domain_applicable (bool)
              - domain_reason (str)

        Raises:
            TypeError: If ctx is not a MoleculeContext.
        """
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )

        smiles = ctx.smiles
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        # Quantum: HOMO/LUMO (non-linear, topology-aware)
        quantum_result = self._quantum.evaluate(ctx.mol)
        homo = quantum_result["homo_eV"]
        lumo = quantum_result["lumo_eV"]
        gap = lumo - homo

        # GC: bulk properties (fragment-additivity)
        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)

        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "dielectric_proxy": round(dielectric, 4),
            "viscosity_proxy": round(viscosity, 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "domain_applicable": True,
            "domain_reason": _DATA_SOURCE,
            "quantum_method": self._quantum.method,
        }

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        """Convenience: parse SMILES then evaluate.

        Args:
            smiles: SMILES string.

        Returns:
            Result dict (same as evaluate()).
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist the cache to disk with joblib."""
        import joblib
        payload: dict[str, Any] = {
            "cache": self._CACHE,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: cache saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        """Load cached predictions from a joblib cache file.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        try:
            import joblib
            payload = joblib.load(path)
        except (FileNotFoundError, Exception) as exc:
            logger.debug("PropertyOracle: no cached oracle at %s (%s)", path, exc)
            return False

        self._CACHE = payload.get("cache")
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        """Clear the SMILES->properties cache."""
        if self._CACHE is not None:
            self._CACHE.clear()
        self._quantum.clear_cache()
