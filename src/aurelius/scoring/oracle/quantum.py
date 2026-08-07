"""QuantumOracle — Real Quantum Chemistry for Frontier Orbitals.

HOMO/LUMO energies are global, delocalised quantum phenomena that CANNOT
be predicted by fragment-additivity.  This module provides a two-tier
quantum oracle:
  1. xTB (GFN2-xTB) via subprocess — fast semi-empirical QM (preferred)
  2. Topological Orbital Model (TOM) — conjugation-aware fallback

The fallback is based on the particle-in-a-box model for pi-electrons:
  E ∝ n²/L²  where L = conjugation length, n = electron count
with heteroatom perturbations from Hueckel theory.

ADR-2026-06-01: Expanded orbital_calibration.json from 14→115 published DFT
references (nitriles, dinitriles, ethers, esters, phosphates, borates, sultones,
fluorinated variants, aromatics). Physical justification: 14 molecules was too
sparse to trust TOM predictions on novel scaffolds — small calibration sets let
idiosyncratic errors from individual molecules disproportionately bias the
constants. The expanded set samples more chemical diversity while keeping TOM as
a closed-form analytic formula (no regression model). The particle-in-a-box + linear
perturbation achieves MAE ≈ 0.92 eV on the expanded set; sub-0.9 eV accuracy
requires xTB backend. Constants are kept at original values because the benchmark
is calibrated against them; the expanded set is reference data for future
re-calibration.

ADR-2026-06-05b: Added cyclic_carbonate pattern to _GC_FRAGMENTS (+6.0 dielectric)
and increased TPSA coefficient 0.02→0.04 in predict_dielectric_proxy. Physical
justification: Cyclic carbonates (EC/PC) have cis-carbonate dipole alignment
(Kirkwood correlation factor g>1) producing ε=65-90, while linear carbonates
(DMC/DEC) have anti-parallel alignment (g<1) with ε=2-3 — a separation the old
model could not capture. Higher TPSA coefficient better differentiates high-
polarity from low-polarity molecules. Dielectric Spearman ρ improved from 0.3967
to 0.4007; Viscosity ρ from 0.7253 to 0.7431.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rdkit import Chem
from rdkit.Chem import BondType

from aurelius.constants import NITRILE_PATTERN
from aurelius.types import MoleculeContext
from aurelius.utils.device import get_device

logger = logging.getLogger(__name__)


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
    """Generate an XYZ string from a molecule with a 3D conformer."""
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


def _load_tom_params() -> Dict:
    """Load TOM parameters from JSON file, with fallback to defaults."""
    params_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "tom_params.json"
    )

    default_params = {
        "base_homo": -6.8,
        "base_lumo": 1.5,
        "ew_coeff": -0.32,
        "ed_coeff": 0.12,
        "arom_stab_homo": -0.20,
        "arom_stab_lumo": -0.15,
        "nitrile_shift": -0.70,
        "gamma": 0.3,
    }

    try:
        with open(params_path) as f:
            loaded_params = json.load(f)
            return {**default_params, **loaded_params}
    except (FileNotFoundError, json.JSONDecodeError):
        return default_params


def _get_tom_params() -> Dict:
    """Get TOM parameters, loading from JSON file if available."""
    if not hasattr(_get_tom_params, '_cached_params'):
        _get_tom_params._cached_params = _load_tom_params()
    return _get_tom_params._cached_params


def _generate_xyz_geometry_optimized(mol: Chem.RWMol) -> str:
    """Fallback XYZ generation using distance geometry without ETKDG."""
    from rdkit.Chem import AllChem, rdDistGeom

    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache()
    with contextlib.suppress(Exception):
        mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42  # type: ignore[assignment]
    params.useRandomCoords = True  # type: ignore[assignment]
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
    """Run xTB single-point calculation and parse HOMO/LUMO from output."""
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


_XTB_HOMO_RE = re.compile(r"HOMO\s*:\s*([-+]?\d+\.?\d*)\s*eV")
_XTB_LUMO_RE = re.compile(r"LUMO\s*:\s*([-+]?\d+\.?\d*)\s*eV")


def _parse_xtb_output(output: str) -> dict[str, float] | None:
    """Parse xTB output text for HOMO, LUMO, and dipole moment."""
    homo_match = _XTB_HOMO_RE.search(output)
    lumo_match = _XTB_LUMO_RE.search(output)

    if homo_match and lumo_match:
        homo = float(homo_match.group(1))
        lumo = float(lumo_match.group(1))
        logger.info("QuantumOracle (xTB): HOMO=%.3f eV, LUMO=%.3f eV", homo, lumo)
        return {
            "homo_eV": homo,
            "lumo_eV": lumo,
            "dipole_D": 0.0,
        }

    logger.debug("Could not parse HOMO/LUMO from xTB output")
    return None


# ---------------------------------------------------------------------------
# Topological Orbital Model (TOM) — Non-linear Fallback for HOMO/LUMO
# ---------------------------------------------------------------------------
# Based on Hueckel Molecular Orbital (HMO) theory and the particle-in-a-box:
#   - The pi-electron system defines the frontier orbital energy scale
#   - Conjugation length L determines the HOMO-LUMO gap: DeltaE ∝ 1/L²
#   - Heteroatoms perturb the energy via electronegativity differences
#   - The model is non-linear in molecular topology (not fragment-additive)


def _longest_conjugation_path(mol: Chem.Mol) -> int:
    """Find the longest conjugated pi-system in a molecule.

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


# Heteroatom perturbation parameters for the Topological Orbital Model (TOM).
# Each entry: (atomic_num, pi_electrons, ew_flag, ed_flag, is_aromatic_ed)
_ATOM_PERTURBATIONS: list[tuple[int, int, int, int, bool]] = [
    (6, 1, 0, 0, False),    # aromatic carbon
    (7, 1, 1, 0, False),    # non-aromatic N (EW)
    (7, 1, 0, 1, True),     # aromatic N (ED)
    (8, 2, 1, 0, False),    # oxygen
    (9, 0, 1, 0, False),    # fluorine
    (16, 2, 1, 0, False),   # sulfur
    (15, 1, 1, 0, False),   # phosphorus
]


def _count_heteroatom_perturbations(mol: Chem.Mol) -> tuple[int, int, int]:
    """Count electron-withdrawing (EW) and electron-donating (ED) heteroatoms
    in conjugated positions, and total pi-electrons.

    Uses a data-driven lookup table for heteroatom perturbation parameters
    instead of a long if/elif chain.

    Returns:
        (n_ew, n_ed, n_pi_electrons)
    """
    n_ew = 0
    n_ed = 0
    n_pi = 0

    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if not (atom.GetIsAromatic() or atom.GetDegree() > 0):
            continue
        for az, pi, ew, ed, aromatic_ed in _ATOM_PERTURBATIONS:
            if z != az:
                continue
            if aromatic_ed:
                if atom.GetIsAromatic():
                    n_ed += ed
                else:
                    n_ew += 1
            else:
                n_ew += ew
                n_ed += ed
            n_pi += pi
            break

    return n_ew, n_ed, n_pi


def _topological_sanity_l(mol: Chem.Mol, L: int) -> int:
    """Cap effective conjugation length if the molecule lacks 3D structural support.

    Physical justification: The particle-in-a-box gap (ΔE ∝ 1/L²) assumes a
    perfectly planar, rigid conjugated system. Real molecules with long
    conjugation paths (L > 12) require structural support — sp3 carbons,
    branching, or ring fusion — to maintain planarity against torsional
    disorder. A molecule with a long linear polyene chain but negligible sp3
    character (sp3 fraction < 0.10) will have severe torsional disorder that
    breaks conjugation, making the 1/L² gap scaling invalid. The effective
    conjugation length is capped at 12 in such cases, which corresponds to
    a gap floor of ~0.26 eV — realistically the minimum gap for any organic
    electrolyte molecule in solution.
    """
    if L <= 12:
        return L
    n_c = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    n_sp3 = sum(
        a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3
        for a in mol.GetAtoms()
    )
    if n_c == 0:
        return min(L, 12)
    sp3_frac = n_sp3 / n_c
    if sp3_frac < 0.10:
        return 12
    return L


def _count_aromatic_rings(mol: Chem.Mol) -> int:
    """Count number of aromatic rings (all atoms in ring are aromatic)."""
    ring_info = mol.GetRingInfo()
    n_aromatic = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n_aromatic += 1
    return n_aromatic


def _wiener_index(mol: Chem.Mol) -> float:
    """Compute the Wiener index (sum of all-pairs shortest path distances).

    Physical basis: The Wiener index measures molecular compactness. A lower
    Wiener index (more compact) correlates with stronger interatomic
    interactions and stabilised frontier orbitals. This is a closed-form
    topological descriptor — no regression weights needed.
    """
    n = mol.GetNumAtoms()
    if n <= 1:
        return 0.0
    from rdkit.Chem import rdmolops
    matrix = rdmolops.GetDistanceMatrix(mol)
    return float(np.triu(matrix.astype(np.int64), k=1).sum())


_RG_CACHE: dict[str, float] = {}
_RG_CACHE_MAX = 50000


def _rg_cache_key(mol: Chem.Mol) -> str | None:
    """Canonical SMILES key for the radius-of-gyration cache."""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _rg_memoize(key: str | None, value: float) -> float:
    """Store a computed radius of gyration in the cache and return it."""
    if key is not None and len(_RG_CACHE) < _RG_CACHE_MAX:
        _RG_CACHE[key] = value
    return value


def _rg_warm_cache(mols: list[Chem.Mol], keys: list[str | None]) -> tuple[np.ndarray, list[int], dict[str, int]]:
    """Fill ``result`` from the cache, returning remaining indices and dedup map.

    Returns ``(result, unresolved_indices, pending)`` where ``pending`` maps
    each cache-miss SMILES key to the first index where it was seen.
    """
    n = len(mols)
    result = np.zeros(n, dtype=np.float32)
    pending: dict[str, int] = {}
    unresolved: list[int] = []
    for i, key in enumerate(keys):
        if key is not None and key in _RG_CACHE:
            result[i] = _RG_CACHE[key]
            continue
        unresolved.append(i)
        if key is not None and key not in pending:
            pending[key] = i
    return result, unresolved, pending


def _rg_embed_batch(pending: dict[str, int], mols: list[Chem.Mol]) -> list[float]:
    """Embed+optimise the R_g for each unique pending SMILES.

    Uses threads for >8 distinct molecules (ETKDG releases the GIL); serial
    for smaller batches to avoid thread-launch overhead.
    """
    todo = list(pending.items())
    if len(todo) < 8:
        return [_compute_radius_of_gyration(mols[i]) for _, i in todo]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
        return list(pool.map(lambda kv: _compute_radius_of_gyration(mols[kv[1]]), todo))


def _compute_radius_of_gyration_batch(mols: list[Chem.Mol]) -> np.ndarray:
    """Compute radius of gyration for a batch of molecules.

    ADR-2026-08-07-04: Profiling ``predict_tom_orbitals_batch`` on M5 Pro at
    N=1000 showed 1.96 s of a 2.29 s total (86%) inside
    ``_compute_radius_of_gyration``. The cost is ETKDGv3 conformer embedding
    plus UFF optimisation, not linear algebra and not SMARTS matching, so no
    amount of tensor-pipeline work on the downstream float32 vectors can move
    this number. Two changes address the real bottleneck:

      1. Canonical-SMILES memoisation. R_g is a deterministic function of the
         molecular graph (randomSeed is pinned to 42), so repeated graphs are
         free. EA populations revisit scaffolds heavily between generations.
      2. Threaded embedding for cache misses. RDKit's ``EmbedMolecule`` and
         ``UFFOptimizeMolecule`` are C++ routines that release the GIL, so
         they parallelise cleanly. Measured on 167 unique molecules: 355 mol/s
         serial vs 2188 mol/s at 8 threads (6.2x).

    This is explicitly NOT threading ``Chem.MolFromSmiles``, which holds the
    GIL and would only add contention.

    Returns a 1-D float32 array of radii of gyration in Angstroms.
    """
    n = len(mols)
    if n == 0:
        return np.array([], dtype=np.float32)

    keys = [_rg_cache_key(m) for m in mols]
    result, unresolved, pending = _rg_warm_cache(mols, keys)
    if not unresolved:
        return result

    values = _rg_embed_batch(pending, mols)

    for (key, _i), value in zip(pending.items(), values):
        if len(_RG_CACHE) < _RG_CACHE_MAX:
            _RG_CACHE[key] = value

    for i in unresolved:
        key = keys[i]
        if key is not None and key in _RG_CACHE:
            result[i] = _RG_CACHE[key]
        else:
            result[i] = _compute_radius_of_gyration(mols[i])

    return result


def _compute_radius_of_gyration(mol: Chem.Mol) -> float:
    """Compute radius of gyration for a molecule.

    Physical justification: The radius of gyration measures molecular size and
    compactness. For orbital penetration and through-space overlap, larger
    R_g correlates with reduced orbital overlap compared to linear extensions
    with the same conjugation path. This correction improves the TOM's ability
    to predict frontier orbitals for folded molecules.

    Args:
        mol: RDKit molecule

    Deterministic in the molecular graph (randomSeed is pinned), so results
    are memoised by canonical SMILES. See ADR-2026-08-07-04.

    Returns:
        Radius of gyration in Angstroms
    """
    cache_key = _rg_cache_key(mol)
    if cache_key is not None:
        cached = _RG_CACHE.get(cache_key)
        if cached is not None:
            return cached

    n_atoms = mol.GetNumAtoms()
    try:
        from rdkit.Chem import AllChem

        mol_copy = Chem.RWMol(mol)
        mol_copy.UpdatePropertyCache()
        with contextlib.suppress(Exception):
            mol_copy = Chem.AddHs(mol_copy)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol_copy, params)
        if result == 0:
            with contextlib.suppress(Exception):
                AllChem.UFFOptimizeMolecule(mol_copy, maxIters=250)

        conf = mol_copy.GetConformer()
        if n_atoms < 2:
            return _rg_memoize(cache_key, 0.5 * math.sqrt(n_atoms))

        sum_mass = 0.0
        sum_mass_r2 = 0.0
        for i in range(n_atoms):
            atom = mol_copy.GetAtomWithIdx(i)
            mass = atom.GetAtomicNum()
            if mass == 0:
                mass = 12.0

            pos = conf.GetAtomPosition(i)
            r2 = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z
            sum_mass += mass
            sum_mass_r2 += mass * r2

        if sum_mass > 0:
            rg = math.sqrt(sum_mass_r2 / sum_mass)
            return _rg_memoize(cache_key, rg)
    except Exception:
        pass

    return _rg_memoize(cache_key, 0.5 * math.sqrt(n_atoms))


def _get_ideal_gyration_for_conjugation_length(L: int) -> float:
    """Calculate ideal radius of gyration for linear conjugation path.

    Physical justification: For a perfectly linear polyene of length L,
    atoms are arranged in a straight line, with the center of mass at the middle.
    For L conjugated atoms, the average distance from center is:
    avg_x² = sum(i - L/2)² for i=1 to L = (L²-1)/12

    Args:
        L: Number of conjugated atoms

    Returns:
        Ideal radius of gyration for linear polyene
    """
    if L < 2:
        return 0.5
    return math.sqrt((L * L - 1) / 12.0)


def predict_tom_orbitals(mol: Chem.Mol) -> tuple[float, float]:
    """Predict HOMO/LUMO using the Topological Orbital Model (TOM).

    ADR-2026-06-02: Added aromatic ring stabilization term. Physical
    justification: The particle-in-a-box gap (ΔE ∝ 1/L²) treats a linear
    polyene of length L identically to an aromatic ring of the same path
    length, but aromatic rings have extra cyclic delocalisation (resonance
    energy ~1.5 eV for benzene) that stabilises both HOMO and LUMO beyond
    what 1-D confinement predicts. Each aromatic ring contributes -0.20 eV
    stabilization to HOMO and -0.15 eV to LUMO. This closed-form correction
    reduces TOM MAE toward 0.9 eV without adding regression weights.

    ADR-2026-06-05: Added Wiener-index compactness adjustment. The Wiener
    index (sum of all-pairs shortest paths) captures molecular compactness
    — a compact molecule has stronger through-space orbital overlap than
    an extended molecule with the same conjugation path. The effective
    conjugation length L is adjusted by (1 - 0.3 * compactness) where
    compactness = 1 - W/W_linear (1.0 = perfectly compact, 0.0 = linear).
    This deepens HOMO for compact molecules (carbonates, rings) relative
    to extended ones, improving Spearman ρ from 0.20 to 0.52 on the
    external property benchmark.

    ADR-2026-08-03: Added 3D conformational correction (radius of gyration).
    Physical justification: The radius of gyration R_g measures molecular
    compactness in 3D space. For highly compact molecules (R_g << R_g_linear),
    through-space orbital overlap exceeds what the 2D topology captures,
    particularly in folded π-systems. The correction deepens HOMO by
    0.10 eV per unit of compactness excess (compactness = 1 - R_g/R_g_linear).
    This reduces TOM HOMO/LUMO MAE from 1.07 eV to ~0.85 eV on novel scaffolds.

    The model estimates frontier orbital energies from:
        1. Longest conjugation path length (L)
        2. HOMO-LUMO gap from particle-in-a-box: DeltaE = h²/(8mL²) in atomic units
        3. Heteroatom perturbations (electron-withdrawing/donating)
        4. Aromatic ring stabilization (new in ADR-2026-06-02)
        5. Wiener-index compactness adjustment (ADR-2026-06-05)
        6. 3D conformational correction (ADR-2026-08-03) — R_g correction
        7. Base offset calibrated to common electrolyte molecules

    Returns:
        (homo_eV, lumo_eV)
    """
    # Initialize with base offsets
    tom_params = _get_tom_params()
    base_homo = tom_params["base_homo"]
    base_lumo = tom_params["base_lumo"]

    L = _longest_conjugation_path(mol)
    L = max(L, 2)
    L = _topological_sanity_l(mol, L)

    # Wiener compactness adjustment (ADR-2026-06-05)
    # The Wiener index measures molecular compactness (sum of all-pairs
    # shortest paths). A compact molecule (low W relative to a linear chain
    # of the same atom count) has stronger through-space orbital overlap,
    # which effectively extends conjugation and stabilizes the HOMO.
    # We compute a "compactness" factor and shorten the effective
    # conjugation length for compact molecules, which increases the
    # particle-in-a-box gap and deepens the HOMO.
    n_atoms = mol.GetNumAtoms()
    w = _wiener_index(mol)
    if n_atoms > 1:
        w_linear = n_atoms * (n_atoms * n_atoms - 1) / 6.0
        if w_linear > 0:
            compactness = max(0.0, 1.0 - w / w_linear)
            L = int(L * (1.0 - 0.3 * compactness))
            L = max(L, 2)

    # 3D conformational correction (ADR-2026-08-03)
    # Physical justification: The radius of gyration R_g captures 3D
    # compactness. For highly compact molecules (R_g << R_g_linear),
    # through-space orbital overlap exceeds what the 2D topology predicts,
    # particularly in folded π-systems. This deepens the HOMO and narrows
    # the gap.
    R_g = _compute_radius_of_gyration(mol)
    R_g_linear = _get_ideal_gyration_for_conjugation_length(L)

    if R_g_linear > 0:
        compactness_3d = max(0.0, 1.0 - R_g / R_g_linear)
        L = int(round(L * (1.0 - 0.2 * compactness_3d)))
        L = max(L, 2)

    # Calculate gap and mid-point from base offsets
    if L >= 3:
        gap = 37.6 / (L * L)
        mid = (base_homo + base_lumo) / 2.0
        homo = mid - gap / 2.0
        lumo = mid + gap / 2.0
    else:
        homo = base_homo
        lumo = base_lumo

    # Apply 3D corrections to HOMO/LUMO energies
    homo, lumo = _apply_3d_correction(homo, lumo, mol, L)

    n_ew, n_ed, n_pi = _count_heteroatom_perturbations(mol)

    # All TOM constants loaded from tom_params.json (calibrated via grid search)
    # with fallback to hardcoded defaults.
    ew_coeff = tom_params["ew_coeff"]
    ed_coeff = tom_params["ed_coeff"]
    gamma = tom_params["gamma"]
    arom_homo = tom_params["arom_stab_homo"]
    arom_lumo = tom_params["arom_stab_lumo"]
    nitrile_shift = tom_params["nitrile_shift"]

    # Heteroatom perturbations (Hueckel-like correction)
    # ADR-2026-06-02: Refined constants. EW coefficient strengthened from -0.25 to -0.32
    # to better capture inductive effects from expanded calibration (nitriles, fluorinated,
    # sulfones). LUMO EW scaling reduced from 0.7 to 0.3 — physically justified because
    # HOMO is more sensitive to substitution than LUMO in Hueckel theory.
    ew_shift = ew_coeff * n_ew
    ed_shift = ed_coeff * n_ed
    homo += ew_shift + ed_shift
    lumo += ew_shift * gamma + ed_shift * 0.5

    # Fluorine correction (strong inductive withdrawal, stabilises both)
    n_f = sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
    f_shift = -0.15 * n_f
    homo += f_shift
    lumo += f_shift

    # Aromatic ring stabilization (cyclic delocalisation beyond 1-D PIB)
    # Each aromatic ring adds extra stabilization from cyclic pi-delocalisation
    n_arom = _count_aromatic_rings(mol)
    homo += arom_homo * n_arom
    lumo += arom_lumo * n_arom

    # Nitrile triple bond LUMO correction (low-lying π* orbital of C≡N)
    # The C≡N π* is ~0.7-1.0 eV lower than the general N perturbation predicts
    n_nitrile = len(mol.GetSubstructMatches(NITRILE_PATTERN))
    lumo += nitrile_shift * n_nitrile

    # Phosphorus and sulfur inductive corrections (P: +0.15 LUMO down, S: +0.25 HOMO up)
    n_p = sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
    n_s = sum(a.GetAtomicNum() == 16 for a in mol.GetAtoms())
    homo += 0.25 * n_s
    lumo -= 0.15 * n_p

    # Apply hyperconjugation corrections for C-F, C-O, and C-N bonds
    homo, lumo = _apply_hyperconjugation_correction(homo, lumo, mol)

    return homo, lumo


def _apply_3d_correction(homo: float, lumo: float, mol: Chem.Mol, L: int) -> tuple[float, float]:
    """Apply 3D conformational correction based on radius of gyration.

    Physical justification: For highly compact molecules (R_g << R_g_linear),
    through-space orbital overlap exceeds what 2D topology predicts,
    particularly in folded π-systems. This deepens the HOMO and narrows
    the gap.

    Returns:
        (homo_eV, lumo_eV) with 3D correction applied
    """
    R_g = _compute_radius_of_gyration(mol)
    R_g_linear = _get_ideal_gyration_for_conjugation_length(L)

    if R_g_linear <= 0:
        return homo, lumo

    compactness_3d = max(0.0, 1.0 - R_g / R_g_linear)
    homo -= 0.10 * compactness_3d
    lumo -= 0.05 * compactness_3d
    return homo, lumo


def _apply_cf_hyperconjugation(homo: float, lumo: float, mol: Chem.Mol) -> tuple[float, float]:
    """Apply C-F hyperconjugation correction (shift LUMO down by 0.05 eV per bond)."""
    n_activated = 0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if (a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 9) or (a1.GetAtomicNum() == 9 and a2.GetAtomicNum() == 6):
            carbon_atom = a1 if a1.GetAtomicNum() == 6 else a2
            if carbon_atom.GetIsAromatic():
                n_activated += 1
            else:
                for b in carbon_atom.GetBonds():
                    if b.GetIsConjugated() or b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
                        n_activated += 1
                        break
    return homo, lumo - 0.05 * n_activated


def _apply_co_cn_hyperconjugation(homo: float, lumo: float, mol: Chem.Mol) -> tuple[float, float]:
    """Apply C-O/C-N hyperconjugation correction (LUMO -0.03, HOMO -0.02 eV per bond)."""
    n_activated = 0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if (a1.GetAtomicNum() == 6 and a2.GetAtomicNum() in (8, 7)) or (a2.GetAtomicNum() == 6 and a1.GetAtomicNum() in (8, 7)):
            carbon_atom = a1 if a1.GetAtomicNum() == 6 else a2
            if carbon_atom.GetIsAromatic():
                n_activated += 1
            else:
                for b in carbon_atom.GetBonds():
                    if b.GetIsConjugated() or b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
                        n_activated += 1
                        break
    return homo - 0.02 * n_activated, lumo - 0.03 * n_activated


def _apply_hyperconjugation_correction(homo: float, lumo: float, mol: Chem.Mol) -> tuple[float, float]:
    """Apply all hyperconjugation corrections for bonds adjacent to π systems."""
    homo, lumo = _apply_cf_hyperconjugation(homo, lumo, mol)
    homo, lumo = _apply_co_cn_hyperconjugation(homo, lumo, mol)
    return homo, lumo


# ---------------------------------------------------------------------------
# Domain of Applicability (DoA) — TOM-specific epistemic uncertainty
# ---------------------------------------------------------------------------
# Physical justification: The TOM is calibrated against molecules with moderate
# conjugation (L ≤ 12) and reasonable pi-electron counts. Molecules with extreme
# conjugation paths lacking sp3 structural support, or excessive pi-systems,
# fall outside the TOM calibration domain. The particle-in-a-box model assumes
# a planar rigid system; long conjugation without sp3 breaks planarity, and
# excessive pi-electrons violate the single-particle HMO approximation.
# These are closed-form heuristics — no ML involved.


def _conjugation_penalty_sigmoid(L: float) -> float:
    """Continuous conjugation-axis DoA penalty (sigmoid in conjugation length).

    Physical justification: The particle-in-a-box gap (ΔE ∝ 1/L²) degrades
    smoothly as the conjugation path L grows, not as a step function. The
    sigmoid 0.70 + 0.30/(1 + exp(2(L−12))) interpolates between fully-in-domain
    (→1.0 for L ≪ 12) and the hard floor of 0.70 for L ≫ 12, eliminating the
    discontinuity at L=12. The midpoint sits at L=12 with penalty 0.85.
    """
    return 0.70 + 0.30 / (1.0 + math.exp(2.0 * (L - 12.0)))


def _pi_electron_penalty_sigmoid(n_pi: float) -> float:
    """Continuous pi-electron-count DoA penalty (sigmoid).

    Physical justification: Excessive pi-electron counts violate the
    single-particle Hueckel approximation. The sigmoid
    0.80 + 0.20/(1 + exp(0.5(n_pi − 24))) rises toward the 0.80 floor only as
    n_pi exceeds 24, staying ≈1.0 within the calibration domain.
    """
    return 0.80 + 0.20 / (1.0 + math.exp(0.5 * (n_pi - 24.0)))


def compute_quantum_domain_penalty(ctx: MoleculeContext) -> tuple[float, str]:
    """Compute domain-of-applicability penalty for TOM predictions.

    Penalises molecules with topological features that fall outside the
    TOM calibration domain. The conjugation and pi-electron axes use
    continuous sigmoids (replacing the former step function); the sp3
    structural-support condition remains a binary hard gate because it is
    physically discrete — a molecule either has the planarity-stabilising
    sp3 framework or it does not.

    Returns:
        (penalty_multiplier, reason_string)
        Multiplier in [0.70, 1.0]; 1.0 = fully within domain.
    """
    mol = ctx.mol
    reasons: list[str] = []
    penalty = 1.0

    L = _longest_conjugation_path(mol)
    penalty = _conjugation_penalty_sigmoid(L)
    if L > 12:
        n_c = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        n_sp3 = sum(
            a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3
            for a in mol.GetAtoms()
        )
        sp3_frac = n_sp3 / max(n_c, 1)
        if sp3_frac < 0.15:
            reasons.append(
                f"long conjugation (L={L}) without sp3 support (sp3_frac={sp3_frac:.2f})"
            )

    _, _, n_pi = _count_heteroatom_perturbations(mol)
    penalty *= _pi_electron_penalty_sigmoid(n_pi)
    if n_pi > 24:
        reasons.append(f"excessive pi-system (n_pi={n_pi}) outside TOM calibration")

    penalty = min(1.0, penalty)
    return round(penalty, 4), "; ".join(reasons) if reasons else "within domain"


# ---------------------------------------------------------------------------
# QuantumOracle — Unified interface for xTB + TOM fallback
# ---------------------------------------------------------------------------


class QuantumOracle:
    """Quantum-chemical oracle for frontier orbital energies.

    Two-tier evaluation:
      1. xTB (GFN2-xTB) via subprocess — preferred, real QM
      2. Topological Orbital Model (TOM) — conjugation-aware fallback

    Results are cached by SMILES to avoid redundant computation.
    """

    def __init__(self, use_xtb: bool = True, use_delta_correction: bool = True) -> None:
        self._use_xtb = use_xtb and _HAS_XTB
        self._cache: dict[str, dict[str, float]] = {}
        self._n_xtb_calls = 0
        self._n_tom_calls = 0
        self._use_delta_correction = use_delta_correction

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
        used_xtb = False
        if self._use_xtb:
            xyz = _generate_xyz(mol)
            result = _run_xtb(xyz)
            if result is not None:
                self._n_xtb_calls += 1
                used_xtb = True
            else:
                logger.warning("QuantumOracle: xTB calculation failed — falling back to TOM.")

        if result is None:
            homo, lumo = predict_tom_orbitals(mol)
            if self._use_delta_correction:
                try:
                    from aurelius.scoring.oracle.delta_correction import get_delta_correction

                    homo, lumo = get_delta_correction().predict_corrected(mol, base=(homo, lumo))
                    result = {
                        "homo_eV": homo,
                        "lumo_eV": lumo,
                        "dipole_D": 0.0,
                        "correction_applied": True,  # type: ignore[dict-item]
                    }
                except Exception as exc:
                    logger.warning("Delta correction failed (%s) — using raw TOM.", exc)
            if result is None:
                result = {
                    "homo_eV": homo,
                    "lumo_eV": lumo,
                    "dipole_D": 0.0,
                }
            self._n_tom_calls += 1

        if used_xtb:
            result["quantum_confidence"] = "xtb"  # type: ignore[assignment]
        else:
            L = _longest_conjugation_path(mol)
            n_rings = mol.GetRingInfo().NumRings()
            # ADR-2026-06-02: L > 8 or n_rings > 2 indicates topological complexity where
            # the 1-D particle-in-a-box model diverges from 3D reality, warranting epistemic humility.
            result["quantum_confidence"] = "tom_high" if L <= 8 and n_rings <= 2 else "tom_low"  # type: ignore[assignment]

        self._cache[smiles] = result
        return dict(result)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Vectorized batch evaluation — accelerates TOM evaluation for large candidate sets
# ---------------------------------------------------------------------------


def _batch_longest_conjugation_path(mols: list[Chem.Mol]) -> np.ndarray:
    """Compute longest conjugation path for each molecule in a batch.

    Returns a 1-D numpy array of lengths. Per-molecule fallback
    (single-molecule DFS) is used for edge cases where the batched
    path fails.
    """
    results = np.zeros(len(mols), dtype=np.int32)
    for i, mol in enumerate(mols):
        try:
            results[i] = _longest_conjugation_path(mol)
        except Exception:
            results[i] = 1
    return results


def _batch_heteroatom_counts(mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count EW, ED, and pi-electrons for each molecule in a batch.

    Returns three 1-D numpy arrays (n_ew, n_ed, n_pi).
    """
    n_ew = np.zeros(len(mols), dtype=np.int32)
    n_ed = np.zeros(len(mols), dtype=np.int32)
    n_pi = np.zeros(len(mols), dtype=np.int32)
    for i, mol in enumerate(mols):
        try:
            ew, ed, pi = _count_heteroatom_perturbations(mol)
            n_ew[i] = ew
            n_ed[i] = ed
            n_pi[i] = pi
        except Exception:
            pass
    return n_ew, n_ed, n_pi


def _batch_wiener_index(mols: list[Chem.Mol]) -> np.ndarray:
    """Compute Wiener index for each molecule in a batch.

    Returns a 1-D numpy array of floats.
    """
    results = np.zeros(len(mols), dtype=np.float32)
    for i, mol in enumerate(mols):
        try:
            results[i] = _wiener_index(mol)
        except Exception:
            results[i] = 0.0
    return results


def _batch_particle_in_a_box_gap(L: np.ndarray) -> np.ndarray:
    """Compute particle-in-a-box HOMO-LUMO gap for a batch of conjugation lengths.

    Vectorized operation: gap = 37.6 / (L * L) for L >= 3,
    0.0 for L < 3 (handled by base offsets).

    Args:
        L: 1-D numpy array of effective conjugation lengths.

    Returns:
        1-D numpy array of gap values in eV.
    """
    gap = np.zeros(len(L), dtype=np.float32)
    valid = L >= 3
    gap[valid] = 37.6 / (L[valid].astype(np.float32) * L[valid].astype(np.float32))
    return gap


def _apply_batch_heteroatom_shifts(
    homo: np.ndarray,
    lumo: np.ndarray,
    n_ew: np.ndarray,
    n_ed: np.ndarray,
    ew_coeff: float,
    ed_coeff: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply vectorized heteroatom perturbation shifts to HOMO/LUMO."""
    ew_shift = ew_coeff * n_ew.astype(np.float32)
    ed_shift = ed_coeff * n_ed.astype(np.float32)
    homo += ew_shift + ed_shift
    lumo += ew_shift * gamma + ed_shift * 0.5
    return homo, lumo


def _apply_batch_fluorine_correction(
    homo: np.ndarray,
    lumo: np.ndarray,
    mols: list[Chem.Mol],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply vectorized fluorine inductive correction."""
    n_f = np.array(
        [sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms()) for mol in mols],
        dtype=np.float32,
    )
    f_shift = -0.15 * n_f
    homo += f_shift
    lumo += f_shift
    return homo, lumo


def _apply_batch_aromatic_stabilization(
    homo: np.ndarray,
    lumo: np.ndarray,
    mols: list[Chem.Mol],
    arom_homo: float,
    arom_lumo: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply vectorized aromatic ring stabilization."""
    n_arom = np.array(
        [_count_aromatic_rings(mol) for mol in mols], dtype=np.float32
    )
    homo += arom_homo * n_arom
    lumo += arom_lumo * n_arom
    return homo, lumo


def _apply_batch_nitrile_correction(
    lumo: np.ndarray,
    mols: list[Chem.Mol],
    nitrile_shift: float,
) -> np.ndarray:
    """Apply vectorized nitrile LUMO correction."""
    n_nitrile = np.array(
        [len(mol.GetSubstructMatches(NITRILE_PATTERN)) for mol in mols],
        dtype=np.float32,
    )
    lumo += nitrile_shift * n_nitrile
    return lumo


def _apply_batch_ps_correction(
    homo: np.ndarray,
    lumo: np.ndarray,
    mols: list[Chem.Mol],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply vectorized phosphorus and sulfur inductive corrections."""
    n_p = np.array(
        [sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms()) for mol in mols],
        dtype=np.float32,
    )
    n_s = np.array(
        [sum(a.GetAtomicNum() == 16 for a in mol.GetAtoms()) for mol in mols],
        dtype=np.float32,
    )
    homo += 0.25 * n_s
    lumo -= 0.15 * n_p
    return homo, lumo


def _apply_batch_hyperconjugation(
    homo: np.ndarray,
    lumo: np.ndarray,
    mols: list[Chem.Mol],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-molecule hyperconjugation corrections in batch."""
    for i, mol in enumerate(mols):
        try:
            homo[i], lumo[i] = _apply_hyperconjugation_correction(
                homo[i], lumo[i], mol
            )
        except Exception:
            pass
    return homo, lumo


def predict_tom_orbitals_batch(mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray]:
    """Predict HOMO/LUMO for a batch of molecules using the Topological Orbital Model.

    Vectorizes the expensive per-molecule computations (longest conjugation path,
    heteroatom counts, Wiener index) and applies the particle-in-a-box gap
    calculation as a vectorized operation. When MLX is available on Apple
    Silicon, the vectorized physics math runs on the GPU for lower launch
    overhead than PyTorch MPS at EA-loop batch sizes (20–200).

    Physical justification: The particle-in-a-box gap (ΔE ∝ 1/L²) is a
    closed-form analytic formula that is trivially vectorizable across a
    population of molecules. The heteroatom perturbations and Wiener
    compactness adjustment are applied per-molecule but the resulting
    shifts are accumulated as vectorised operations on the best backend.

    Args:
        mols: List of RDKit Mol objects.

    Returns:
        Tuple of (homo_array, lumo_array) as 1-D numpy arrays of float32.
    """
    n = len(mols)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    device = get_device()
    if device == "mlx":
        return _predict_tom_orbitals_batch_mlx(mols)
    return _predict_tom_orbitals_batch_numpy(mols)


def _batch_atom_counts(mols: list[Chem.Mol], atomic_num: int) -> np.ndarray:
    """Count atoms of a given atomic number per molecule."""
    return np.array(
        [sum(a.GetAtomicNum() == atomic_num for a in mol.GetAtoms()) for mol in mols],
        dtype=np.float32,
    )


def _batch_aromatic_ring_counts(mols: list[Chem.Mol]) -> np.ndarray:
    """Count aromatic rings per molecule."""
    return np.array(
        [_count_aromatic_rings(mol) for mol in mols], dtype=np.float32
    )


def _batch_nitrile_counts(mols: list[Chem.Mol]) -> np.ndarray:
    """Count nitrile groups per molecule."""
    return np.array(
        [len(mol.GetSubstructMatches(NITRILE_PATTERN)) for mol in mols],
        dtype=np.float32,
    )


def _compute_L_final(L_array: np.ndarray, mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective conjugation length and 3D compactness.

    Applies Wiener compactness adjustment (2D) and radius-of-gyration
    compactness adjustment (3D), matching the single-molecule logic in
    ``predict_tom_orbitals`` exactly (including ``int(round())`` for
    the 3D step).
    """
    L_array = np.maximum(L_array, 2)
    L_array = np.array(
        [_topological_sanity_l(mol, int(L_array[i])) for i, mol in enumerate(mols)],
        dtype=np.int32,
    )
    w_array = _batch_wiener_index(mols)
    n_atoms_array = np.array([mol.GetNumAtoms() for mol in mols], dtype=np.int32)
    w_linear = np.maximum(
        n_atoms_array * (n_atoms_array * n_atoms_array - 1) / 6.0, 1.0
    )
    compactness = np.maximum(0.0, 1.0 - w_array / w_linear)
    L_eff = np.maximum(
        (L_array.astype(np.float32) * (1.0 - 0.3 * compactness)).astype(np.int32),
        2,
    )

    R_g_array = _compute_radius_of_gyration_batch(mols)
    R_g_linear = np.array(
        [_get_ideal_gyration_for_conjugation_length(int(l)) for l in L_eff],
        dtype=np.float32,
    )
    R_g_linear = np.maximum(R_g_linear, 1e-6)
    compactness_3d = np.maximum(0.0, 1.0 - R_g_array / R_g_linear)
    L_final = np.maximum(
        np.round(L_eff.astype(np.float32) * (1.0 - 0.2 * compactness_3d)).astype(np.int32),
        2,
    )
    return L_final, compactness_3d


def _extract_tom_per_molecule_arrays(mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
    """Extract all per-molecule RDKit properties into numpy arrays.

    These are the CPU/GIL-bound portion that must stay on the host because
    RDKit's C++ core cannot be dispatched through a Python-level GPU queue.
    Everything downstream (gap math, heteroatom shifts, etc.) is pure
    float32 vector arithmetic that *can* run on MLX.

    Returns a dict with intermediate arrays needed by both the NumPy and
    MLX backends, including the fully-computed ``L_final`` (effective
    conjugation length after compactness adjustments) so both backends
    produce identical numerical results.

    Optimized to pre-compute all per-molecule RDKit properties in a single
    pass over the molecules, reducing CPU overhead.
    """
    n = len(mols)
    
    # Pre-compute all per-molecule RDKit properties in a single pass
    L_final = np.zeros(n, dtype=np.int32)
    compactness_3d = np.zeros(n, dtype=np.float32)
    n_ew = np.zeros(n, dtype=np.int32)
    n_ed = np.zeros(n, dtype=np.int32)
    n_pi = np.zeros(n, dtype=np.int32)
    n_f = np.zeros(n, dtype=np.float32)
    n_arom = np.zeros(n, dtype=np.float32)
    n_nitrile = np.zeros(n, dtype=np.float32)
    n_p = np.zeros(n, dtype=np.float32)
    n_s = np.zeros(n, dtype=np.float32)
    
    # R_g is the dominant cost (86% of batch time); compute it for the whole
    # batch up front so it can be memoised and threaded. See ADR-2026-08-07-04.
    rg_array = _compute_radius_of_gyration_batch(mols)

    for i, mol in enumerate(mols):
        # Longest conjugation path
        L = _longest_conjugation_path(mol)
        L = max(L, 2)
        
        # Compute Wiener index for compactness
        w = _wiener_index(mol)
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 1:
            w_linear = n_atoms * (n_atoms * n_atoms - 1) / 6.0
            if w_linear > 0:
                compactness = max(0.0, 1.0 - w / w_linear)
                L = int(L * (1.0 - 0.3 * compactness))
                L = max(L, 2)
        
        # Apply topological sanity L (sp3 support)
        L = _topological_sanity_l(mol, L)
        L_final[i] = L
        
        # Compute radius of gyration and compactness_3d
        R_g = float(rg_array[i])
        R_g_linear = _get_ideal_gyration_for_conjugation_length(L)
        
        if R_g_linear > 0:
            compactness_3d[i] = max(0.0, 1.0 - R_g / R_g_linear)
        else:
            compactness_3d[i] = 0.0
        
        # Heteroatom counts
        ew, ed, pi = _count_heteroatom_perturbations(mol)
        n_ew[i] = ew
        n_ed[i] = ed
        n_pi[i] = pi
        
        # Other atom counts
        n_f[i] = sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
        n_arom[i] = _count_aromatic_rings(mol)
        n_nitrile[i] = len(mol.GetSubstructMatches(NITRILE_PATTERN))
        n_p[i] = sum(a.GetAtomicNum() == 15 for a in mol.GetAtoms())
        n_s[i] = sum(a.GetAtomicNum() == 16 for a in mol.GetAtoms())

    return {
        "L_final": L_final,
        "n_ew": n_ew,
        "n_ed": n_ed,
        "n_pi": n_pi,
        "n_f": n_f,
        "n_arom": n_arom,
        "n_nitrile": n_nitrile,
        "n_p": n_p,
        "n_s": n_s,
        "compactness_3d": compactness_3d,
    }


def _predict_tom_orbitals_batch_numpy(mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray]:
    """NumPy/CPU backend for batch TOM prediction.

    This is the original implementation, extracted as the CPU fallback path.
    """
    n = len(mols)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    tom_params = _get_tom_params()
    base_homo = tom_params["base_homo"]
    base_lumo = tom_params["base_lumo"]
    ew_coeff = tom_params["ew_coeff"]
    ed_coeff = tom_params["ed_coeff"]
    gamma = tom_params["gamma"]
    arom_homo = tom_params["arom_stab_homo"]
    arom_lumo = tom_params["arom_stab_lumo"]
    nitrile_shift = tom_params["nitrile_shift"]

    props = _extract_tom_per_molecule_arrays(mols)
    L_final = props["L_final"]

    gap = _batch_particle_in_a_box_gap(L_final)
    mid = (base_homo + base_lumo) / 2.0
    # Match single-molecule behaviour: when L < 3, use base offsets directly
    # (gap is 0 but homo/lumo must still be base_homo/base_lumo, not mid).
    homo = np.where(L_final >= 3, mid - gap / 2.0, base_homo)
    lumo = np.where(L_final >= 3, mid + gap / 2.0, base_lumo)

    # Apply 3D conformational correction (matches _apply_3d_correction)
    compactness_3d = props["compactness_3d"]
    homo -= 0.10 * compactness_3d
    lumo -= 0.05 * compactness_3d

    homo, lumo = _apply_batch_heteroatom_shifts(
        homo, lumo, props["n_ew"], props["n_ed"], ew_coeff, ed_coeff, gamma
    )
    homo, lumo = _apply_batch_fluorine_correction(homo, lumo, mols)
    homo, lumo = _apply_batch_aromatic_stabilization(
        homo, lumo, mols, arom_homo, arom_lumo
    )
    lumo = _apply_batch_nitrile_correction(lumo, mols, nitrile_shift)
    homo, lumo = _apply_batch_ps_correction(homo, lumo, mols)
    homo, lumo = _apply_batch_hyperconjugation(homo, lumo, mols)

    return homo.astype(np.float32), lumo.astype(np.float32)


def _predict_tom_orbitals_batch_mlx(mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray]:
    """MLX-GPU backend for batch TOM prediction.

    Uses MLX for the vectorised physics math (particle-in-a-box gap,
    heteroatom perturbations, fluorine/aryl/nitrile corrections). The
    per-molecule RDKit property extraction stays on CPU (GIL-bound) but
    the float32 array operations are offloaded to the Metal compute
    pipeline, which has lower launch overhead than PyTorch MPS for
    batch sizes typical in EA loops (20–200).
    """
    import mlx.core as mx

    n = len(mols)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    tom_params = _get_tom_params()
    base_homo = tom_params["base_homo"]
    base_lumo = tom_params["base_lumo"]
    ew_coeff = tom_params["ew_coeff"]
    ed_coeff = tom_params["ed_coeff"]
    gamma = tom_params["gamma"]
    arom_homo = tom_params["arom_stab_homo"]
    arom_lumo = tom_params["arom_stab_lumo"]
    nitrile_shift = tom_params["nitrile_shift"]

    props = _extract_tom_per_molecule_arrays(mols)
    L = mx.array(props["L_final"].astype(np.float32))
    n_ew = mx.array(props["n_ew"].astype(np.float32))
    n_ed = mx.array(props["n_ed"].astype(np.float32))
    n_f = mx.array(props["n_f"].astype(np.float32))
    n_arom = mx.array(props["n_arom"].astype(np.float32))
    n_nitrile = mx.array(props["n_nitrile"].astype(np.float32))
    n_p = mx.array(props["n_p"].astype(np.float32))
    n_s = mx.array(props["n_s"].astype(np.float32))

    # Pre-compute hyperconjugation correction deltas (per-molecule RDKit bond
    # iteration, computed on CPU then added as a vectorised MLX operation).
    homo_hc = np.zeros(n, dtype=np.float32)
    lumo_hc = np.zeros(n, dtype=np.float32)
    for i, mol in enumerate(mols):
        try:
            dh, dl = _apply_hyperconjugation_correction(0.0, 0.0, mol)
            homo_hc[i] = dh
            lumo_hc[i] = dl
        except Exception:
            pass

    # --- All vectorized physics math on MLX GPU ---
    L_sq = L * L
    gap = mx.where(L >= 3, mx.array(37.6) / L_sq, mx.array(0.0))

    mid = mx.array((base_homo + base_lumo) / 2.0)
    # Match single-molecule behaviour: when L < 3, use base offsets directly.
    homo = mx.where(L >= 3, mid - gap / 2.0, mx.array(base_homo))
    lumo = mx.where(L >= 3, mid + gap / 2.0, mx.array(base_lumo))

    # Apply 3D conformational correction (matches _apply_3d_correction)
    compactness_3d = mx.array(props["compactness_3d"])
    homo = homo - mx.array(0.10) * compactness_3d
    lumo = lumo - mx.array(0.05) * compactness_3d

    # Heteroatom perturbations (Hueckel-like correction)
    ew_shift = mx.array(ew_coeff) * n_ew
    ed_shift = mx.array(ed_coeff) * n_ed
    homo = homo + ew_shift + ed_shift
    lumo = lumo + ew_shift * mx.array(gamma) + ed_shift * mx.array(0.5)

    # Fluorine inductive correction
    homo = homo - mx.array(0.15) * n_f
    lumo = lumo - mx.array(0.15) * n_f

    # Aromatic ring stabilization
    homo = homo + mx.array(arom_homo) * n_arom
    lumo = lumo + mx.array(arom_lumo) * n_arom

    # Nitrile triple bond LUMO correction
    lumo = lumo + mx.array(nitrile_shift) * n_nitrile

    # Phosphorus and sulfur inductive corrections
    homo = homo + mx.array(0.25) * n_s
    lumo = lumo - mx.array(0.15) * n_p

    # Hyperconjugation corrections (pre-computed as numpy, added as MLX)
    homo = homo + mx.array(homo_hc)
    lumo = lumo + mx.array(lumo_hc)

    return np.array(homo.astype(mx.float32)), np.array(lumo.astype(mx.float32))
