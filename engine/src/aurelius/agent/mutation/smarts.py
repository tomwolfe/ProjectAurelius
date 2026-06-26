"""SMARTS definitions and electrolyte-likeness validators for mutation.

Contains:
  - Electrolyte-relevant SMARTS reaction library
  - Data-driven electrolyte-likeness check registry
  - Anti-gaming topology helpers (conjugation limits)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from aurelius.constants import (
    ELECTROCHEMICALLY_UNSTABLE_PATTERNS as _EC_UNSTABLE_PATTERNS,
)
from aurelius.constants import (
    ELECTROLYTE_MIN_HETEROATOM_RATIO,
)
from aurelius.constants import (
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_UNSTABLE_PATTERNS,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS reaction library
# ---------------------------------------------------------------------------
ELECTROLYTE_SMARTS: list[tuple[str, str]] = [
    ("[CH3:1]>>[F:1]", "Methyl to fluorine"),
    ("[CH3:1]>>[C:1](F)(F)F", "Methyl to trifluoromethyl"),
    ("[OH:1]>>[F:1]", "Hydroxyl to fluorine"),
    ("[C:1]>>[C:1]OC(F)(F)F", "Add trifluoromethoxy"),
    ("[CH2:1]>>[C:1](F)F", "Methylene to difluoromethylene"),
    ("[C:1](=O)[O:2]>>[C:1](=O)[O:2]C", "Ester to methyl ester"),
    ("[C:1](=O)[OH:1]>>[C:1](=O)[O:1]C", "Carboxylic acid to methyl ester"),
    ("[OH:1]>>[O:1]C(=O)OC", "Hydroxyl to carbonate"),
    ("[OH:1]>>[O:1]C(=O)OCC", "Hydroxyl to ethyl carbonate"),
    ("[OH:1]>>[O:1]C(=O)OC(F)(F)F", "Hydroxyl to fluorinated carbonate"),
    ("[C:1]>>[C:1]OC", "Add methoxy"),
    ("[C:1]>>[C:1]OCC", "Add ethoxy"),
    ("[C:1]>>[C:1]OCCOC", "Add diethylene glycol ether"),
    ("[C:1]>>[C:1]S(=O)(=O)C", "Add methyl sulfone"),
    ("[C:1]>>[C:1]S(=O)(=O)F", "Add sulfonyl fluoride"),
    ("[C:1]>>[C:1]S(=O)(=O)CF", "Add fluoromethyl sulfone"),
    ("[Br:1]>>[C:1]#N", "Bromo to nitrile"),
    ("[C:1]I>>[C:1]#N", "Iodo to nitrile"),
    ("[C:1]>>[C:1]C#N", "Add acetonitrile"),
    ("[OH:1]>>[O:1]P(=O)(OC)OC", "Hydroxyl to dimethyl phosphate"),
    ("[C:1]>>[C:1](C)", "Methylation"),
    ("[C:1]>>[C:1]CC", "Ethylation"),
    # ------------------------------------------------------------------
    # Scaffold-hopping reactions — modify core ring topology
    # ------------------------------------------------------------------
    # These transform 1,4-dioxane (the seed's only ring) into molecules
    # with different ring scaffolds: 1,4-oxathiane, tetrahydropyran,
    # morpholine, and linear ethers. Each product has a novel Murcko
    # scaffold relative to the seed pool.
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[S:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1",
        "oxygen_to_sulfur_in_dioxane",
    ),
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[CH2:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1",
        "oxygen_to_methylene_in_dioxane",
    ),
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[N:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1",
        "oxygen_to_amine_in_dioxane",
    ),
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[CH3:7][O:1][CH2:2][CH2:3][O:4][CH2:5][CH2:6][CH3:8]",
        "dioxane_ring_opening",
    ),
    # Scaffold-hopping: ring-closure of dimethyl carbonate to ethylene carbonate
    ("[CH3:1]OC(=O)O[CH3:2]>>[CH2:1]1OC(=O)O[CH2:2]1", "dimethyl_carbonate_to_ec"),
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[Se:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1",
        "oxygen_to_selenium_in_dioxane",
    ),
    (
        "[O:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1>>[S:1](=O)1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1",
        "oxygen_to_sulfoxide_in_dioxane",
    ),
    # General functional group transformations
    # Enables chain elongation and new functionality across seeds
    ("[CH3:1]>>[CH3:1][CH2:2]", "methyl_to_ethyl"),
    ("[OH:1]>>[O:1]S(=O)(=O)C", "hydroxyl_to_methyl_sulfonate"),
    ("[C:1]#[N:2]>>[C:1](=[O:2])[NH2:3]", "nitrile_to_primary_amide"),
]

# ---------------------------------------------------------------------------
# Electrochemical Stability SMARTS — filter during mutation to save compute
# ---------------------------------------------------------------------------

ELECTROLYTE_FRAGMENT_POOL: list[str] = [
    "COC(=O)OC",
    "CCOC(=O)OCC",
    "O=C1OCCCO1",
    "O=C1OCCO1",
    "O=C1OC(F)CO1",
    "FC(F)(F)OCOC(=O)OC(F)(F)F",
    "CCOC",
    "CCOCC",
    "COCCOC",
    "COCCOCCOC",
    "C1CCOC1",
    "C1COCCO1",
    "CS(=O)(=O)C",
    "CS(=O)(=O)CC",
    "FC(F)(F)S(=O)(=O)C(F)(F)F",
    "CC#N",
    "N#CCC#N",
    "N#CCCC#N",
    "COS(=O)(=O)OC",
    "CF",
    "C(F)(F)F",
    "CC(F)(F)F",
    "COP(=O)(OC)OC",
    "OB(OC)OC",
    "O=C1OC=CO1",
    "O=S1(=O)OCC1",
    "O=S1(=O)OCCO1",
    "FC(F)(F)C(F)(F)F",
    "FC(F)(F)C(F)(F)C(F)(F)F",
]

# ---------------------------------------------------------------------------
# Anti-gaming topology helpers
# ---------------------------------------------------------------------------


def find_max_conjugated_path(mol: Chem.Mol) -> int:
    """Find the longest conjugated pi-system in a molecule (atom count).

    Prevents the mutation engine from creating infinitely conjugated
    structures that would "game" additive property models.
    """
    visited: set[int] = set()
    max_path = [0]

    def _conjugated(a: Chem.Atom, b: Chem.Atom) -> bool:
        bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
        if bond is None:
            return False
        if bond.GetIsConjugated():
            return True
        bt = bond.GetBondType()
        if bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
            return True
        return a.GetIsAromatic() or b.GetIsAromatic()

    def _dfs(idx: int, length: int) -> None:
        visited.add(idx)
        max_path[0] = max(max_path[0], length)
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            n_idx = nb.GetIdx()
            if n_idx not in visited and _conjugated(atom, nb):
                _dfs(n_idx, length + 1)
        visited.discard(idx)

    for atom in mol.GetAtoms():
        _dfs(atom.GetIdx(), 1)

    return max_path[0]


# ---------------------------------------------------------------------------
# Data-driven electrolyte-likeness validators
# ---------------------------------------------------------------------------
# Each validator is a (name, predicate) tuple. The predicate receives a
# MoleculeContext and returns True if the check passes (electrolyte-like).
# This replaces a 60-line wall of sequential if-blocks with a declarative,
# composable rule set that is easy to audit, extend, or prune.
#
# NOTE: These are lightweight pre-compute gates to prevent the EA from
# wasting cycles on obviously invalid topologies during mutation. The
# definitive, strict viability gate is src/aurelius/screening/tier1/filter.py.

_ELECTROLYTE_CHECKS: list[tuple[str, Callable[[MoleculeContext], bool]]] = []


def _register(fn: Callable[[MoleculeContext], bool]) -> Callable[[MoleculeContext], bool]:
    _ELECTROLYTE_CHECKS.append((fn.__name__, fn))
    return fn


@_register
def aromatic_ring_limit(ctx: MoleculeContext) -> bool:
    return rdMolDescriptors.CalcNumAromaticRings(ctx.mol) <= 2


@_register
def has_heteroatom(ctx: MoleculeContext) -> bool:
    hetero_atoms = {8, 9, 15, 16}
    return sum(a.GetAtomicNum() in hetero_atoms for a in ctx.mol.GetAtoms()) >= 1


@_register
def heteroatom_ratio_min(ctx: MoleculeContext) -> bool:
    n_total = sum(a.GetAtomicNum() > 1 for a in ctx.mol.GetAtoms())
    if n_total == 0:
        return True
    o_f = sum(a.GetAtomicNum() in (8, 9) for a in ctx.mol.GetAtoms())
    return o_f / n_total >= ELECTROLYTE_MIN_HETEROATOM_RATIO


def _count_by_atomic_num(mol: Chem.Mol, nums: set[int]) -> int:
    return sum(a.GetAtomicNum() in nums for a in mol.GetAtoms())


_HALOGEN_NUMS: set[int] = {9, 17, 35}
_HEAVY_HALOGEN_NUMS: set[int] = {17, 35}
_OXYGEN_NITROGEN: set[int] = {7, 8}


@_register
def halogen_ratio_limit(ctx: MoleculeContext) -> bool:
    n_total = _count_by_atomic_num(ctx.mol, set(range(2, 118)))
    if n_total == 0:
        return True
    n_halogen = _count_by_atomic_num(ctx.mol, _HALOGEN_NUMS)
    n_heavy = _count_by_atomic_num(ctx.mol, _HEAVY_HALOGEN_NUMS)
    if n_halogen / n_total > 0.9:
        return False
    if n_heavy / n_total > 0.5:
        return False
    return not (n_halogen > n_total * 0.6 and _count_by_atomic_num(ctx.mol, _OXYGEN_NITROGEN) == 0)


@_register
def electrochemically_stable(ctx: MoleculeContext) -> bool:
    for pattern, _name in _EC_UNSTABLE_PATTERNS:
        if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
            return False
    return True


@_register
def hydrolytically_stable(ctx: MoleculeContext) -> bool:
    for pattern, _name, _severity in _HYDRO_UNSTABLE_PATTERNS:
        if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
            return False
    return True


@_register
def ring_strain_limit(ctx: MoleculeContext) -> bool:
    ring_info = ctx.mol.GetRingInfo()
    if ring_info is not None and ring_info.NumRings() > 0:
        for ring in ring_info.BondRings():
            if len(ring) <= 4:
                return False
        if ring_info.NumRings() > 3:
            return False
    return True


@_register
def conjugation_limit(ctx: MoleculeContext) -> bool:
    return find_max_conjugated_path(ctx.mol) <= 16


@_register
def sp3_fraction_min(ctx: MoleculeContext) -> bool:
    n_sp3 = sum(a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3 for a in ctx.mol.GetAtoms())
    n_c = sum(a.GetAtomicNum() == 6 for a in ctx.mol.GetAtoms())
    if n_c >= 4:
        return n_sp3 / n_c >= 0.20
    return True


@_register
def valence_sanity(ctx: MoleculeContext) -> bool:
    max_valence: dict[int, int] = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1}
    for atom in ctx.mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in max_valence and atom.GetExplicitValence() > max_valence[z]:
            return False
    return True


@_register
def polarity_ratio_min(ctx: MoleculeContext) -> bool:
    mw = ctx.mw
    tpsa = ctx.tpsa
    return not (mw > 200 and tpsa / mw < 0.05)


# ---------------------------------------------------------------------------
# Anti-gaming: Steric Hindrance Check
# ---------------------------------------------------------------------------
# Physical basis: Molecules with severe steric clashes have unrealistic
# 3D geometries. The mutation engine can generate hyper-branched or
# overcrowded molecules that "game" additive property models by packing
# many polar groups in a small volume, but are synthetically inaccessible
# and physically unrealistic. UFF energy minimization is used as a
# quick filter — molecules with UFF energy > 500 kcal/mol likely contain
# severe Van der Waals clashes or extreme branching near polar groups.
# This is a pre-compute gate; the definitive steric check happens during
# xTB/DFT geometry optimization in the scoring tier.

_STERIC_ENERGY_CUTOFF: float = 500.0  # kcal/mol


@_register
def steric_hindrance_check(ctx: MoleculeContext) -> bool:
    """Reject molecules with severe steric clashes via UFF energy.

    Generates a 3D conformer using ETKDGv3 with random coordinates,
    runs UFF minimization (maxIts=200), and checks the final energy.
    Returns False if the energy exceeds the steric cutoff, indicating
    severe Van der Waals clashes.

    Molecules with MW < 150 are exempted (too small for steric crowding).
    """
    if ctx.mw < 150:
        return True
    mol = ctx.mol
    try:
        mol_copy = Chem.RWMol(mol)
        mol_copy.UpdatePropertyCache()
        mol_copy = Chem.AddHs(mol_copy)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol_copy, params)
        if result != 0:
            return True
        ff = AllChem.UFFGetMoleculeForceField(mol_copy)
        if ff is None:
            return True
        ff.Minimize(maxIts=200)
        energy = ff.CalcEnergy()
        return energy < _STERIC_ENERGY_CUTOFF
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Anti-gaming: Cross-Conjugation Check
# ---------------------------------------------------------------------------
# Physical basis: Cross-conjugated systems (e.g., divinyl ketones,
# benzophenone) have branched pi-electron networks where the 1D
# particle-in-a-box model used by TOM fails catastrophically. The
# particle-in-a-box assumes linear pi-delocalisation; cross-conjugation
# branches the pi system, invalidating the 1/L² gap scaling. These
# molecules can "game" the TOM predictor by appearing to have a long
# conjugation path while actually having a much wider gap.
#
# We detect cross-conjugation by finding any atom with >2 conjugated
# neighbors where at least 2 of those are non-oxygen (carbon) conjugated
# substituents. This excludes normal resonance in esters and carbonates
# (O-C(=O)-O) where oxygen lone-pair donation is not true cross-conjugation.


def _is_conjugated_bond(mol: Chem.Mol, i: int, j: int) -> bool:
    """Check if bond between atoms i and j is conjugated."""
    bond = mol.GetBondBetweenAtoms(i, j)
    if bond is None:
        return False
    if bond.GetIsConjugated():
        return True
    bt = bond.GetBondType()
    if bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
        return True
    a1 = mol.GetAtomWithIdx(i)
    a2 = mol.GetAtomWithIdx(j)
    return bool(a1.GetIsAromatic() or a2.GetIsAromatic())


def _has_cross_conjugation(mol: Chem.Mol) -> bool:
    """Check if molecule contains a cross-conjugated pi system.

    Cross-conjugation occurs when a conjugated atom has >2 conjugated
    neighbors with at least 2 of those being non-oxygen (carbon)
    conjugated substituents. This excludes esters/carbonates where
    oxygen lone-pair donation creates resonance but not cross-conjugation.
    """
    for atom in mol.GetAtoms():
        n_conj = 0
        n_conj_carbon = 0
        for nb in atom.GetNeighbors():
            if _is_conjugated_bond(mol, atom.GetIdx(), nb.GetIdx()):
                n_conj += 1
                if nb.GetAtomicNum() == 6:
                    n_conj_carbon += 1
        if n_conj > 2 and n_conj_carbon >= 2:
            return True
    return False


# Pre-compiled SMARTS for known cross-conjugated motifs
_CROSS_CONJ_MOTIFS: list[Chem.Mol] = [
    Chem.MolFromSmarts("[CH2]=[CH]C(=O)[CH]=[CH2]"),     # divinyl ketone
    Chem.MolFromSmarts("O=C(c1ccccc1)c2ccccc2"),          # benzophenone
    Chem.MolFromSmarts("[CH2]=[CH]-[CX3]1[CX3]=[CX3][CX3]=[CX3]1"),  # vinyl-aromatic ketone
    Chem.MolFromSmarts("[CH2]=[CH]C(=C)[CH]=[CH2]"),      # divinyl methylene (fulvene-like)
]


@_register
def cross_conjugation_check(ctx: MoleculeContext) -> bool:
    """Reject molecules with cross-conjugated pi systems.

    Cross-conjugated molecules are penalised because the 1D particle-in-a-box
    model (TOM) overestimates their HOMO-LUMO gap. Both a topology-based
    detection algorithm AND SMARTS motif matching are used for coverage.
    """
    if _has_cross_conjugation(ctx.mol):
        return False
    return all(
        not (motif is not None and ctx.mol.HasSubstructMatch(motif))
        for motif in _CROSS_CONJ_MOTIFS
    )


def is_electrolyte_like(ctx: MoleculeContext) -> bool:
    return all(check(ctx) for _name, check in _ELECTROLYTE_CHECKS)
