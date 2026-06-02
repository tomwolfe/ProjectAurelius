"""Group-Contribution (Fragment-Additivity) Models — Bulk Properties Only.

Predicts dielectric, viscosity, and Li+ solvation via functional-group
additivity with Michaelis-Menten saturation and non-linear cross-terms.
"""

from __future__ import annotations

import math

from rdkit import Chem

from aurelius.constants import MAX_DIELECTRIC_PER_TPSA
from aurelius.types import MoleculeContext

_DATA_SOURCE: str = "hybrid (GC bulk + Quantum orbital)"


def get_data_source() -> str:
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models — Bulk Properties Only
# ---------------------------------------------------------------------------

# (pattern, name, dielectric_contrib, viscosity_contrib, li_solvation_contrib)
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
_GC_SATURATION_K: float = 0.693  # ln(2), half-max at count=1


def _saturate_contrib(count: int, max_contrib: float) -> float:
    """Michaelis-Menten style saturation for fragment additivity."""
    return max_contrib * (1.0 - math.exp(-_GC_SATURATION_K * count))


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each pre-compiled fragment pattern in a molecule."""
    counts: dict[str, int] = {}
    for pattern, name, _dd, _dv, _ls in _GC_FRAGMENTS:
        matches = mol.GetSubstructMatches(pattern)
        counts[name] = len(matches)
    return counts


# Non-linear cross-term corrections for dielectric proxy.
_CROSS_TERMS: list[tuple[str, str, float, str]] = [
    ("carbonate", "ether", 0.8, "carbonate-ether synergy (glyme-carbonate hybrids)"),
    ("nitrile", "ether", 0.3, "nitrile-ether synergy"),
    ("carbonate", "fluorine", -0.5, "fluorinated carbonate suppression"),
    ("sulfone", "ether", 0.4, "sulfone-ether synergy"),
    ("carbonate", "nitrile", -0.3, "carbonate-nitrile antagonism"),
    ("alcohol", "carbonate", -0.4, "alcohol-carbonate H-bond competition"),
    ("sulfone", "carbonate", -0.3, "sulfone-carbonate polarity competition"),
]


def _compute_dielectric_cross_terms(counts: dict[str, int]) -> float:
    """Compute non-linear cross-term contributions to dielectric proxy."""
    correction = 0.0
    for frag_a, frag_b, boost, _desc in _CROSS_TERMS:
        if counts.get(frag_a, 0) > 0 and counts.get(frag_b, 0) > 0:
            correction += boost
    return correction


def predict_dielectric_proxy(ctx: MoleculeContext) -> float:
    """Predict a dielectric constant proxy via fragment-additivity + TPSA cap
    + non-linear cross-term corrections.
    """
    mol = ctx.mol
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, dd, _dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += _saturate_contrib(n, dd * 2.0)

    value += _compute_dielectric_cross_terms(counts)

    tpsa = ctx.tpsa
    value += tpsa * 0.02

    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)

    return max(1.0, value)


def _count_branch_points(mol: Chem.Mol) -> int:
    """Count topological branch points (sp3 atoms with degree >= 3)."""
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetDegree() >= 3
        and a.GetHybridization() == Chem.HybridizationType.SP3
    )


def _count_stereocenters(mol: Chem.Mol) -> int:
    """Count atom stereocenters."""
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)


def predict_viscosity_proxy(ctx: MoleculeContext) -> float:
    """Predict a viscosity proxy via fragment-additivity + branching penalty."""
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

    n_branch = _count_branch_points(mol)
    value += n_branch * 0.80

    n_stereo = _count_stereocenters(mol)
    value += n_stereo * 0.05

    return max(0.1, value)


def predict_li_solvation_proxy(ctx: MoleculeContext) -> float:
    """Predict a Li+ solvation energy proxy via fragment-additivity."""
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
# Mixture Property Prediction — Ideal Thermodynamic Mixing Rules
# ---------------------------------------------------------------------------
# These implement volume-fraction weighted averaging for dielectric and
# Li+ solvation (extensive properties in ideal mixtures) and log-linear
# (Grunberg-Nissan) mixing for viscosity.


def predict_mixture_dielectric(d1: float, d2: float, frac1: float = 0.5) -> float:
    """Ideal volume-fraction weighted dielectric for a binary mixture."""
    return frac1 * d1 + (1.0 - frac1) * d2


def predict_mixture_viscosity(v1: float, v2: float, frac1: float = 0.5) -> float:
    """Ideal log-linear (Grunberg-Nissan) viscosity for a binary mixture."""
    v1_s = max(v1, 0.001)
    v2_s = max(v2, 0.001)
    ln_mix = frac1 * math.log(v1_s) + (1.0 - frac1) * math.log(v2_s)
    return math.exp(ln_mix)


def predict_mixture_li_solvation(ls1: float, ls2: float, frac1: float = 0.5) -> float:
    """Additive Li+ solvation for a binary mixture."""
    return frac1 * ls1 + (1.0 - frac1) * ls2


def mixture_synergy_bonus(
    d1: float, d2: float, v1: float, v2: float, frac1: float = 0.5
) -> float:
    """Non-linear synergy bonus for complementary binary electrolyte mixtures.

    A complementary pair combines a high-dielectric component (d > 4.0) with
    a low-viscosity component (v < 1.5). Neither pure component simultaneously
    achieves both benefits because polar groups that raise dielectric also
    increase viscosity. The mixture's synergy reflects this thermodynamic
    complementarity — it is a non-linear effect that single-molecule scoring
    cannot capture.

    The bonus scales with how well the mixture's combined dielectric and
    viscosity meet both targets simultaneously.
    """
    has_high_d = max(d1, d2) > 4.0
    has_low_v = min(v1, v2) < 1.5

    if not (has_high_d and has_low_v):
        return 0.0

    f2 = 1.0 - frac1
    d_mix = frac1 * max(d1, 0.0) + f2 * max(d2, 0.0)
    v_mix = math.exp(
        frac1 * math.log(max(v1, 0.001)) + f2 * math.log(max(v2, 0.001))
    )

    score = d_mix / 4.0 + 1.5 / max(v_mix, 0.01)
    return min(max(0.0, score), 6.0)
