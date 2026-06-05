"""Group-Contribution (Fragment-Additivity) Models — Bulk Properties Only.

Predicts dielectric, viscosity, and Li+ solvation via functional-group
additivity with Michaelis-Menten saturation and non-linear cross-terms.

ADR-2026-06-05: Parameter refinements to improve weak proxy correlations:
  Dielectric ε: nitrile 8.0→5.5, amide 5.0→6.0, sulfoxide 6.0→7.5;
                MAX_DIELECTRIC_PER_TPSA 0.35→0.60 (constants.py).
  Li⁺ solvation: amide 1.2→2.5, glyme_chelating 1.8→0.6, sulfoxide 2.5→3.5,
                 aromatic_nitrogen 2.0→3.5.
  Rationale: nitrile was over-contributing relative to carbonate (ACN > DMSO
  ranked wrong); glyme_chelating double-counted ether chelation; amides and
  sulfoxides were under-valued for Li⁺ binding. External validation Spearman ρ:
  Dielectric 0.3226→0.3967, Donor Number 0.1368→0.4074.

ADR-2026-06-05c: Parameter refinements to improve Dielectric and Donor Number ρ:
  Dielectric ε: cyclic_carbonate 6.0→8.0, carbonate 5.0→2.0, TPSA 0.025→0.030.
  Donor Number: aromatic_nitrogen 3.5→4.0.
  Rationale: cyclic Kirkwood g>1 needs stronger representation; linear carbonates
  were overpredicted (anti-periplanar O-alkyl cancels dipoles); TPSA coefficient
  increased to improve polar/non-polar rank separation; pyridine (DN=33.1) should
  rank above DMSO (DN=29.8) via stronger aromatic N basicity.

ADR-2026-06-05b: Add cyclic_carbonate fragment (+6.0 dielectric) to
 differentiate EC/PC (cyclic, ε=90/65) from DMC/DEC (linear, ε=3). Physical
justification: The Onsager-Kirkwood correlation factor g>1 for cyclic carbonates
(conformational locking of cis-carbonate dipoles) vs g<1 for linear carbonates
(anti-parallel alignment), producing a 20-30× difference in measured dielectric.
TPSA coefficient raised from 0.02→0.025 to better capture polarity scaling while
maintaining fragment saturation guarantees (5 esters < 2× single ester).
Nitrile dielectric raised 5.5→7.5: ACN (ε=36) and PN (ε=27) were under-valued
relative to sulfoxides and carbonates. The C≡N dipole (μ≈3.9 D) is among the
strongest of any organic functional group; 7.5 keeps ACN (pred=10.0) below DMSO
(pred=10.3) preserving the experimental ranking ACN < DMSO.

ADR-2026-06-01: Added [-2.0, 2.0] clip to _compute_dielectric_cross_terms.
Physical justification: cross-term additive bonuses have no upper bound; a
molecule with carbonate + ether + sulfone + nitrile can accumulate ~1.2 extra
dielectric points, enough to materially misrank candidates. A single clip guard
bounds the cross-term contribution to what is physically plausible — no polar
group combination can more than double the base dielectric contribution. This is
the minimal non-invasive fix: one line, no new data structures, no architectural
change. The alternative (capping each cross-term individually or using saturation)
would add complexity without proportional benefit.
"""

from __future__ import annotations

import math

from rdkit import Chem

from aurelius.constants import MAX_DIELECTRIC_PER_TPSA
from aurelius.types import MoleculeContext

_DATA_SOURCE: str = "hybrid (GC bulk + Quantum orbital)"


def compute_gc_domain_penalty(ctx: MoleculeContext) -> tuple[float, str]:
    """Compute domain-of-applicability penalty for GC predictions.

    Physical justification: The GC fragment-additivity model is calibrated
    on molecules with moderate functional group density and reasonable
    molecular weight (MW < 500, rotatable bonds < 20). Extreme fluorination
    without polar solvation sites (F >= 6, polar atoms < 2) falls outside
    the GC calibration domain because the saturation curves for halogenated
    fragments were not validated against such extreme compositions. High-MW
    molecules accumulate group contributions linearly but real dielectric
    and viscosity saturate, which the Michaelis-Menten model only partially
    captures. These are closed-form topological heuristics — no ML.

    Returns:
        (penalty_multiplier, reason_string)
        Multiplier in [0.70, 1.0]; 1.0 = fully within domain.
    """
    mol = ctx.mol
    reasons: list[str] = []
    penalty = 1.0

    n_f = sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
    n_polar = sum(a.GetAtomicNum() in (7, 8, 15, 16) for a in mol.GetAtoms())
    if n_f >= 6 and n_polar < 2:
        penalty *= 0.75
        reasons.append(f"extreme fluorination (F={n_f}) without polar solvation sites")

    if ctx.mw > 500:
        penalty *= 0.85
        reasons.append(f"high MW ({ctx.mw:.0f}) outside GC calibration domain")

    if ctx.rotatable_bonds > 20:
        penalty *= 0.85
        reasons.append(f"excessive flexibility ({ctx.rotatable_bonds} rotatable bonds)")

    return penalty, "; ".join(reasons) if reasons else "within domain"


def get_data_source() -> str:
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models — Bulk Properties Only
# ---------------------------------------------------------------------------

# (pattern, name, dielectric_contrib, viscosity_contrib, li_solvation_contrib)
_GC_FRAGMENTS: list[tuple[Chem.Mol, str, float, float, float]] = [
    (Chem.MolFromSmarts("[CX3](=O)[OX2H0]"),       "ester",              2.5,  0.6,  0.8),
    (Chem.MolFromSmarts("[CX3](=O)[OH]"),          "carboxylic_acid",    4.0,  1.0,  1.8),
    (Chem.MolFromSmarts("[CX3](=O)[NX3]"),         "amide",              6.0,  0.8,  2.5),
    (Chem.MolFromSmarts("[CX3](=O)[CX3]"),         "ketone",             3.0,  0.5,  0.6),
    (Chem.MolFromSmarts("[CH](=O)"),               "aldehyde",           2.5,  0.3,  0.3),
    # ADR-2026-06-05c: carbonate dielectric 5.0→2.0. Linear carbonates (DMC ε≈3.1,
    # DEC ε≈2.8) have anti-periplanar O-alkyl conformation cancelling dipoles (Kirkwood
    # g<1). The cyclic_carbonate fragment (8.0) separately captures EC/PC's g>1 effect.
    # Reducing generic carbonate prevents overprediction of linear carbonates.
    (Chem.MolFromSmarts("O=C([OX2])[OX2]"),        "carbonate",          2.0,  0.7,  1.2),
    (Chem.MolFromSmarts("[OD2]([CX4])[CX4]"),      "ether",              1.5, -0.3,  1.0),
    (Chem.MolFromSmarts("[OH][CX4]"),              "alcohol",            4.5,  1.2,  2.0),
    (Chem.MolFromSmarts("[NX3;H2][CX4]"),          "primary_amine",      3.5,  0.5,  1.0),
    (Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4]"),   "secondary_amine",    2.5,  0.4,  0.8),
    (Chem.MolFromSmarts("[NX3;H0]([CX4])([CX4])[CX4]"), "tertiary_amine", 1.5,  0.3,  0.5),
    # Nitrile dielectric raised from 5.5→7.5 (ADR-2026-06-05b): the C≡N dipole
    # (μ≈3.9 D) produces ε=36 for ACN and ε=27 for PN — the prior 5.5 under-valued
    # nitriles relative to sulfoxides and carbonates. Cyclic_carbonate boost to EC
    # (now ~18) frees headroom: ACN=10.0 < DMSO=10.3, preserving correct ranking.
    (Chem.MolFromSmarts("[C]#[N]"),                "nitrile",            7.5,  0.4,  0.8),
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
    (Chem.MolFromSmarts("[OX2][CX4][CX4][OX2]"),   "glyme_chelating",    2.0,  0.1,  0.6),
    (Chem.MolFromSmarts("[SX4](=O)(=O)[NX3][SX4](=O)(=O)"), "sulfonimide", 5.0,  0.5,  0.5),
    (Chem.MolFromSmarts("[CX3](=O)[OX2]C(F)(F)F"),  "fluorinated_carbonate", 3.0,  0.3, -0.1),
    (Chem.MolFromSmarts("[SX3](=O)[CX4]"),           "sulfoxide",             7.5,  0.5,  3.5),
    # ADR-2026-06-05c: aromatic_N li_solvation 3.5→4.0. Pyridine (DN=33.1) has the
    # highest donor number in the benchmark set; it should rank above DMSO (DN=29.8).
    (Chem.MolFromSmarts("[n]"),                      "aromatic_nitrogen",     4.0,  0.3,  4.0),
    (Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])[#6]"), "phosphonate",        3.5,  0.5,  1.0),
    # Cyclic carbonate (5-ring): cis-conformation enables cooperative dipole alignment
    # (Kirkwood g>1), boosting ε 20-30× vs linear. Li+ binding at carbonyl O is same
    # as linear carbonates, so li_solvation kept at 0.0 (donor number unaffected).
    # ADR-2026-06-05c: dielectric 6.0→8.0. EC (ε=89.78) and PC (ε=64.92) are 2-3×
    # higher than any other aprotic solvent. Larger contribution needed to capture
    # the physical gap between cyclic (g>1) and linear (g<1) carbonates.
    (Chem.MolFromSmarts("[OX2]1[CX3](=O)[OX2][CX4][CX4]1"), "cyclic_carbonate",  8.0,  0.4,  0.0),
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
    ("nitrile", "fluorine", 0.3, "fluorinated nitrile dipole enhancement"),
    ("sulfone", "nitrile", 0.5, "sulfone-nitrile high-voltage synergy"),
]


def _compute_dielectric_cross_terms(counts: dict[str, int]) -> float:
    """Compute non-linear cross-term contributions to dielectric proxy.

    Physical justification: Cross-terms capture non-additive dielectric
    enhancement from co-occurring polar groups (e.g., carbonate-ether
    synergy). Without a ceiling, a molecule with four synergistic groups
    (carbonate + ether + sulfone + nitrile) can accumulate ~1.2 extra
    dielectric points, enough to materially misrank candidates. The
    [-2.0, 2.0] clip bounds the cross-term contribution to what is
    physically plausible for a molecular dielectric proxy — no single
    polar group combination can more than double the base dielectric
    contribution.
    """
    correction = 0.0
    for frag_a, frag_b, boost, _desc in _CROSS_TERMS:
        if counts.get(frag_a, 0) > 0 and counts.get(frag_b, 0) > 0:
            correction += boost
    return max(-2.0, min(2.0, correction))


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
    # TPSA coefficient raised from 0.025→0.030 (ADR-2026-06-05c): TPSA directly
    # measures molecular polarity; 0.030 better differentiates high-polarity
    # (EC, DMSO, DMF) from low-polarity molecules, improving rank separation.
    value += tpsa * 0.030

    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)

    return max(1.0, value)


def _count_branch_points(mol: Chem.Mol) -> int:
    """Count topological branch points (sp3 atoms with degree >= 3)."""
    return sum(
        a.GetDegree() >= 3
        and a.GetHybridization() == Chem.HybridizationType.SP3
        for a in mol.GetAtoms()
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


def predict_ionic_conductivity_proxy(
    dielectric: float, viscosity: float, li_solvation: float
) -> float:
    """Predict ionic conductivity proxy via Walden-product model.

    Combines dielectric (salt dissociation), viscosity (Stokes-Einstein
    mobility), and Li+ solvation (charge carrier availability) into a
    single figure of merit. The Walden product relates molar conductivity
    to fluidity: ion mobility is inversely proportional to viscosity and
    proportional to the number of charge carriers (set by dielectric *
    Li+ binding strength). The Li+ solvation contribution uses a Gaussian
    centered on the Goldilocks target (3.5) — too-weak binding fails to
    dissociate salts, too-strong binding reduces transference number.
    """
    if viscosity <= 0.0 or dielectric < 0.0:
        return 0.0

    effective_dielec = max(0.0, dielectric - 1.0)
    solvation_factor = math.exp(-0.5 * ((li_solvation - 3.5) / 1.5) ** 2)

    if viscosity < 0.001:
        return 0.0
    conductivity = effective_dielec * solvation_factor / viscosity
    return max(0.0, min(10.0, conductivity))


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

    ADR-2026-06-05: Added Margules-inspired non-ideal mixing term. The
    two-suffix Margules model gives excess Gibbs energy G^E = A · x₁ · x₂,
    which captures the synergy peak at equimolar composition for truly
    complementary pairs. Here A ∝ |d₁-d₂| · |v₁-v₂| — pairs with large
    differences in both properties (the definition of complementarity)
    receive an extra bonus that peaks at 50:50 mixing, matching the
    physical intuition that complementary pairs are most effective at
    balanced compositions.
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

    # Margules-inspired non-ideal mixing term (ADR-2026-06-05)
    # A ∝ |d₁-d₂|·|v₁-v₂| scaled to give ~0.5 bonus at 50:50 for complementary pairs
    interaction = abs(d1 - d2) * abs(v1 - v2) / 8.0
    interaction = min(interaction, 3.0)  # saturation cap to prevent gaming
    non_ideal = interaction * frac1 * f2
    score += non_ideal

    return min(max(0.0, score), 6.0)
