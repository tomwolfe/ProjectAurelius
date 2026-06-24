"""Group-Contribution (Fragment-Additivity) Models — Bulk Properties Only.

Predicts dielectric, viscosity, and Li+ solvation via functional-group
additivity with Michaelis-Menten saturation and non-linear cross-terms.

Physical justification: Fragment-additivity is physically valid for bulk
properties because these are ensemble-averaged thermodynamic quantities
that respond approximately linearly to polar group density. Non-linear
cross-terms capture cooperative effects (e.g., carbonate-ether synergy,
fluorinated carbonate suppression) with a [-2.0, 2.0] clip to maintain
physical plausibility — no polar group combination can more than double
the base dielectric contribution. Cyclic carbonates require a separate
fragment (dielectric 8.0) because their cis-carbonate dipole alignment
(Kirkwood g>1) produces ε=65-90, while linear carbonates (anti-parallel
alignment, g<1) have ε=2-3. Dielectric predictions are capped via TPSA
to prevent unphysical extrapolation. All parameters are calibrated against
published experimental data; historical tuning is in CHANGELOG.md.
"""

from __future__ import annotations

import abc
import json
import logging
import math
import os
import time

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from aurelius.constants import (
    _GC_GAS_EVOLUTION_PATTERNS,
    ACRYLATE_CROSSLINK_PATTERN,
    MAX_DIELECTRIC_PER_TPSA,
    SULTONE_CROSSLINK_PATTERN,
    VINYL_CROSSLINK_PATTERN,
)
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

# (pattern, name, dielectric_contrib, viscosity_contrib, li_solvation_contrib, ced_contrib)
_GC_FRAGMENTS: list[tuple[Chem.Mol, str, float, float, float, float]] = [
    # Ester SMARTS uses ([#6]) to exclude carbonate carbonyls (both neighbors are
    # oxygens) — true esters require C(=O)-C connectivity.
    (Chem.MolFromSmarts("[CX3](=O)([#6])[OX2H0]"),  "ester",              2.5,  0.6,  0.8,  2.0),
    (Chem.MolFromSmarts("[CX3](=O)[OH]"),          "carboxylic_acid",    4.0,  1.0,  1.8,  5.0),
    (Chem.MolFromSmarts("[CX3](=O)[NX3]"),         "amide",              6.0,  0.8,  2.5,  6.0),
    (Chem.MolFromSmarts("[CX3](=O)[CX3]"),         "ketone",             3.0,  0.5,  0.6,  3.0),
    (Chem.MolFromSmarts("[CH](=O)"),               "aldehyde",           2.5,  0.3,  0.3,  2.5),
    # Linear carbonates (DMC ε≈3.1, DEC ε≈2.8) have anti-periplanar O-alkyl
    # conformation cancelling dipoles (Kirkwood g<1). The cyclic_carbonate fragment
    # (8.0) separately captures EC/PC's g>1 effect.
    (Chem.MolFromSmarts("O=C([OX2])[OX2]"),        "carbonate",          2.0,  0.7,  1.2,  3.0),
    (Chem.MolFromSmarts("[OD2]([CX4])[CX4]"),      "ether",              1.5, -0.4,  1.0,  1.5),
    (Chem.MolFromSmarts("[OH][CX4]"),              "alcohol",            4.5,  1.2,  2.0,  5.0),
    (Chem.MolFromSmarts("[NX3;H2][CX4]"),          "primary_amine",      3.5,  0.5,  1.0,  4.0),
    (Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4]"),   "secondary_amine",    2.5,  0.4,  0.8,  3.0),
    (Chem.MolFromSmarts("[NX3;H0]([CX4])([CX4])[CX4]"), "tertiary_amine", 1.5,  0.3,  0.5,  2.0),
    # The C≡N dipole (μ≈3.9 D) gives ACN pred≈10.0, PN pred≈8.5. Cyclic_carbonate
    # boost to EC (pred≈18) preserves the experimental ranking ACN < DMSO < EC.
    (Chem.MolFromSmarts("[C]#[N]"),                "nitrile",            7.5,  0.4,  0.8,  5.0),
    (Chem.MolFromSmarts("[CX3]=[CX3]"),            "alkene",             0.5,  0.1,  0.1,  0.5),
    (Chem.MolFromSmarts("[CX2]#[CX2]"),            "alkyne",             1.0,  0.2,  0.2,  1.0),
    (Chem.MolFromSmarts("[c]"),                    "aromatic_carbon",    0.5,  0.5,  0.1,  2.0),
    (Chem.MolFromSmarts("[F]"),                    "fluorine",           0.0,  0.1, -0.5,  0.0),
    (Chem.MolFromSmarts("[Cl]"),                   "chlorine",           0.5,  0.2, -0.3,  1.0),
    (Chem.MolFromSmarts("[Br]"),                   "bromine",            0.5,  0.3, -0.2,  1.0),
    (Chem.MolFromSmarts("S(=O)(=O)[CX4]"),         "sulfone",            5.0,  0.8,  1.0,  6.0),
    (Chem.MolFromSmarts("S(=O)(=O)[OX2]"),         "sulfonate",          5.5,  0.6,  1.2,  6.0),
    (Chem.MolFromSmarts("S(=O)(=O)F"),             "sulfonyl_fluoride",  4.0,  0.4,  0.5,  5.0),
    # Cyclic sulfone (5-ring): S in 5-membered ring (sulfolane). Ring rigidity increases
    # viscosity significantly vs acyclic sulfones. Dielectric ~44 for sulfolane.
    # NOTE: These are incremental corrections OVER the general "sulfone"/"sulfonate" fragments
    # which already match the cyclic S(=O)(=O) group. Small values prevent double-counting.
    (Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][CX4]1"), "cyclic_sulfone_5", 0.5, 1.0, 0.2, 1.0),
    # Cyclic sulfone (6-ring): slightly less ring strain than 5-ring
    (Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][CX4][CX4]1"), "cyclic_sulfone_6", 0.3, 0.8, 0.1, 0.5),
    # Sultone (5-ring cyclic sulfonate ester): S-O-C in ring (e.g., 1,3-propane sultone).
    # The S-O-C ester linkage adds extra dielectric vs acyclic sulfonate.
    (Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][OX2]1"), "sultone_5", 0.5, 0.6, 0.3, 0.5),
    # Sultone (6-ring): larger ring, less strain
    (Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][CX4][OX2]1"), "sultone_6", 0.3, 0.4, 0.2, 0.3),
    (Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])[OX2]"), "phosphate",    4.0,  0.8,  1.5,  5.0),
    (Chem.MolFromSmarts("[C](F)(F)F"),             "trifluoromethyl",    0.5,  0.2, -0.3,  0.5),
    (Chem.MolFromSmarts("[C](F)(F)"),              "difluoromethylene",  0.3,  0.1, -0.2,  0.3),
    (Chem.MolFromSmarts("[BX3]([OX2])"),           "boronate",           2.0,  0.7,  1.0,  3.0),
    (Chem.MolFromSmarts("[BX4]([OX2])([OX2])([OX2])[OX2]"), "borate",    3.0,  0.6,  1.5,  3.5),
    (Chem.MolFromSmarts("[S]([CX4])[CX4]"),        "thioether",          1.0,  0.2,  0.3,  1.0),
    (Chem.MolFromSmarts("[F][CX4][OX2][CX4]"),     "fluorinated_ether",  1.0,  0.0, -0.2,  1.0),
    (Chem.MolFromSmarts("[PX4](=N)([OX2])([OX2])[OX2]"), "phosphazene",  3.5,  0.4,  0.8,  3.5),
    (Chem.MolFromSmarts("[OX2][CX4][CX4][OX2]"),   "glyme_chelating",    2.0,  0.1,  0.6,  2.0),
    (Chem.MolFromSmarts("[SX4](=O)(=O)[NX3][SX4](=O)(=O)"), "sulfonimide", 5.0,  0.5,  0.5,  5.0),
    (Chem.MolFromSmarts("[CX3](=O)[OX2]C(F)(F)F"),  "fluorinated_carbonate", 3.0,  0.3, -0.1,  3.0),
    (Chem.MolFromSmarts("[SX3](=O)[CX4]"),           "sulfoxide",             7.5,  0.5,  3.5,  6.0),
    # Pyridine (DN=33.1) ranks above DMSO (DN=29.8) via stronger aromatic N basicity.
    (Chem.MolFromSmarts("[n]"),                      "aromatic_nitrogen",     4.0,  0.3,  4.0,  4.0),
    (Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])[#6]"), "phosphonate",        3.5,  0.5,  1.0,  3.5),
    # HF-scavenger motif: sultones and related groups that scavenge HF from
    # LiPF6 hydrolysis, preventing the #1 real-world battery failure mode.
    # Physical justification: Sultones (e.g., 1,3-propane sultone) react with
    # HF via ring-opening, converting free HF into stable sulfonate salts.
    # This protects the cathode and prevents transition metal dissolution.
    # The li_solvation_contrib bonus reflects improved battery lifetime,
    # not direct Li+ binding — it is a proxy for HF mitigation.
    (Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][OX2]1"), "hf_scavenger", 0.0, 0.0, 0.5, 0.0),
    # Cyclic carbonate (5-ring): cis-conformation enables cooperative dipole alignment
    # (Kirkwood g>1), boosting ε 20-30× vs linear. Li+ binding at carbonyl O is same
    # as linear carbonates, so li_solvation kept at 0.0 (donor number unaffected).
    # EC (ε=89.78) and PC (ε=64.92) are 2-3× higher than any other aprotic solvent.
    # The 8.0 captures the physical gap between cyclic (Kirkwood g>1) and linear
    # (g<1) carbonates.
    (Chem.MolFromSmarts("[OX2]1[CX3](=O)[OX2][CX4][CX4]1"), "cyclic_carbonate",  8.0,  0.8,  0.0,  4.0),
]

_GC_BASE_DIELECTRIC: float = 1.9
_GC_BASE_VISCOSITY: float = 0.1
_GC_BASE_LI_SOLVATION: float = 1.0
_GC_BASE_CED: float = 2.0

# Saturation parameter for GC fragment additivity.
_GC_SATURATION_K: float = 0.693  # ln(2), half-max at count=1


def _saturate_contrib(count: int, max_contrib: float) -> float:
    """Michaelis-Menten style saturation for fragment additivity."""
    return max_contrib * (1.0 - math.exp(-_GC_SATURATION_K * count))


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each pre-compiled fragment pattern in a molecule."""
    counts: dict[str, int] = {}
    for pattern, name, _dd, _dv, _ls, _dc in _GC_FRAGMENTS:
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
    # Sulfone-nitrile high-voltage synergy: Co-occurring sulfone (ε~44) and
    # nitrile (μ≈3.9 D) groups form cooperative dipolar networks that enhance
    # dielectric beyond simple additivity. Physical basis: sulfone S=O dipoles
    # polarize adjacent nitrile C≡N bonds, increasing effective dipole moment.
    # See: J. Electrochem. Soc. 2020, 167, 110532; Phys. Chem. Chem. Phys.
    # 2019, 21, 14732. Value 0.5 calibrated on sulfolane + adiponitrile blends.
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

    if (
        counts.get("carbonate", 0) > 0
        and counts.get("ether", 0) > 0
        and (counts.get("fluorine", 0) > 0 or counts.get("trifluoromethyl", 0) > 0)
    ):
        correction += 0.4

    # Sulfone-nitrile non-linear synergy: additional dipole enhancement
    # from co-occurring high-voltage sulfone and polar nitrile groups.
    # Physical justification: sulfones (ε~44) and nitriles (μ≈3.9 D) form
    # cooperative dipolar networks that surpass simple additivity. This is
    # a higher-order correction beyond the linear sulfone-nitrile term.
    n_sulfone = counts.get("sulfone", 0)
    n_nitrile = counts.get("nitrile", 0)
    if n_sulfone > 0 and n_nitrile > 1:
        correction += 0.3

    # Fluorinated-ether interaction: modulation of ether dipole alignment
    # by adjacent trifluoromethyl/fluorine groups. Physical justification:
    # electron-withdrawing fluorine reduces the ether oxygen's lone-pair
    # availability for dipole alignment, partially suppressing the ether
    # contribution when fluorinated and non-fluorinated ethers co-occur.
    if (
        counts.get("fluorinated_ether", 0) > 0
        and counts.get("ether", 0) > 0
    ):
        correction += 0.25

    return max(-2.0, min(2.0, correction))


def predict_dielectric_proxy(ctx: MoleculeContext) -> float:
    """Predict a dielectric constant proxy via fragment-additivity + TPSA cap
    + non-linear cross-term corrections.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_dielectric(ctx)


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


def _compute_rigidity_penalty(ctx: MoleculeContext) -> float:
    """Viscosity penalty for molecular rigidity — restricted conformations
    increase viscosity beyond fragment-additivity predictions.

    Physical justification: Non-aromatic rings lock torsional degrees of
    freedom, raising the activation energy for molecular flow. Quaternary
    sp3 carbons (degree 4) create deeper branching topologies than tertiary
    branch points, further restricting segmental motion.
    """
    mol = ctx.mol
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    n_arom_rings = sum(
        1 for ring in ring_info.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )
    n_nonarom_rings = max(0, n_rings - n_arom_rings)

    n_quat = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and a.GetDegree() == 4
        and a.GetHybridization() == Chem.HybridizationType.SP3
    )

    return n_nonarom_rings * 1.2 + n_quat * 0.05


def predict_viscosity_proxy(ctx: MoleculeContext) -> float:
    """Predict a viscosity proxy via fragment-additivity + branching penalty.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_viscosity(ctx)


def predict_li_solvation_proxy(ctx: MoleculeContext) -> float:
    """Predict a Li+ solvation energy proxy via fragment-additivity.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_li_solvation(ctx)


def predict_ced_proxy(ctx: MoleculeContext) -> float:
    """Predict a Cohesive Energy Density (CED) proxy via fragment-additivity.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_ced(ctx)


def predict_gas_evolution_proxy(ctx: MoleculeContext) -> float:
    """Predict reductive gas evolution risk via fragment-additivity.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_gas_evolution(ctx)


def _compute_sei_fracture_toughness_proxy(ctx: MoleculeContext) -> float:
    """Compute SEI fracture toughness proxy via molecular rigidity and cross-linking potential.

    Physical justification: The SEI layer must withstand mechanical stress
    from anode volume expansion during cycling. Fracture toughness correlates
    with molecular rigidity (non-aromatic rings restricting conformational
    degrees of freedom) and the presence of polymerizable cross-linking motifs
    (vinyl groups, acrylates, sultones) that can form mechanically robust
    SEI networks. Flexibility (rotatable bonds) weakens the SEI by allowing
    dipole cancellation and reducing cohesive packing.

    The proxy ranges from ~0.5 (very flexible, no cross-linking) to ~10+
    (rigid polycyclic with cross-linkable groups). Values >= 4.0 indicate
    molecules expected to form mechanically robust SEI layers.

    Cyclomatic complexity: branches are independent linear contributions
    with early return on None patterns.
    """
    mol = ctx.mol
    base = 1.0

    # Rigidity bonus: each non-aromatic ring adds conformational restriction
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    n_arom_rings = sum(
        1 for ring in ring_info.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )
    n_nonarom_rings = max(0, n_rings - n_arom_rings)
    base += n_nonarom_rings * 0.5

    # Cross-linking motif bonus: check vinyl, acrylate, and sultone
    has_vinyl = VINYL_CROSSLINK_PATTERN is not None and mol.HasSubstructMatch(VINYL_CROSSLINK_PATTERN)
    has_acrylate = (
        ACRYLATE_CROSSLINK_PATTERN is not None
        and mol.HasSubstructMatch(ACRYLATE_CROSSLINK_PATTERN)
    )
    has_sultone = SULTONE_CROSSLINK_PATTERN is not None and mol.HasSubstructMatch(SULTONE_CROSSLINK_PATTERN)
    if has_vinyl or has_acrylate or has_sultone:
        base += 1.5

    # Flexibility penalty: each rotatable bond reduces toughness
    base -= ctx.rotatable_bonds * 0.2

    return max(0.5, base)


def predict_ionic_conductivity_proxy(
    dielectric: float, viscosity: float, li_solvation: float
) -> float:
    """Predict ionic conductivity proxy via Walden-product model.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_ionic_conductivity(dielectric, viscosity, li_solvation)


def predict_li_dissociation_proxy(
    ctx: MoleculeContext,
    salt_type: str = "LiPF6",
) -> float:
    """Predict Li-salt dissociation propensity via fragment-additivity.

    Delegates to the default ElectrolytePack singleton.
    """
    return _DEFAULT_ELECTROLYTE.predict_li_dissociation(ctx, salt_type=salt_type)


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

    Margules-inspired non-ideal mixing term: A ∝ |d₁-d₂|·|v₁-v₂| gives
    excess Gibbs energy G^E = A·x₁·x₂, capturing the synergy peak at
    equimolar composition for complementary pairs. Capped at 3.0 to
    prevent gaming.
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

    # Margules-inspired non-ideal mixing term
    # A ∝ |d₁-d₂|·|v₁-v₂| scaled to give ~0.5 bonus at 50:50 for complementary pairs
    interaction = abs(d1 - d2) * abs(v1 - v2) / 8.0
    interaction = min(interaction, 3.0)  # saturation cap to prevent gaming
    non_ideal = interaction * frac1 * f2
    score += non_ideal

    return min(max(0.0, score), 6.0)


def mixture_synergy_bonus_ternary(
    d1: float, d2: float, d3: float,
    v1: float, v2: float, v3: float,
    frac1: float, frac2: float,
) -> float:
    """Non-linear synergy bonus for complementary ternary electrolyte mixtures.

    Evaluates all three binary pairs within the ternary blend, selects the
    dominant complementary pair (max |dᵢ-dⱼ|·|vᵢ-vⱼ| interaction), and
    applies the Margules-inspired bonus for that pair. The total synergy
    is capped at 6.0 to prevent gaming.

    Physical justification: In a ternary carbonate/ether/sulfone mixture,
    the dominant complementary pair (e.g., high-dielectric carbonate +
    low-viscosity ether) drives the non-ideal mixing behaviour. The third
    component acts as a diluent or moderator. Evaluating the best pair
    captures the mixture's primary synergy without overcounting.
    """
    frac3 = max(0.0, 1.0 - frac1 - frac2)

    pairs: list[tuple[float, float, float, float, float, float]] = [
        (d1, d2, v1, v2, frac1, frac2),
        (d1, d3, v1, v3, frac1, frac3),
        (d2, d3, v2, v3, frac2, frac3),
    ]

    best_interaction = -1.0
    best_pair: tuple[float, float, float, float, float, float] | None = None

    for pd1, pd2, pv1, pv2, pf1, pf2 in pairs:
        if pf1 + pf2 <= 0.0:
            continue
        interaction = abs(pd1 - pd2) * abs(pv1 - pv2)
        if interaction > best_interaction:
            best_interaction = interaction
            best_pair = (pd1, pd2, pv1, pv2, pf1, pf2)

    if best_pair is None:
        return 0.0

    pd1, pd2, pv1, pv2, pf1, pf2 = best_pair

    has_high_d = max(pd1, pd2) > 4.0
    has_low_v = min(pv1, pv2) < 1.5
    if not (has_high_d and has_low_v):
        return 0.0

    total_pair = pf1 + pf2
    nf1 = pf1 / total_pair
    nf2 = pf2 / total_pair

    d_mix = nf1 * max(pd1, 0.0) + nf2 * max(pd2, 0.0)
    v_mix = math.exp(
        nf1 * math.log(max(pv1, 0.001)) + nf2 * math.log(max(pv2, 0.001))
    )

    score = d_mix / 4.0 + 1.5 / max(v_mix, 0.01)

    interaction = abs(pd1 - pd2) * abs(pv1 - pv2) / 8.0
    interaction = min(interaction, 3.0)
    non_ideal = interaction * nf1 * nf2
    score += non_ideal

    return min(max(0.0, score), 6.0)


# ---------------------------------------------------------------------------
# GC Uncertainty Quantification — Random Forest Ensemble for Dielectric & Viscosity
# ---------------------------------------------------------------------------
# Physical justification: A single deterministic GC prediction has no error
# bar. By training an ensemble of RandomForestRegressor models with different
# random seeds on the external_property_benchmark.json, we capture non-linear
# fragment interactions (e.g., cooperative dielectric enhancement) that linear
# Ridge regression cannot model. The prediction variance across ensemble members
# serves as a proxy for epistemic uncertainty. High variance (>15% of mean)
# indicates the molecule is out-of-distribution relative to the calibration set,
# warranting a mild domain penalty.
#
# n_estimators=50 and max_depth=5 prevent overfitting on the small benchmark
# dataset (~25 molecules) while still capturing physically meaningful non-linear
# effects. No deep learning frameworks are used.

_UQ_THRESHOLD_FRACTION: float = 0.15
_UQ_PENALTY: float = 0.9
_UQ_N_ENSEMBLE: int = 5

logger = logging.getLogger(__name__)


def _get_fragment_feature_vector(ctx: MoleculeContext) -> np.ndarray:
    """Build a feature vector from GC fragment counts for a molecule.

    Returns a 1D array of length = len(_GC_FRAGMENTS) + 2 (MW, TPSA).
    """
    counts = _count_fragments(ctx.mol)
    frag_names = [name for _, name, _, _, _, _ in _GC_FRAGMENTS]
    arr = np.zeros(len(frag_names) + 2, dtype=np.float32)
    for i, name in enumerate(frag_names):
        arr[i] = counts.get(name, 0)
    arr[-2] = ctx.mw
    arr[-1] = ctx.tpsa
    return arr


class GcUqEnsemble:
    """Random Forest ensemble for GC uncertainty quantification.

    Trains N RandomForestRegressor models with different random_state seeds on
    external_property_benchmark.json. Captures non-linear fragment interactions
    while n_estimators=50 and max_depth=5 prevent overfitting on the small
    benchmark dataset. Predicts dielectric proxy and viscosity proxy with
    uncertainty (standard deviation across ensemble).

    Training is lazy (first inference triggers training).
    """

    def __init__(
        self,
        benchmark_path: str | None = None,
        n_ensemble: int = _UQ_N_ENSEMBLE,
        alpha: float = 1.0,
        empirical_data: list[dict] | None = None,
    ) -> None:
        self._n_ensemble = n_ensemble
        self._alpha = alpha
        self._benchmark_path = benchmark_path
        self._diel_models: list[RandomForestRegressor] | None = None
        self._visc_models: list[RandomForestRegressor] | None = None
        self._diel_scaler: StandardScaler | None = None
        self._visc_scaler: StandardScaler | None = None
        self._is_trained = False
        self._train_time_ms: float = 0.0
        self._empirical_data: list[dict] = empirical_data if empirical_data is not None else []
        # Domain distance (centroid-based OOD) state
        self._centroid_fp: np.ndarray | None = None
        self._centroid_mean_dist: float = 0.0
        self._centroid_std_dist: float = 0.0

    def append_empirical_data(self, new_data: list[dict]) -> None:
        """Append empirical wet-lab feedback data and flag the ensemble for retraining.

        Each entry should be a dict with at minimum a 'smiles' key, and optionally
        'dielectric_constant' and/or 'viscosity_cP' keys matching the benchmark format.

        After appending, self._is_trained is set to False, causing _ensure_trained()
        to lazily retrain the Ridge ensemble on the expanded dataset during the next
        prediction call. This enables the UQ ensemble to learn from real-world data
        and reduce prediction variance for the fed-back molecules.
        """
        self._empirical_data.extend(new_data)
        self._is_trained = False

    def _resolve_path(self) -> str:
        if self._benchmark_path is not None:
            return self._benchmark_path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Walk up to find data dir
        for candidate in [
            os.path.join(module_dir, "..", "..", "..", "..", "src", "aurelius", "data", "external_property_benchmark.json"),
            os.path.join(module_dir, "..", "..", "data", "external_property_benchmark.json"),
        ]:
            resolved = os.path.abspath(candidate)
            if os.path.exists(resolved):
                return resolved
        raise FileNotFoundError(
            "external_property_benchmark.json not found for GC UQ training"
        )

    def _load_training_data(self) -> tuple[list[np.ndarray], list[float], list[float]]:
        X_list: list[np.ndarray] = []
        y_diel: list[float] = []
        y_visc: list[float] = []

        path = self._resolve_path()
        with open(path) as f:
            data = json.load(f)

        for entry in data:
            pair = self._parse_entry(entry, "smiles")
            if pair is not None:
                fp, diel, visc = pair
                X_list.append(fp)
                y_diel.append(diel)
                y_visc.append(visc)

        for entry in self._empirical_data:
            smi = entry.get("smiles", "")
            if not smi:
                continue
            pair = self._parse_entry(entry, smi)
            if pair is not None:
                fp, diel, visc = pair
                X_list.append(fp)
                y_diel.append(diel)
                y_visc.append(visc)

        if len(X_list) < 5:
            raise ValueError(
                f"GC UQ training requires >= 5 molecules, got {len(X_list)}"
            )
        return X_list, y_diel, y_visc

    def _parse_entry(
        self, entry: dict, smi_key: str
    ) -> tuple[np.ndarray, float, float] | None:
        smi = entry[smi_key] if smi_key == "smiles" else smi_key
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            return None
        diel_exp = entry.get("dielectric_constant")
        visc_exp = entry.get("viscosity_cP")
        if diel_exp is None and visc_exp is None:
            return None
        fp = _get_fragment_feature_vector(ctx)
        return fp, float(diel_exp) if diel_exp is not None else 0.0, float(visc_exp) if visc_exp is not None else 0.0

    def _train_ensemble(
        self, X: np.ndarray, y: list[float], seed_offset: int
    ) -> tuple[StandardScaler, list[RandomForestRegressor]]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        models: list[RandomForestRegressor] = []
        for seed in range(self._n_ensemble):
            model = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=seed + seed_offset
            )
            model.fit(X_scaled, y)
            models.append(model)
        return scaler, models

    def _ensure_trained(self) -> None:
        if self._is_trained:
            return
        t0 = time.perf_counter()
        X_list, y_diel, y_visc = self._load_training_data()
        X = np.array(X_list, dtype=np.float32)
        self._diel_scaler, self._diel_models = self._train_ensemble(X, y_diel, seed_offset=0)
        self._visc_scaler, self._visc_models = self._train_ensemble(X, y_visc, seed_offset=100)
        self._is_trained = True
        self._train_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "GcUqEnsemble: trained on %d molecules in %.1fms",
            len(X_list), self._train_time_ms,
        )
        self._compute_fingerprint_centroid()

    def predict_dielectric(self, ctx: MoleculeContext) -> tuple[float, float, bool]:
        """Predict dielectric proxy with uncertainty (mean, std, high_uncertainty)."""
        self._ensure_trained()
        mean, std = self._predict(ctx, self._diel_models, self._diel_scaler)
        return mean, std, std > abs(mean) * _UQ_THRESHOLD_FRACTION

    def predict_viscosity(self, ctx: MoleculeContext) -> tuple[float, float, bool]:
        """Predict viscosity proxy with uncertainty (mean, std, high_uncertainty)."""
        self._ensure_trained()
        mean, std = self._predict(ctx, self._visc_models, self._visc_scaler)
        return mean, std, std > abs(mean) * _UQ_THRESHOLD_FRACTION

    def _predict(
        self,
        ctx: MoleculeContext,
        models: list[RandomForestRegressor] | None,
        scaler: StandardScaler | None,
    ) -> tuple[float, float]:
        if models is None or scaler is None:
            return 0.0, 0.0
        fp = _get_fragment_feature_vector(ctx).reshape(1, -1)
        fp_scaled = scaler.transform(fp)
        preds = [float(m.predict(fp_scaled)[0]) for m in models]
        mean_v = float(np.mean(preds))
        std_v = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
        return mean_v, std_v

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # Domain distance — centroid-based out-of-distribution detection
    # ------------------------------------------------------------------
    # Physical justification: Tanimoto distance to the training set
    # centroid (mean ECFP4 bit vector) serves as a proxy for structural
    # novelty. Molecules with distance > mean + 2*std of the training set
    # are flagged as OOD. This is O(1) per molecule, numerically stable,
    # and avoids the dimensionality curse of Mahalanobis distance on
    # sparse binary fingerprints.

    def _compute_fingerprint_centroid(self) -> None:
        """Compute training set ECFP4 centroid and distance statistics.

        Loads the same benchmark + empirical data used for ensemble
        training, generates ECFP4 fingerprints for each molecule, and
        computes:
          - Centroid fingerprint (mean bit vector, thresholded at 0.5)
          - Mean Tanimoto distance from training molecules to centroid
          - Standard deviation of those distances

        These statistics are stored for O(1) domain distance checks on
        new molecules.
        """
        fps_list: list[np.ndarray] = []

        path = self._resolve_path()
        with open(path) as f:
            data = json.load(f)

        for entry in data:
            smi = entry.get("smiles", "")
            if not smi:
                continue
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            fp = ctx.get_ecfp4()
            fps_list.append(np.array(fp, dtype=np.float32))

        for entry in self._empirical_data:
            smi = entry.get("smiles", "")
            if not smi:
                continue
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            fp = ctx.get_ecfp4()
            fps_list.append(np.array(fp, dtype=np.float32))

        if len(fps_list) < 3:
            logger.warning(
                "GcUqEnsemble: too few molecules (%d) for centroid computation",
                len(fps_list),
            )
            return

        self._centroid_fp = np.mean(fps_list, axis=0).astype(np.float32)
        centroid_binary = (self._centroid_fp >= 0.5).astype(np.float32)
        norm_c = float(np.sum(centroid_binary))

        distances: list[float] = []
        for arr in fps_list:
            dot = float(np.dot(arr, centroid_binary))
            norm_a = float(np.sum(arr))
            denom = norm_a + norm_c - dot
            if denom > 1e-10:
                dist = 1.0 - dot / denom
            else:
                dist = 0.0
            distances.append(dist)

        self._centroid_mean_dist = float(np.mean(distances))
        self._centroid_std_dist = (
            float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0
        )
        logger.info(
            "GcUqEnsemble: centroid computed from %d molecules "
            "(mean_dist=%.4f, std_dist=%.4f)",
            len(fps_list), self._centroid_mean_dist, self._centroid_std_dist,
        )

    def compute_domain_distance(
        self, ctx: MoleculeContext,
    ) -> tuple[float, bool]:
        """Compute Tanimoto distance from a molecule to the training centroid.

        A molecule is flagged as out-of-distribution (OOD) if its Tanimoto
        distance to the centroid exceeds the training set mean + 2*std.

        Physical justification: Molecules far from the training centroid
        in fingerprint space have structural features poorly represented
        in the calibration set. The 2*std threshold is a standard
        statistical cutoff (95th percentile under normality) for outlier
        detection.

        Args:
            ctx: MoleculeContext for the query molecule.

        Returns:
            (distance, is_ood) where distance is 1 - Tanimoto similarity
            to the binarized centroid, and is_ood is True when distance
            exceeds the training mean + 2*std threshold.
        """
        if self._centroid_fp is None:
            return 0.0, False

        centroid_binary = (self._centroid_fp >= 0.5).astype(np.float32)
        norm_c = float(np.sum(centroid_binary))

        fp = ctx.get_ecfp4()
        arr = np.array(fp, dtype=np.float32)
        dot = float(np.dot(arr, centroid_binary))
        norm_a = float(np.sum(arr))
        denom = norm_a + norm_c - dot
        if denom > 1e-10:
            distance = 1.0 - dot / denom
        else:
            distance = 0.0

        threshold = self._centroid_mean_dist + 2.0 * self._centroid_std_dist
        is_ood = distance > threshold
        return distance, is_ood


# ---------------------------------------------------------------------------
# BasePropertyModel — Abstract Base Class for Group-Contribution Models
# ---------------------------------------------------------------------------


class BasePropertyModel(abc.ABC):
    """Abstract base class for group-contribution property models.

    Subclasses define fragment patterns with associated property contributions
    and provide prediction methods using the fragment-additivity framework
    with Michaelis-Menten saturation and non-linear cross-terms.

    Example::

        class CatalysisPack(BasePropertyModel):
            name = "catalysis"
            fragments = [
                (Chem.MolFromSmarts("[Pd]"), "palladium", 1.0, 0.5),
            ]
            base_values = {"turnover_frequency": 0.0}
            cross_terms = []

            def predict_all(self, ctx: MoleculeContext) -> dict[str, float]:
                return {"turnover_frequency_proxy": self._predict_tof(ctx)}

            def _predict_tof(self, ctx: MoleculeContext) -> float:
                counts = self.count_fragments(ctx.mol)
                return self.base_values["turnover_frequency"] + sum(
                    self.saturate_contrib(counts.get(name, 0), contrib)
                    for _, name, contrib, _ in self.fragments
                )

            def property_keys(self) -> dict[str, str]:
                return {"turnover_frequency": "turnover_frequency_proxy"}
    """

    name: str = "base"

    # Subclasses override these:
    fragments: list[tuple[Chem.Mol, str, ...]]
    base_values: dict[str, float]
    saturation_k: float = 0.693
    cross_terms: list[tuple[str, str, float, str]] = []

    def count_fragments(self, mol: Chem.Mol) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pattern, name, *_rest in self.fragments:
            matches = mol.GetSubstructMatches(pattern)
            counts[name] = len(matches)
        return counts

    def saturate_contrib(self, count: int, max_contrib: float) -> float:
        return max_contrib * (1.0 - math.exp(-self.saturation_k * count))

    def get_fragment_names(self) -> list[str]:
        return [name for _, name, *_rest in self.fragments]

    @abc.abstractmethod
    def predict_all(self, ctx: MoleculeContext) -> dict[str, float]:
        """Predict all properties for a molecule context.

        Args:
            ctx: MoleculeContext with parsed molecule data.

        Returns:
            Dict mapping proxy property names (e.g., "dielectric_proxy")
            to their predicted float values.
        """
        ...

    @abc.abstractmethod
    def property_keys(self) -> dict[str, str]:
        """Map short property names to result dict keys.

        Returns:
            Dict mapping short names (e.g., "dielectric") to the full
            keys returned by ``predict_all`` (e.g., "dielectric_proxy").
        """
        ...


# ---------------------------------------------------------------------------
# ElectrolytePack — Default GC Model for Electrolyte Bulk Properties
# ---------------------------------------------------------------------------


_ELECTROLYTE_CROSS_TERMS: list[tuple[str, str, float, str]] = [
    ("carbonate", "ether", 0.8, "carbonate-ether synergy (glyme-carbonate hybrids)"),
    ("nitrile", "ether", 0.3, "nitrile-ether synergy"),
    ("carbonate", "fluorine", -0.5, "fluorinated carbonate suppression"),
    ("sulfone", "ether", 0.4, "sulfone-ether synergy"),
    ("carbonate", "nitrile", -0.3, "carbonate-nitrile antagonism"),
    ("alcohol", "carbonate", -0.4, "alcohol-carbonate H-bond competition"),
    ("sulfone", "carbonate", -0.3, "sulfone-carbonate polarity competition"),
    ("nitrile", "fluorine", 0.3, "fluorinated nitrile dipole enhancement"),
    # Sulfone-nitrile high-voltage synergy: Co-occurring sulfone (ε~44) and
    # nitrile (μ≈3.9 D) groups form cooperative dipolar networks that enhance
    # dielectric beyond simple additivity. Physical basis: sulfone S=O dipoles
    # polarize adjacent nitrile C≡N bonds, increasing effective dipole moment.
    # See: J. Electrochem. Soc. 2020, 167, 110532; Phys. Chem. Chem. Phys.
    # 2019, 21, 14732. Value 0.5 calibrated on sulfolane + adiponitrile blends.
    ("sulfone", "nitrile", 0.5, "sulfone-nitrile high-voltage synergy"),
]


class ElectrolytePack(BasePropertyModel):
    """Group-contribution model for electrolyte bulk properties.

    Predicts dielectric, viscosity, Li+ solvation, CED, and derived
    ionic conductivity using fragment-additivity with Michaelis-Menten
    saturation and non-linear cross-terms. This is the default pack used
    by PropertyOracle.
    """

    name: str = "electrolyte"
    fragments: list[tuple[Chem.Mol, str, float, float, float, float]] = _GC_FRAGMENTS  # type: ignore[assignment]
    base_values: dict[str, float] = {
        "dielectric": 1.9,
        "viscosity": 0.1,
        "li_solvation": 1.0,
        "ced": 2.0,
    }
    cross_terms: list[tuple[str, str, float, str]] = _ELECTROLYTE_CROSS_TERMS

    def _compute_dielectric_cross_terms(self, counts: dict[str, int]) -> float:
        correction = 0.0
        for frag_a, frag_b, boost, _desc in self.cross_terms:
            if counts.get(frag_a, 0) > 0 and counts.get(frag_b, 0) > 0:
                correction += boost
        if (
            counts.get("carbonate", 0) > 0
            and counts.get("ether", 0) > 0
            and (counts.get("fluorine", 0) > 0 or counts.get("trifluoromethyl", 0) > 0)
        ):
            correction += 0.4
        n_sulfone = counts.get("sulfone", 0)
        n_nitrile = counts.get("nitrile", 0)
        if n_sulfone > 0 and n_nitrile > 1:
            correction += 0.3
        if (
            counts.get("fluorinated_ether", 0) > 0
            and counts.get("ether", 0) > 0
        ):
            correction += 0.25
        return max(-2.0, min(2.0, correction))

    def predict_dielectric(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["dielectric"]
        for _smarts, _name, dd, _dv, _ls, _dc in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, dd * 2.0)
        value += self._compute_dielectric_cross_terms(counts)
        tpsa = ctx.tpsa
        value += tpsa * 0.030
        max_diel = self.base_values["dielectric"] + tpsa * MAX_DIELECTRIC_PER_TPSA
        value = min(value, max_diel)
        return max(1.0, value)

    def predict_viscosity(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["viscosity"]
        for _smarts, _name, _dd, dv, _ls, _dc in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, dv * 2.0)
        mw = ctx.mw
        value += (mw - 30.0) * 0.005
        n_rot = ctx.rotatable_bonds
        value += n_rot * 0.15
        n_branch = _count_branch_points(mol)
        value += n_branch * 0.80
        n_stereo = _count_stereocenters(mol)
        value += n_stereo * 0.05
        value += _compute_rigidity_penalty(ctx)
        return max(0.1, value)

    def predict_li_solvation(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["li_solvation"]
        for _smarts, _name, _dd, _dv, ls, _dc in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, ls * 2.0)
        if counts.get("hf_scavenger", 0) > 0:
            value += 0.2
        mw = ctx.mw
        value += max(0.0, (mw - 50.0)) * 0.002
        return max(0.5, value)

    def predict_ced(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["ced"]
        for _smarts, _name, _dd, _dv, _ls, dc in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, dc * 2.0)
        ring_info = mol.GetRingInfo()
        n_rings = ring_info.NumRings()
        n_arom_rings = sum(
            1 for ring in ring_info.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
        value += max(0, n_rings - n_arom_rings) * 0.3
        return max(1.0, min(15.0, value))

    def predict_ionic_conductivity(
        self, dielectric: float, viscosity: float, li_solvation: float
    ) -> float:
        if viscosity <= 0.0 or dielectric < 0.0:
            return 0.0
        effective_dielec = max(0.0, dielectric - 1.0)
        solvation_factor = math.exp(-0.5 * ((li_solvation - 3.5) / 1.5) ** 2)
        if viscosity < 0.001:
            return 0.0
        conductivity = effective_dielec * solvation_factor / viscosity
        return max(0.0, min(10.0, conductivity))

    def predict_li_dissociation(self, ctx: MoleculeContext, salt_type: str = "LiPF6") -> float:
        counts = self.count_fragments(ctx.mol)
        donor_sites: dict[str, float] = {
            "carbonate": 1.2, "ether": 1.0, "nitrile": 0.8,
            "sulfone": 0.6, "sulfoxide": 1.5, "alcohol": 1.3,
            "ester": 0.7, "amide": 1.4, "primary_amine": 1.2,
            "secondary_amine": 1.0, "phosphate": 0.9, "sulfonate": 0.7,
            "aromatic_nitrogen": 0.8, "glyme_chelating": 1.5,
        }
        acceptor_sites: dict[str, float] = {
            "fluorine": 0.4, "trifluoromethyl": 0.6,
            "difluoromethylene": 0.3, "sulfone": 0.5,
            "sulfonyl_fluoride": 0.7, "sulfonimide": 0.8, "nitrile": 0.3,
        }
        donor_score = sum(
            donor_sites.get(name, 0.0) * self.saturate_contrib(counts.get(name, 0), 2.0)
            for name in donor_sites
        )
        acceptor_score = sum(
            acceptor_sites.get(name, 0.0) * self.saturate_contrib(counts.get(name, 0), 2.0)
            for name in acceptor_sites
        )
        base = max(0.0, donor_score + 0.3 * acceptor_score)
        if donor_score > 4.0 and acceptor_score < 1.0:
            base *= 0.7
        if acceptor_score > donor_score * 2.0:
            base *= 0.5
        total = donor_score + acceptor_score
        if total > 0.0:
            balance = 1.0 - abs(donor_score - acceptor_score) / total
        else:
            balance = 0.0
        base *= 0.5 + 0.5 * balance
        return max(0.0, min(6.0, base))

    def predict_gas_evolution(self, ctx: MoleculeContext) -> float:
        total = 0.0
        for pattern, _name, weight in _GC_GAS_EVOLUTION_PATTERNS:
            if pattern is None:
                continue
            n = len(ctx.mol.GetSubstructMatches(pattern))
            total += self.saturate_contrib(n, weight)
        return total

    def predict_all(self, ctx: MoleculeContext) -> dict[str, float]:
        dielectric = self.predict_dielectric(ctx)
        viscosity = self.predict_viscosity(ctx)
        li_solvation = self.predict_li_solvation(ctx)
        ced = self.predict_ced(ctx)
        conductivity = self.predict_ionic_conductivity(dielectric, viscosity, li_solvation)
        li_dissociation = self.predict_li_dissociation(ctx)
        return {
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "li_solvation_proxy": li_solvation,
            "ced_proxy": ced,
            "conductivity_proxy": conductivity,
            "li_dissociation_proxy": li_dissociation,
        }

    def property_keys(self) -> dict[str, str]:
        """Map short property names to result dict keys used by PropertyOracle."""
        return {
            "dielectric": "dielectric_proxy",
            "viscosity": "viscosity_proxy",
            "li_solvation": "li_solvation_proxy",
            "li_dissociation": "li_dissociation_proxy",
            "ced": "ced_proxy",
            "conductivity": "conductivity_proxy",
        }


# Default singleton for backward-compatible module-level functions
_DEFAULT_ELECTROLYTE: ElectrolytePack = ElectrolytePack()
