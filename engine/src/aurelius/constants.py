"""Constants used across the Aurelius pipeline.

All physical thresholds are centralized here with docstrings explaining
the physical justification for each value.
"""

from __future__ import annotations

from rdkit import Chem

FINGERPRINT_SIZE: int = 2048

# ---------------------------------------------------------------------------
# Ed25519 Public Key for Kernel Signature Verification
# ---------------------------------------------------------------------------
# The Certification Lab signs kernels using Ed25519 with a private key known
# only to the Lab. The Engine verifies kernels using this public key.
# This key corresponds to the Aurelius development seed.
# In production, replace this with the actual Lab's published public key.

KERNEL_PUBLIC_KEY: bytes = bytes.fromhex(
    "4e176c659b2f3544ee549931dfdedefbd4f95366dbf94ee6a06f05d0c7ef76cf"
)

# ---------------------------------------------------------------------------
# Frontier Orbital (HOMO/LUMO) Thresholds
# ---------------------------------------------------------------------------
# Physical basis: HOMO energy measures oxidative stability of the electrolyte.
# At the cathode (~4.0 V vs Li/Li+), solvent molecules with HOMO > -6.0 eV
# are thermodynamically susceptible to oxidation. The threshold of -6.0 eV
# corresponds to approximately 4.0 V vs Li/Li+ oxidation onset (empirical
# correlation: E_ox ≈ -HOMO_eV - 2.0 eV).

HOMO_THRESHOLD: float = -6.0
"""HOMO must be below this value (more negative) for oxidative stability at 4.0 V cathode."""

HOMO_SIGMOID_STEEPNESS: float = 5.0
"""Steepness of the sigmoid penalty for HOMO above threshold. k=5 gives
a sharp transition within ~0.5 eV of the threshold."""

# Physical basis: LUMO energy determines the SEI (Solid-Electrolyte Interphase)
# formation potential. An ideal LUMO is centered near -1.0 eV vs vacuum,
# which translates to ~1.0 V vs Li/Li+ SEI formation — high enough to form
# a stable passivation layer before electrolyte reduction, but not so high
# that it causes excessive first-cycle capacity loss.

LUMO_TARGET: float = -1.0
"""Target LUMO energy (eV) for ideal SEI formation at ~1.0 V vs Li/Li+."""

LUMO_SIGMA: float = 0.75
"""Gaussian width for LUMO reward. sigma=0.75 eV rewards LUMO in [-1.75, -0.25] eV."""

# ---------------------------------------------------------------------------
# Dielectric Constant Thresholds
# ---------------------------------------------------------------------------
# Physical basis: A minimum dielectric constant (ε > 5) is required to
# dissociate Li/Na salts into free ions. Low-ε solvents like pure ethers
# (ε ≈ 2-3) form ion pairs rather than free ions, reducing conductivity.
# High-ε solvents like EC (ε ≈ 90) or propylene carbonate (ε ≈ 65) are
# excellent salt dissociators. The proxy is a unitless model-derived value.

DIELECTRIC_TARGET: float = 5.0
"""Minimum target dielectric proxy value. Values below this indicate poor salt dissolution."""

DIELECTRIC_SIGMOID_STEEPNESS: float = 1.5
"""Steepness of the sigmoid reward for dielectric proxy above target."""

# New: TPSA-based dielectric cap coefficient
# Physical basis: Polar surface area limits the maximum achievable
# dielectric constant. A molecule with small TPSA cannot sustain
# a high dielectric regardless of fragment stacking.
MAX_DIELECTRIC_PER_TPSA: float = 0.60
"""Upper bound on dielectric contribution per unit TPSA (Å²).
Dielectric_proxy_max = base + TPSA * MAX_DIELECTRIC_PER_TPSA."""

# ---------------------------------------------------------------------------
# Viscosity Thresholds
# ---------------------------------------------------------------------------
# Physical basis: Ion mobility is inversely proportional to viscosity
# (Stokes-Einstein relation). High viscosity (> 5 cP at 25°C) severely
# limits Li+ transport. The proxy is a unitless model-derived value where
# higher = worse.

VISCOSITY_THRESHOLD: float = 2.5
"""Maximum target viscosity proxy value. Values above this indicate poor ion mobility."""

VISCOSITY_SIGMOID_STEEPNESS: float = 2.0
"""Steepness of the sigmoid penalty for viscosity above threshold."""

# ---------------------------------------------------------------------------
# SA Score (Synthetic Accessibility) Thresholds
# ---------------------------------------------------------------------------
# The custom electrolyte_synthetic_accessibility score ranges from 1 (easy)
# to 10 (very hard). Unlike RDKit's ChEMBL-trained SA score, it rewards
# common electrolyte motifs (carbonates, sulfones, nitriles, fluorinated
# groups) and penalises ring strain, stereocenters, and unstable bonds.
# For electrolyte molecules, scores below 4 are readily synthesizable.

SA_THRESHOLD: float = 4.0
"""SA score threshold below which molecules are considered readily synthesizable."""

SA_SIGMOID_STEEPNESS: float = 2.0
"""Steepness of the SA penalty sigmoid."""

# ---------------------------------------------------------------------------
# Resolved-score weighting for multi-objective composite
# ---------------------------------------------------------------------------
# Weights sum to 1.0. LUMO and Dielectric are primary drivers (SEI formation
# and salt dissociation), HOMO and Viscosity are secondary constraints, and
# SA is a soft filter.

SCORE_WEIGHT_LUMO: float = 0.23
"""Weight for LUMO SEI-formation reward."""

SCORE_WEIGHT_HOMO: float = 0.17
"""Weight for HOMO oxidative-stability penalty."""

SCORE_WEIGHT_DIELECTRIC: float = 0.17
"""Weight for dielectric-constant (salt dissolution) reward."""

SCORE_WEIGHT_VISCOSITY: float = 0.10
"""Weight for viscosity (ion mobility) penalty."""

SCORE_WEIGHT_LI_SOLVATION: float = 0.20
"""Weight for Li+ solvation energy proxy — penalises binding that is too tight
(poor transference number) or too weak (poor conductivity)."""

SCORE_WEIGHT_CED: float = 0.01
"""Weight for cohesive energy density (CED) proxy — SEI mechanical robustness."""

SCORE_WEIGHT_SA: float = 0.01
"""Weight for synthetic accessibility penalty.

Note: Al corrosion penalty is applied as a strict multiplicative factor
at the end of scoring (like hydrolytic instability), not as an additive
weight. This avoids the double-dipping bug where corrosion risk was
accidentally rewarded by the additive term."""

# ---------------------------------------------------------------------------
# Viability Threshold
# ---------------------------------------------------------------------------
# The total score ranges from 0 to 100. Molecules with total_score >= 50
# are considered viable candidates for further investigation.

VIABILITY_THRESHOLD: float = 50.0
"""Minimum total score for a molecule to be considered a viable discovery."""

DISCOVERY_THRESHOLD: float = 65.0
"""Total score threshold for a molecule to be flagged as a high-confidence discovery."""

# ---------------------------------------------------------------------------
# F/P/S Correction Layer — Inductive Shifts for Elements Absent from QM9
# ---------------------------------------------------------------------------
# Physical basis: QM9 contains only CHON elements. For electrolyte screening,
# fluorine (F), phosphorus (P), and sulfur (S) are critical. These inductive
# shifts are applied on top of RF predictions to correct for the QM9 blindspot.
#
# Values derived from literature on inductive effects of heteroatoms on
# frontier orbital energies:
#   - Fluorine: electron-withdrawing inductive effect stabilises both HOMO and
#     LUMO (negative shift) [J. Phys. Chem. A 2018, 122, 1234]
#   - CF3: strongly electron-withdrawing, ~3× per-fluorine effect
#   - Sulfone: strong electron withdrawal from S(=O)(=O)
#   - Phosphate: moderate electron withdrawal from P(=O)(OR)3

F_CORRECTION_HOMO: float = -0.15
"""Per-fluorine inductive shift on HOMO (eV)."""

F_CORRECTION_LUMO: float = -0.20
"""Per-fluorine inductive shift on LUMO (eV)."""

CF3_CORRECTION_HOMO: float = -0.50
"""Per-CF3-group inductive shift on HOMO (eV)."""

CF3_CORRECTION_LUMO: float = -0.30
"""Per-CF3-group inductive shift on LUMO (eV)."""

SULFONE_CORRECTION_HOMO: float = -0.50
"""Sulfone group shift on HOMO (eV)."""

SULFONE_CORRECTION_LUMO: float = -1.20
"""Sulfone group shift on LUMO (eV)."""

PHOSPHATE_CORRECTION_HOMO: float = -0.50
"""Phosphate group shift on HOMO (eV)."""

PHOSPHATE_CORRECTION_LUMO: float = -1.00
"""Phosphate group shift on LUMO (eV)."""

# ---------------------------------------------------------------------------
# Aluminium Corrosion Proxy — High-LUMO Fluorinated Solvent Penalty
# ---------------------------------------------------------------------------
# Physical basis: Fluorinated solvents with high LUMO (easily reduced) form
# AlF3 at the aluminium current collector, causing pitting corrosion.
# The penalty applies when LUMO > AL_CORROSION_LUMO_THRESHOLD AND the
# molecule contains AL_CORROSION_MIN_FluorINE fluorine atoms or CF3 groups.

AL_CORROSION_LUMO_THRESHOLD: float = -0.5
"""LUMO above this threshold (less negative) triggers Al corrosion risk."""

AL_CORROSION_MIN_FLUORINE: int = 2
"""Minimum number of fluorine atoms to trigger Al corrosion penalty."""

AL_CORROSION_PENALTY_FACTOR: float = 0.7
"""Multiplicative penalty applied when Al corrosion criteria are met."""

# ---------------------------------------------------------------------------
# Li+ Solvation Energy Proxy — Fragment-Additivity Target
# ---------------------------------------------------------------------------
# Physical basis: Li+ solvation energy (binding strength) is the single most
# important property for battery electrolyte conductivity. A molecule that
# binds Li+ too tightly yields a poor transference number (Li+ stays bound
# to the solvent). A molecule that binds too weakly cannot dissociate Li
# salts into free charge carriers.
#
# The GC proxy yields a unitless value. The ideal range corresponds to
# moderate binding: strong enough to dissociate LiPF6 (~3.0-4.5 eV binding),
# weak enough for acceptable transference.
#
# Target values from: J. Phys. Chem. B 2015, 119, 1315; Phys. Chem. Chem.
# Phys. 2018, 20, 12972.

LI_SOLVATION_TARGET: float = 3.5
"""Target Li+ solvation proxy value (moderate binding — Goldilocks zone)."""

LI_SOLVATION_SIGMA: float = 1.0
"""Gaussian width for Li+ solvation reward. sigma=1.0 rewards proxy in [2.5, 4.5]."""

# ---------------------------------------------------------------------------
# Cohesive Energy Density (CED) Proxy Thresholds
# ---------------------------------------------------------------------------
# Physical basis: CED measures the mechanical robustness of the SEI layer.
# CED = sum(molar attraction constants)^2 / molar volume. A higher CED proxy
# indicates stronger intermolecular cohesion, which translates to a more
# mechanically stable, defect-resistant SEI. The target of 5.0 corresponds
# to the approximate CED of sulfolane, a known SEI-forming additive with
# excellent mechanical properties.

CED_TARGET: float = 5.0
"""Target CED proxy value for mechanically robust SEI formation."""

CED_SIGMOID_STEEPNESS: float = 1.5
"""Steepness of the sigmoid reward for CED proxy above target."""

# ---------------------------------------------------------------------------
# Electrolyte-Like Filter Thresholds (Mutation Engine)
# ---------------------------------------------------------------------------

ELECTROLYTE_MIN_HETEROATOM_RATIO: float = 0.25
"""Minimum ratio of heteroatoms (O, F) to total heavy atoms for BRICS products."""

# ---------------------------------------------------------------------------
# Pre-compiled SMARTS patterns — compiled once at module load time
# ---------------------------------------------------------------------------
# These replace string-based SMARTS definitions that were previously
# recompiled inside per-molecule loops (massive performance bottleneck).
# Each pattern is compiled exactly once, at import time.

# Shared hydrolytically unstable motifs (used by pipeline.py, mutation.py)
# Format: (pattern, name, severity)
HYDROLYTICALLY_UNSTABLE_PATTERNS: list[tuple[Chem.Mol, str, float]] = [
    (Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])[OX2]"), "anhydride", 0.3),
    (Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX2]#[N]"), "acyl_cyanide", 0.4),
    (Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2][CX3](=[OX1])"), "sulfonate_ester", 0.2),
    (Chem.MolFromSmarts("[PX4](=[OX1])([OX2][CX4])[OX2][CX4]"), "phosphate_ester", 0.15),
    (Chem.MolFromSmarts("[Si]([OX2])[OX2]"), "silyl_ether", 0.3),
    (Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX2]=[CX2]"), "enol_ester", 0.35),
    (Chem.MolFromSmarts("[#6][CX3](=[OX1])[OX2][CX3](=[OX1])[#6]"), "geminal_diester", 0.2),
    (Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"), "acyl_halide", 0.4),
    (Chem.MolFromSmarts("[C]=[C]=[O]"), "terminal_ketene", 0.5),
]

# Gas-evolution-prone motifs for reductive gas generation proxy
# Physical basis: Linear carbonates decompose via one-electron reduction
# to liberate CO₂, while vulnerable acyclic ester arms with C-H bonds
# are susceptible to radical-mediated gas evolution. Cyclic carbonates
# (EC, PC) are excluded because their ring-constrained decomposition
# produces desirable SEI components with less gas. Fluorination of the
# alkyl chain mitigates gas generation by eliminating the vulnerable
# C-H bonds (replaced by C-F).
# Format: (pattern, name, penalty_weight)
_GC_GAS_EVOLUTION_PATTERNS: list[tuple[Chem.Mol, str, float]] = [
    (Chem.MolFromSmarts("O=C([OX2])[OX2]"), "carbonate", 1.5),
    (Chem.MolFromSmarts("[C;!R;H1,H2,H3][OX2][CX3](=[OX1])"), "acyclic_vulnerable_ester", 0.5),
]

# Acyl halide pattern — highly reactive toward hydrolysis, toxic
ACYL_HALIDE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]")
# Terminal ketene pattern — violently reactive toward water and nucleophiles
TERMINAL_KETENE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[C]=[C]=[O]")

# Electrochemically unstable motifs (used by mutation.py)
ELECTROCHEMICALLY_UNSTABLE_PATTERNS: list[tuple[Chem.Mol, str]] = [
    (Chem.MolFromSmarts("[OX2][OX2]"), "peroxide"),
    (Chem.MolFromSmarts("[CX4H1]([OX2H0])([OX2H0])"), "acetal"),
    (Chem.MolFromSmarts("[CX4H1]([OX2H0])([OH])"), "hemiacetal"),
    (Chem.MolFromSmarts("[OX2]1[OX2][OX2]1"), "trioxirane"),
    (Chem.MolFromSmarts("[CH2]1[CH2][CH2]1"), "cyclopropane"),
    (Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2]1"), "cyclobutane"),
    (Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"), "acyl_halide"),
    (Chem.MolFromSmarts("[C]=[C]=[O]"), "terminal_ketene"),
    # Reductive cleavage motifs for linear carbonates/esters with branched
    # O-alkyl groups that stabilise the radical formed after one-electron
    # reduction, promoting C-O bond cleavage and CO₂ evolution.
    (Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CH]([CH3])[CH3]"), "reductive_carbonate_cleavage_sec"),
    (Chem.MolFromSmarts("[CX3](=[OX1])[OX2][C]([CH3])([CH3])[CH3]"), "reductive_carbonate_cleavage_tert"),
]

# Individual pre-compiled patterns for chem_utils.py SA score
PEROXIDE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[OX2][OX2]")
ALDEHYDE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CH](=O)")
ANHYDRIDE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])")
CARBONATE_PATTERN: Chem.Mol = Chem.MolFromSmarts("O=C([OX2])[OX2]")
ETHER_PATTERN: Chem.Mol = Chem.MolFromSmarts("[OD2]([CX4])[CX4]")
SULFONE_SA_PATTERN: Chem.Mol = Chem.MolFromSmarts("S(=O)(=O)[CX4]")
NITRILE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[C]#[N]")
EPOXIDE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[OX2]1[CX4][CX4]1")

# TOM / Al-corrosion shared patterns (used by oracle.py, pipeline.py)
SULFONE_PATTERN: Chem.Mol = Chem.MolFromSmarts("S(=O)(=O)")
CF3_PATTERN: Chem.Mol = Chem.MolFromSmarts("[C](F)(F)F")
CARBONYL_F_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CX3](=O)[CH2][F]")
SULFONYL_F_PATTERN: Chem.Mol = Chem.MolFromSmarts("[SX4](=O)(=O)[F]")

# Hypofluorite pattern — O-F single bond (violently reactive, not viable as solvent)
HYPOFLUORITE_PATTERN: Chem.Mol = Chem.MolFromSmarts("[OX2][F]")
HYPOFLUORITE_PENALTY_FACTOR: float = 0.50
"""Multiplicative penalty for molecules containing O-F (hypofluorite) bonds.
Hypofluorites are violently reactive oxidisers that decompose exothermically
at room temperature, making them completely unsuitable as battery electrolyte
solvents (the EA exploits the methyl-to-fluorine SMARTS reaction to generate
these from carbonate/ether seed molecules)."""

# ---------------------------------------------------------------------------
# Commercial Building Blocks — Sigma-Aldrich Precursors for Electrolyte
# ---------------------------------------------------------------------------
# Hardcoded list of commercially available precursor/building block SMILES.
# Used by the BRICS building-block grounding penalty to ensure that
# EA-discovered molecules are synthesizable from real precursors.
# Focused on common electrolyte motifs: carbonates, ethers, nitriles,
# sulfones, fluorinated fragments, and simple alkyl building blocks.

COMMERCIAL_BUILDING_BLOCK_SMILES: tuple[str, ...] = (
    "CO",                    # Methanol
    "CCO",                   # Ethanol
    "CCCCO",                 # 1-Butanol
    "CCCO",                  # 1-Propanol
    "COC(=O)OC",             # Dimethyl carbonate
    "CCOC(=O)OCC",           # Diethyl carbonate
    "C1COC(=O)O1",           # Ethylene carbonate
    "CC1COC(=O)O1",          # Propylene carbonate
    "C1CCOC1",               # THF
    "C1COCCO1",              # 1,4-Dioxane
    "COCCOC",                # DME
    "CC#N",                  # Acetonitrile
    "CS(=O)(=O)C",           # Dimethyl sulfone
    "C1CS(=O)(=O)CC1",       # Sulfolane
    "CC(=O)OCC",             # Ethyl acetate
    "CC(=O)O",               # Acetic acid
    "FC(F)F",                # Fluoroform (CF3H)
    "CN",                    # Methylamine
    "CCOCC",                 # Diethyl ether
    "C1CO1",                 # Ethylene oxide
    "CS(=O)C",               # DMSO
    "CCN",                   # Ethylamine
    "CC(C)=O",               # Acetone
    "C(=O)O",                # Formic acid
    "CCOC=O",                # Ethyl formate
    "FC(F)(F)C(F)(F)F",      # Perfluoroethane (C2F6)
    # Expanded to cover common BRICS fragments from electrolyte candidates
    "COC(=O)C(F)(F)F",       # Methyl trifluoroacetate
    "CCC#N",                 # Propionitrile
    "FC(F)(F)C#N",           # Trifluoroacetonitrile
    "C=O",                   # Formaldehyde
    "COCOC",                 # Dimethoxymethane
    "CS(=O)(=O)F",           # Methanesulfonyl fluoride
    "FC(F)(F)OC(F)(F)F",     # Perfluoro ether
    "CS(=O)(=O)O",           # Methanesulfonic acid
    "FC(F)(F)CS(=O)(=O)C",   # Trifluoromethyl methyl sulfone
    "FC(F)(F)OC",            # Trifluoromethyl methyl ether
    "FC(F)(F)S(=O)(=O)F",    # Trifluoromethanesulfonyl fluoride
    "FCS(=O)(=O)F",          # Fluoromethylsulfonyl fluoride
    "CCS(=O)(=O)CC",         # Diethyl sulfone
    # Coverage for sulfone-cyano, sulfone-CF3, fluoro-nitrile combinations
    "CS(=O)(=O)CC#N",        # Methylsulfonylacetonitrile
    "N#CCS(=O)(=O)CC#N",     # Bis(cyanoethyl)sulfone (or similar)
    "FC(F)(F)C(F)(F)S(=O)(=O)C(F)(F)F",  # Perfluorobutyl sulfone
    "O=S(=O)(CC(F)(F)F)CC(F)(F)F",       # Bis(trifluoroethyl) sulfone
    "N#CCO",                 # Glycolonitrile (cyano-methanol)
    "FC(F)(F)CO",            # Trifluoroethanol
    "FC(F)(F)C(F)(F)CO",     # Perfluoropropanol
    "FC(F)(F)S(=O)(=O)C(F)(F)F",         # Trifluoromethanesulfonic anhydride like
    # Fluorinated alcohols
    "FC(F)(F)C(O)C(F)(F)F",              # Hexafluoroisopropanol (HFIP)
    "FC(F)(F)CC(F)(F)CO",                # 2,2,3,3-Tetrafluoro-1,4-butanediol
    "FC(F)(F)C(F)(F)C(F)(F)CO",          # 2,2,3,3,4,4,5,5-Octafluoro-1-pentanol
    # Cyclic sulfates / sultones
    "O=S1(=O)OCCO1",                     # Ethylene sulfate (cyclic)
    "O=S1(=O)OCCCO1",                    # 1,3-Propylene sulfate
    "O=S1(=O)CCCCO1",                    # 1,4-Butylene sulfate
    # Branched carbonates
    "CC(C)OC(=O)OC(C)C",                 # Diisopropyl carbonate
    "CC(C)(C)OC(=O)OC(C)(C)C",           # Di-tert-butyl carbonate
    "CC(C)OC(=O)OCC",                    # Ethyl isopropyl carbonate
    # Dinitriles / aromatic nitriles
    "N#CCCC#N",                          # Adiponitrile
    "N#CCCCC#N",                         # Pimelonitrile
    "N#CCCCCC#N",                        # Suberonitrile
    "N#Cc1ccccc1",                       # Benzonitrile
    "N#CCc1ccccc1",                      # Phenylacetonitrile
    # Fluorinated / aromatic sulfones
    "CCS(=O)(=O)C(F)(F)F",               # Ethyl trifluoromethyl sulfone
    "O=S(=O)(c1ccccc1)c1ccccc1",         # Diphenyl sulfone
    # Organophosphates
    "COP(=O)(OC)OC",                     # Trimethyl phosphate
    "CCOP(=O)(OCC)OCC",                  # Triethyl phosphate
    # Glyme ethers / specialty
    "COCCOCCOC",                         # Triglyme
    "CC1CCCO1",                          # 2-Methyltetrahydrofuran
)

# ---------------------------------------------------------------------------
# Stable SEI-Forming Motifs — Substructure Patterns
# ---------------------------------------------------------------------------
# Physical basis: Certain functional groups are empirically known to form
# stable Solid-Electrolyte Interphase (SEI) layers in lithium-ion batteries.
# Molecules containing these motifs are more likely to produce a durable,
# low-impedance SEI. This is used as a SMARTS-based motif check in the
# PropertyOracle — no BDE (Bond Dissociation Energy) calculation is needed.
#
# STABLE_SEI_MOTIFS: Pre-compiled SMARTS patterns for motifs that correlate
# with stable SEI formation in the literature:
#   1. CF3 (fluorinated carbon) — forms LiF-rich SEI
#   2. Cyclic carbonate (e.g., EC, PC) — the canonical SEI-forming motif
#   3. Sultone (5-ring cyclic sulfonate ester) — HF-scavenger, stable SEI

STABLE_SEI_MOTIFS: tuple[Chem.Mol, ...] = (
    Chem.MolFromSmarts("[C](F)(F)F"),                          # fluorinated carbon (CF3)
    Chem.MolFromSmarts("[OX2]1[CX3](=O)[OX2][CX4][CX4]1"),    # cyclic carbonate (5-ring)
    Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][OX2]1"),# sultone (5-ring)
)

# SEI formation LUMO window
# Physical basis: LUMO in [-1.5, 0.0] eV vs vacuum corresponds to
# SEI formation at ~1.5-3.0 V vs Li/Li+. Molecules with LUMO in this
# window are thermodynamically capable of reductive SEI formation.
SEI_LUMO_LOWER: float = -1.5
"""Lower bound of SEI formation LUMO window (eV)."""
SEI_LUMO_UPPER: float = 0.0
"""Upper bound of SEI formation LUMO window (eV)."""

SEI_MOTIF_PENALTY_FACTOR: float = 0.85
"""Multiplicative penalty applied when a molecule has LUMO in the SEI
formation window but lacks any known stable SEI-forming motif."""

# ---------------------------------------------------------------------------
# SEI Fracture Toughness Proxy — Cross-linking Motif Patterns
# ---------------------------------------------------------------------------
# Physical basis: The SEI layer undergoes mechanical stress during anode
# volume expansion (up to 10% for graphite, >300% for silicon). SEI fracture
# toughness correlates with molecular rigidity and the presence of
# polymerizable / cross-linkable functional groups that can form a
# mechanically robust, defect-resistant SEI network.
#
# These SMARTS match motifs known to promote cross-linking or rigid SEI:
#   1. Vinyl (terminal C=C) — radical-polymerizable, forms cross-linked
#      poly(vinylene) networks in the SEI
#   2. Acrylate (C=C-C(=O)-O-) — radical-polymerizable ester, forms
#      cross-linked polyacrylate SEI layers
#
# Sultone (already defined in STABLE_SEI_MOTIFS) is also cross-linkable.

VINYL_CROSSLINK_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CX3]=[CX3]")
"""Alkene (C=C, including terminal vinyl) — radical-polymerizable cross-linking motif."""

ACRYLATE_CROSSLINK_PATTERN: Chem.Mol = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX2]=[CX2]")
"""Acrylate / enol ester — radical-polymerizable cross-linking motif."""

SULTONE_CROSSLINK_PATTERN: Chem.Mol = Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][OX2]1")
"""Sultone (5-ring cyclic sulfonate ester) — HF-scavenger and cross-linkable SEI motif."""

# ---------------------------------------------------------------------------
# SEI Fracture Toughness Proxy — Scoring Thresholds
# ---------------------------------------------------------------------------
# Physical basis: The SEI fracture proxy combines molecular rigidity
# (ring count minus aromatic rings) and cross-linking motif presence into
# a unitless score. A value >= 4.0 indicates molecules expected to form
# mechanically robust SEI layers resistant to anode expansion fracture.

SEI_FRACTURE_TARGET: float = 4.0
"""Target SEI fracture toughness proxy value for a robust SEI layer."""

SEI_FRACTURE_SIGMOID_STEEPNESS: float = 1.5
"""Steepness of the sigmoid reward for SEI fracture proxy above target."""

# ---------------------------------------------------------------------------
# SEI Fracture Resolved-score Weight
# ---------------------------------------------------------------------------
# Physical basis: SEI fracture toughness is a first-order constraint for
# battery cycle life — a fractured SEI exposes fresh anode to electrolyte,
# causing capacity fade. Weight allocated from CED (partial overlap),
# SA (soft filter), and Viscosity (mechanical property overlap).
# Weights sum to 1.0 across all SCORE_WEIGHT_*.

SCORE_WEIGHT_SEI_FRACTURE: float = 0.06
"""Weight for SEI fracture toughness proxy — mechanical robustness of SEI."""

SCORE_WEIGHT_GAS_EVOLUTION: float = 0.05
"""Weight for gas evolution penalty — penalises molecules prone to reductive
gas generation (CO₂/CO from linear carbonates and vulnerable esters)."""

# ---------------------------------------------------------------------------
# Net Progress Normalization Constants
# ---------------------------------------------------------------------------
# Ceilings for Net Progress simplicity cost normalization. Chosen to reflect
# the approximate upper bound of a lean, maintainable v10.x codebase.
# Adjust ONLY if a fundamental architectural shift is approved.
NET_PROGRESS_LOC_NORM: float = 5000.0
NET_PROGRESS_CC_NORM: float = 5.0
NET_PROGRESS_DEP_NORM: float = 10.0
NET_PROGRESS_ARCH_NORM: float = 50.0

# ---------------------------------------------------------------------------
# Mixture Synergy Constants
# ---------------------------------------------------------------------------
# Physical basis: The Margules-inspired interaction term scales the non-ideal
# mixing contribution as A = |d₁-d₂|·|v₁-v₂|·scale. The scale of 0.125
# (derived from /8.0) gives ~0.5 bonus at 50:50 for complementary pairs.
# The cap of 6.0 prevents unbounded synergy scoring from gaming the
# multi-objective optimisation.

MIXTURE_SYNERGY_CAP: float = 6.0
"""Upper bound for the mixture synergy bonus (prevents gaming)."""

MARGULES_INTERACTION_SCALE: float = 0.125
"""Scaling factor for the Margules-inspired non-ideal mixing term (1/8 = 0.125)."""

COMPLEMENTARITY_DIELECTRIC_THRESH: float = 4.0
"""Dielectric threshold above which a component is considered 'high-dielectric'."""

COMPLEMENTARITY_VISCOSITY_THRESH: float = 1.5
"""Viscosity threshold below which a component is considered 'low-viscosity'."""
