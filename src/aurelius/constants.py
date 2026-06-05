"""Constants used across the Aurelius pipeline.

All physical thresholds are centralized here with docstrings explaining
the physical justification for each value.
"""

from __future__ import annotations

from rdkit import Chem

FINGERPRINT_SIZE: int = 2048

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

SCORE_WEIGHT_VISCOSITY: float = 0.14
"""Weight for viscosity (ion mobility) penalty."""

SCORE_WEIGHT_LI_SOLVATION: float = 0.20
"""Weight for Li+ solvation energy proxy — penalises binding that is too tight
(poor transference number) or too weak (poor conductivity)."""

SCORE_WEIGHT_SA: float = 0.09
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

)
