"""PropertyOracle — Multi-objective property prediction via fragment-additivity (Group Contribution).

This module provides a pure, transparent **Fragment-Additivity (Group Contribution)** model
for predicting electrolyte-relevant properties:

  1. **HOMO energy** (eV) — oxidative stability
  2. **LUMO energy** (eV) — SEI formation window
  3. **Dielectric Constant proxy** (unitless) — salt dissolution capability
  4. **Viscosity proxy** (unitless) — ion mobility indicator

All models are linear group-contribution models requiring no training data.
Fragment contributions are calibrated from literature group-contribution methods
(Joback-style) and known inductive effects for electrolyte motifs (S, P, F).

Usage:
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
    print(result["homo_eV"])            # e.g. -6.82
    print(result["lumo_eV"])            # e.g. -0.94
    print(result["dielectric_proxy"])    # e.g. 2.4
    print(result["viscosity_proxy"])     # e.g. 1.3
"""

from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem

logger = logging.getLogger(__name__)

_DATA_SOURCE: str = "pure fragment-additivity (Group Contribution — no training data required)"


def get_data_source() -> str:
    """Return a human-readable string describing the oracle's data source."""
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models
# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS patterns with property contributions.
#
# For HOMO/LUMO:
#   Shifts represent the change in frontier orbital energy when the fragment
#   is added to a simple alkane scaffold. Values are calibrated from QM9
#   statistics for CHON groups, and from literature-based inductive effects
#   for S/P/F groups absent from QM9.
#   A positive Δ means the orbital energy increases (less stable, closer to zero).
#   A negative Δ means the orbital energy decreases (more stable, more negative).
#
# For Dielectric proxy:
#   Contribution to the effective polarity / dielectric constant.
#   Higher values → higher dielectric constant (better salt dissolution).
#   Calibrated from Joback group contributions to dielectric constant and
#   known values for common electrolyte solvents.
#
# For Viscosity proxy:
#   Contribution to liquid-phase viscosity.
#   Higher values → higher viscosity (worse ion mobility).
#   Calibrated from Joback group contributions to viscosity and
#   known values for common electrolyte solvents.

# (smarts, name, homo_shift, lumo_shift, dielectric_contrib, viscosity_contrib)
_GC_FRAGMENTS: list[tuple[str, str, float, float, float, float]] = [
    # Carbonyl groups (from QM9 fit + Joback dielectric/viscosity)
    ("[CX3](=O)[OX2H0]",       "ester",              0.6, -0.4,  2.5,  0.6),
    ("[CX3](=O)[OH]",          "carboxylic_acid",    0.3, -0.6,  4.0,  1.0),
    ("[CX3](=O)[NX3]",         "amide",              0.8, -0.2,  5.0,  0.8),
    ("[CX3](=O)[CX3]",         "ketone",             0.7, -0.8,  3.0,  0.5),
    ("[CH](=O)",               "aldehyde",           0.5, -1.2,  2.5,  0.3),
    # Carbonates — high dielectric, critical for electrolyte applications
    ("O=C([OX2])[OX2]",        "carbonate",          1.0, -0.5,  5.0,  0.7),
    # Ethers — moderate dielectric, low viscosity (desirable)
    ("[OD2]([CX4])[CX4]",      "ether",              0.4,  0.2,  1.5, -0.3),
    # Alcohols — high dielectric but protic (limited use)
    ("[OH][CX4]",              "alcohol",            0.3,  0.1,  4.5,  1.2),
    # Amines
    ("[NX3;H2][CX4]",          "primary_amine",      0.8,  0.5,  3.5,  0.5),
    ("[NX3;H1]([CX4])[CX4]",   "secondary_amine",    0.9,  0.4,  2.5,  0.4),
    ("[NX3;H0]([CX4])([CX4])[CX4]", "tertiary_amine", 0.5, 0.3,  1.5,  0.3),
    # Nitriles — very high dielectric, low viscosity (excellent)
    ("[C]#[N]",                "nitrile",           -0.3, -0.6,  8.0,  0.4),
    # Alkenes
    ("[CX3]=[CX3]",            "alkene",             0.8,  0.5,  0.5,  0.1),
    # Alkynes
    ("[CX2]#[CX2]",            "alkyne",             0.6, -0.3,  1.0,  0.2),
    # Aromatic carbon
    ("[c]",                    "aromatic_carbon",    1.2,  1.0,  0.5,  0.5),
    # Halogens
    ("[F]",                    "fluorine",          -0.15, -0.08, 0.0,  0.1),
    ("[Cl]",                   "chlorine",          -0.2, -0.15,  0.5,  0.2),
    ("[Br]",                   "bromine",           -0.15, -0.1,   0.5,  0.3),
    # Electrolyte-specific groups (S, P, B — absent from QM9)
    ("S(=O)(=O)[CX4]",         "sulfone",           -0.5, -1.2,  5.0,  0.5),
    ("S(=O)(=O)[OX2]",         "sulfonate",         -0.6, -1.0,  5.5,  0.6),
    ("S(=O)(=O)F",             "sulfonyl_fluoride", -0.7, -1.3,  4.0,  0.4),
    ("[PX4](=O)([OX2])([OX2])[OX2]", "phosphate",   -0.5, -1.0,  4.0,  0.8),
    ("[C](F)(F)F",             "trifluoromethyl",   -0.5, -0.3,  0.5,  0.2),
    ("[C](F)(F)",              "difluoromethylene", -0.3, -0.2,  0.3,  0.1),
    ("[BX3]([OX2])",           "boronate",          -0.4, -1.5,  2.0,  0.7),
    ("[S]([CX4])[CX4]",        "thioether",          0.2, -0.3,  1.0,  0.2),
]

# Base values for a simple alkane (ethane reference)
_GC_BASE_HOMO: float = -9.2
_GC_BASE_LUMO: float = 2.8
_GC_BASE_DIELECTRIC: float = 1.9   # ethane ε ≈ 1.9
_GC_BASE_VISCOSITY: float = 0.1    # ethane is nearly inviscid


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each fragment SMARTS pattern in a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping fragment name to count.
    """
    counts: dict[str, int] = {}
    for _smarts, name, _dh, _dl, _dd, _dv in _GC_FRAGMENTS:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(_smarts))
        counts[name] = len(matches)
    return counts


def predict_fragment_additivity(
    mol: Chem.Mol,
) -> tuple[float, float]:
    """Predict HOMO/LUMO using a fragment-additivity (group contribution) model.

    HOMO = base_homo + sum(n_i * Δhomo_i)
    LUMO = base_lumo + sum(n_i * Δlumo_i)

    Args:
        mol: RDKit Mol object.

    Returns:
        (homo_eV, lumo_eV) tuple.
    """
    counts = _count_fragments(mol)
    homo = _GC_BASE_HOMO
    lumo = _GC_BASE_LUMO
    for _smarts, _name, dh, dl, _dd, _dv in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        homo += n * dh
        lumo += n * dl
    return homo, lumo


def predict_dielectric_proxy(mol: Chem.Mol) -> float:
    """Predict a dielectric constant proxy via fragment-additivity.

    The proxy represents the effective polarity/dielectric contribution
    of the molecule. Higher values indicate better salt dissolution capability.

    Returns:
        Unitless dielectric proxy (typically 1–15 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, _dh, _dl, dd, _dv in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dd
    # TPSA-based correction: molecules with higher polar surface area
    # have higher dielectric constants
    from rdkit.Chem import rdMolDescriptors
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    value += tpsa * 0.02
    return max(1.0, value)


def predict_viscosity_proxy(mol: Chem.Mol) -> float:
    """Predict a viscosity proxy via fragment-additivity.

    The proxy represents relative viscosity contribution.
    Higher values indicate higher viscosity (worse ion mobility).

    Returns:
        Unitless viscosity proxy (typically 0.5–5.0 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_VISCOSITY
    for _smarts, _name, _dh, _dl, _dd, dv in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dv
    # MW correction: larger molecules are more viscous
    from rdkit.Chem import rdMolDescriptors
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    value += (mw - 30.0) * 0.005
    # Rotatable bond correction: flexible chains increase viscosity
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    value += n_rot * 0.15
    return max(0.1, value)


# ---------------------------------------------------------------------------
# PropertyOracle — Pure Fragment-Additivity Multi-Objective Predictor
# ---------------------------------------------------------------------------

class PropertyOracle:
    """Multi-objective property oracle using fragment-additivity (no training data).

    Predicts four electrolyte-relevant properties:
      - **HOMO energy** (eV) — oxidative stability
      - **LUMO energy** (eV) — SEI formation window
      - **Dielectric proxy** — salt dissolution capability
      - **Viscosity proxy** — ion mobility indicator (higher = worse)

    The model requires zero training data. All predictions are derived from
    linear group-contribution equations calibrated from literature values.
    Predictions are cached by SMILES string to avoid redundant computation.

    Requirements:
        - ``rdkit`` must be importable
    """

    _CACHE: dict[str, dict[str, Any]] | None = None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the PropertyOracle.

        Args:
            model_path: Reserved for future compatibility (unused with GC model).
        """
        pass

    def evaluate(self, smiles: str, mol: Chem.Mol | None = None) -> dict[str, Any]:
        """Evaluate a molecule and return predicted properties.

        All four property models (HOMO, LUMO, Dielectric, Viscosity) run
        simultaneously from the same fragment counts. Results are cached
        by SMILES string.

        Args:
            smiles: Canonical or isomeric SMILES string.
            mol: Optional pre-parsed RDKit Mol object (avoids redundant parsing).

        Returns:
            Dictionary with keys:
              - ``homo_eV`` (float)
              - ``lumo_eV`` (float)
              - ``gap_eV`` (float)
              - ``dielectric_proxy`` (float)
              - ``viscosity_proxy`` (float)
              - ``domain_applicable`` (True — GC handles all chemistry)
              - ``domain_reason`` (str — always "fragment-additivity model")

        Raises:
            ValueError: If SMILES is invalid.
        """
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        homo, lumo = predict_fragment_additivity(mol)
        gap = lumo - homo
        dielectric = predict_dielectric_proxy(mol)
        viscosity = predict_viscosity_proxy(mol)

        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "dielectric_proxy": round(dielectric, 4),
            "viscosity_proxy": round(viscosity, 4),
            "domain_applicable": True,
            "domain_reason": "fragment-additivity (GC) model — handles all chemistries",
        }

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def evaluate_with_ood_penalty(self, smiles: str) -> dict[str, Any]:
        """Evaluate a molecule (backward-compatible wrapper).

        With the pure GC oracle there is no OOD penalty — all molecules
        are handled by the same model.

        Args:
            smiles: Canonical SMILES string.

        Returns:
            Result dict.
        """
        return self.evaluate(smiles)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist the cache to disk with joblib (no model to save).

        The GC model requires no training, so only the cache is persisted.

        Args:
            path: File path for the joblib dump.
        """
        import joblib

        payload: dict[str, Any] = {
            "cache": self._CACHE,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: cache saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        """Load cached predictions from a joblib cache file.

        Args:
            path: File path to the joblib dump.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        import joblib

        try:
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
