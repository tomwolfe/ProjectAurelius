"""PropertyOracle — Hybrid QSPR oracle: Random Forest (HOMO/LUMO) + Group Contribution (Dielectric/Viscosity).

Architecture
------------
A **two-tier hybrid** model for predicting electrolyte-relevant properties:

  **Tier 1 — Random Forest (HOMO / LUMO)**
    A Scikit-Learn ``RandomForestRegressor`` trained on the bundled QM9 subset
    (500 molecules). Featurization uses ECFP4 (Morgan radius=2, 2048 bits)
    concatenated with five RDKit global descriptors (MW, LogP, TPSA, RingCount,
    RotatableBonds) → 2053-dim input vector.

    If the RF weights file (``models/oracle_rf.joblib``) is not found, the
    oracle transparently falls back to the linear fragment-additivity model
    for HOMO/LUMO with a logged warning.

  **Tier 2 — Fragment-Additivity GC (Dielectric / Viscosity)**
    The dielectric proxy and viscosity proxy remain linear group-contribution
    models (Joback-style) because QM9 does not contain those properties.
    Anti-gaming diminishing-returns are applied to prevent fragment-stacking.

Usage:
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
    print(result["homo_eV"])            # e.g. -6.82  (RF or GC)
    print(result["lumo_eV"])            # e.g. -0.94  (RF or GC)
    print(result["dielectric_proxy"])    # e.g. 2.4   (GC + anti-gaming)
    print(result["viscosity_proxy"])     # e.g. 1.3   (GC + anti-gaming)

Training:
    aurelius train --model-path models/oracle_rf.joblib
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

_DATA_SOURCE: str = "pure fragment-additivity (Group Contribution — fallback mode)"

# Default path for the trained RF model
DEFAULT_RF_MODEL_PATH: str = "models/oracle_rf.joblib"


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

# Anti-gaming — diminishing returns for fragment stacking
# If any fragment appears more than this many times, apply sqrt scaling
_ANTI_GAMING_MAX_EFFECTIVE: int = 2

def _apply_anti_gaming(counts: dict[str, int]) -> dict[str, int]:
    """Apply diminishing-returns to fragment counts to prevent stacking.

    For each fragment count > ``_ANTI_GAMING_MAX_EFFECTIVE``, the effective
    count becomes ``max_effective + sqrt(count - max_effective)``, so that
    stacking many copies of the same polar group gives sub-linear benefit.

    Args:
        counts: Raw fragment counts from ``_count_fragments``.

    Returns:
        Effective fragment counts with diminishing returns applied.
    """
    effective: dict[str, int] = {}
    for name, count in counts.items():
        if count > _ANTI_GAMING_MAX_EFFECTIVE:
            effective[name] = int(round(
                _ANTI_GAMING_MAX_EFFECTIVE + (count - _ANTI_GAMING_MAX_EFFECTIVE) ** 0.5
            ))
        else:
            effective[name] = count
    return effective


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


def train_oracle_rf(save_path: str = DEFAULT_RF_MODEL_PATH) -> str:
    """Train a RandomForest regressor for HOMO/LUMO on the bundled QM9 data.

    Featurises each molecule with ``generate_full_feature_vector`` (2048-bit
    ECFP4 + 5 RDKit descriptors → 2053-dim) and trains a multi-output
    ``RandomForestRegressor``.

    Args:
        save_path: Where to persist the trained model via ``joblib.dump``.

    Returns:
        The absolute path the model was saved to.

    Raises:
        RuntimeError: If the QM9 data cannot be loaded or featurisation fails
            for all entries.
    """
    from aurelius.data.loaders import load_qm9_homo_lumo_data
    from aurelius.utils.chem_utils import generate_full_feature_vector

    logger.info("Loading QM9 HOMO/LUMO data...")
    data = load_qm9_homo_lumo_data()

    X_list: list[np.ndarray] = []
    y_homo: list[float] = []
    y_lumo: list[float] = []
    skipped = 0
    for smiles, homo, lumo in data:
        try:
            fp = generate_full_feature_vector(smiles)
            X_list.append(fp)
            y_homo.append(homo)
            y_lumo.append(lumo)
        except Exception:
            skipped += 1
            continue

    if not X_list:
        raise RuntimeError("All QM9 molecules failed featurisation — cannot train RF model.")
    if skipped:
        logger.warning("Skipped %d/%d molecules that could not be featurised.", skipped, len(data))

    X = np.array(X_list, dtype=np.float32)
    y = np.column_stack([y_homo, y_lumo])

    logger.info("Training RandomForest on %d samples with %d features...", X.shape[0], X.shape[1])
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("RF model saved to %s", save_path)
    return save_path


def predict_fragment_additivity(
    mol: Chem.Mol,
) -> tuple[float, float]:
    """Predict HOMO/LUMO using a fragment-additivity (group contribution) model.

    .. note::
       This is the **fallback** for when no RF model is available. When the
       RF model is loaded via ``PropertyOracle(model_path=...)``, HOMO/LUMO
       predictions come from the Random Forest instead.

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
    """Predict a dielectric constant proxy via fragment-additivity + anti-gaming.

    The proxy represents the effective polarity/dielectric contribution
    of the molecule. Higher values indicate better salt dissolution capability.

    Anti-gaming: if the same highly-polar fragment appears more than
    ``_ANTI_GAMING_MAX_EFFECTIVE`` times, its effective contribution follows a
    diminishing-return (sqrt) curve to discourage fragment-stacking.

    Returns:
        Unitless dielectric proxy (typically 1–15 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    effective = _apply_anti_gaming(counts)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, _dh, _dl, dd, _dv in _GC_FRAGMENTS:
        n = effective.get(_name, 0)
        value += n * dd
    # TPSA-based correction: molecules with higher polar surface area
    # have higher dielectric constants
    from rdkit.Chem import rdMolDescriptors
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    value += tpsa * 0.02
    return max(1.0, value)


def predict_viscosity_proxy(mol: Chem.Mol) -> float:
    """Predict a viscosity proxy via fragment-additivity + anti-gaming.

    The proxy represents relative viscosity contribution.
    Higher values indicate higher viscosity (worse ion mobility).

    Anti-gaming: if the same highly-polar fragment appears more than
    ``_ANTI_GAMING_MAX_EFFECTIVE`` times, its effective contribution follows a
    diminishing-return (sqrt) curve to discourage fragment-stacking.

    Returns:
        Unitless viscosity proxy (typically 0.5–5.0 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    effective = _apply_anti_gaming(counts)
    value = _GC_BASE_VISCOSITY
    for _smarts, _name, _dh, _dl, _dd, dv in _GC_FRAGMENTS:
        n = effective.get(_name, 0)
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
# PropertyOracle — Hybrid RF + GC Multi-Objective Predictor
# ---------------------------------------------------------------------------

class PropertyOracle:
    """Multi-objective property oracle using a hybrid RF + GC architecture.

    Predicts four electrolyte-relevant properties:

      - **HOMO energy** (eV) — Random Forest (QM9-trained) or GC fallback
      - **LUMO energy** (eV) — Random Forest (QM9-trained) or GC fallback
      - **Dielectric proxy** — Group Contribution + anti-gaming (always)
      - **Viscosity proxy** — Group Contribution + anti-gaming (always)

    **Hybrid workflow:**

    1. If a trained RF ``.joblib`` file is found at ``model_path``, it is
       loaded and used for HOMO/LUMO predictions.  The RF featurises each
       molecule as ECFP4 + 5 RDKit descriptors (2053-dim).
    2. If no RF file is found, the oracle falls back to the linear
       fragment-additivity (GC) model for HOMO/LUMO with a logged warning.
    3. Dielectric and Viscosity always use the GC model with anti-gaming
       diminishing-returns applied to prevent fragment-stacking.

    Predictions are cached by SMILES string to avoid redundant computation.

    Requirements:
        - ``rdkit``, ``scikit-learn``, ``numpy``, ``joblib``
    """

    _CACHE: dict[str, dict[str, Any]] | None = None
    _rf_model: RandomForestRegressor | None = None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the PropertyOracle.

        Attempts to load a pre-trained RF model for HOMO/LUMO prediction.
        Falls back to pure GC (fragment-additivity) if no model is found.

        Args:
            model_path: Path to a trained RF ``.joblib`` file. If *None*,
                defaults to ``DEFAULT_RF_MODEL_PATH`` (``models/oracle_rf.joblib``).
        """
        global _DATA_SOURCE
        self._rf_model = None

        if model_path is None:
            model_path = DEFAULT_RF_MODEL_PATH

        if os.path.exists(model_path):
            try:
                self._rf_model = joblib.load(model_path)
                _DATA_SOURCE = (
                    "hybrid: RF (QM9-trained) for HOMO/LUMO + "
                    "fragment-additivity GC for dielectric/viscosity"
                )
                logger.info("RF model loaded from %s", model_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load RF model from %s: %s. Falling back to pure GC.",
                    model_path, exc,
                )
                self._rf_model = None
        else:
            logger.warning(
                "RF model not found at %s. Falling back to pure GC for HOMO/LUMO.",
                model_path,
            )

        if self._rf_model is None:
            _DATA_SOURCE = "pure fragment-additivity (Group Contribution — fallback mode)"

    def _predict_homo_lumo_rf(self, smiles: str) -> tuple[float, float]:
        """Predict HOMO/LUMO using the loaded Random Forest model.

        Args:
            smiles: SMILES string.

        Returns:
            (homo_eV, lumo_eV) tuple.
        """
        from aurelius.utils.chem_utils import generate_full_feature_vector

        fp = generate_full_feature_vector(smiles)
        pred = self._rf_model.predict(fp.reshape(1, -1))[0]  # type: ignore[union-attr]
        homo = float(np.clip(pred[0], -15.0, 5.0))
        lumo = float(np.clip(pred[1], -5.0, 10.0))
        return homo, lumo

    def evaluate(self, smiles: str, mol: Chem.Mol | None = None) -> dict[str, Any]:
        """Evaluate a molecule and return predicted properties.

        Uses the RF model for HOMO/LUMO when available (falls back to GC),
        and always uses GC + anti-gaming for dielectric/viscosity.
        Results are cached by SMILES string.

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
              - ``domain_applicable`` (bool)
              - ``domain_reason`` (str)

        Raises:
            ValueError: If SMILES is invalid.
        """
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # HOMO / LUMO via RF (with GC fallback)
        if self._rf_model is not None:
            try:
                homo, lumo = self._predict_homo_lumo_rf(smiles)
                domain_reason = (
                    "RF (QM9-trained) for HOMO/LUMO + "
                    "fragment-additivity GC for dielectric/viscosity"
                )
            except Exception:
                logger.warning(
                    "RF prediction failed for %s, falling back to GC.", smiles
                )
                homo, lumo = predict_fragment_additivity(mol)
                domain_reason = "fragment-additivity (GC) fallback for HOMO/LUMO"
        else:
            homo, lumo = predict_fragment_additivity(mol)
            domain_reason = "fragment-additivity (GC) model"

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
            "domain_reason": domain_reason,
        }

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def evaluate_with_ood_penalty(self, smiles: str) -> dict[str, Any]:
        """Evaluate a molecule (backward-compatible wrapper).

        With the hybrid oracle there is no out-of-distribution penalty —
        all valid molecules receive a prediction. The ``domain_reason``
        field indicates whether RF or GC was used.

        Args:
            smiles: Canonical SMILES string.

        Returns:
            Result dict.
        """
        return self.evaluate(smiles)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist the cache to disk with joblib.

        The RF model weights are saved separately via ``train_oracle_rf()``;
        only the SMILES cache is persisted here.

        Args:
            path: File path for the joblib dump.
        """
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
