"""PropertyOracle — Hybrid QSPR oracle with F/P/S correction for QM9 blindspot.

Architecture
------------
A **three-tier hybrid** model for predicting electrolyte-relevant properties:

  **Tier 1 — Random Forest (HOMO / LUMO)**
    A Scikit-Learn ``RandomForestRegressor`` trained on the bundled QM9 subset.
    Featurization uses the pre-computed ``MoleculeContext.get_feature_vector()``
    (2048-bit ECFP4 + 5 RDKit global descriptors → 2053-dim).

  **Tier 2 — F/P/S Inductive Correction Layer**
    Since QM9 contains only CHON, the RF is blind to fluorine, phosphorus,
    and sulfur.  A physics-informed correction layer applies inductive shifts
    on top of RF predictions for any F/P/S-containing groups detected via
    SMARTS.  This ensures that fluorinated carbonates, sulfones, and
    phosphates receive physically plausible HOMO/LUMO values even though
    the RF was never trained on such elements.

  **Tier 3 — Fragment-Additivity GC (Dielectric / Viscosity)**
    The dielectric proxy and viscosity proxy remain linear group-contribution
    models (Joback-style).  A TPSA-based cap replaces the legacy sqrt
    anti-gaming to prevent unrealistic dielectric stacking on small molecules.

Usage:
    from aurelius.scoring.oracle import PropertyOracle
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = oracle.evaluate(ctx)
    print(result["homo_eV"])            # e.g. -6.82  (RF + F/P/S correction)
    print(result["lumo_eV"])            # e.g. -0.94  (RF + F/P/S correction)
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

from aurelius.constants import (
    CF3_CORRECTION_HOMO,
    CF3_CORRECTION_LUMO,
    F_CORRECTION_HOMO,
    F_CORRECTION_LUMO,
    FINGERPRINT_SIZE,
    MAX_DIELECTRIC_PER_TPSA,
    PHOSPHATE_CORRECTION_HOMO,
    PHOSPHATE_CORRECTION_LUMO,
    SULFONE_CORRECTION_HOMO,
    SULFONE_CORRECTION_LUMO,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_DATA_SOURCE: str = "pure fragment-additivity (Group Contribution — fallback mode)"

DEFAULT_RF_MODEL_PATH: str = "models/oracle_rf.joblib"


def get_data_source() -> str:
    """Return a human-readable string describing the oracle's data source."""
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# F/P/S SMARTS patterns for the inductive correction layer
# ---------------------------------------------------------------------------
# These patterns are detected on top of RF predictions to correct for the
# QM9 blindspot.  Each correction returns (delta_homo, delta_lumo).

_FPS_CORRECTIONS: list[tuple[str, str, float, float]] = [
    ("[F]",               "fluorine",          F_CORRECTION_HOMO,       F_CORRECTION_LUMO),
    ("[C](F)(F)F",        "trifluoromethyl",   CF3_CORRECTION_HOMO,     CF3_CORRECTION_LUMO),
    ("S(=O)(=O)",         "sulfone",           SULFONE_CORRECTION_HOMO, SULFONE_CORRECTION_LUMO),
    ("[PX4](=O)([OX2])",  "phosphate",         PHOSPHATE_CORRECTION_HOMO, PHOSPHATE_CORRECTION_LUMO),
]


def compute_fps_corrections(mol: Chem.Mol) -> tuple[float, float]:
    """Compute HOMO/LUMO inductive corrections for F/P/S groups absent from QM9.

    Args:
        mol: RDKit Mol object.

    Returns:
        (delta_homo, delta_lumo) in eV.
    """
    dh = 0.0
    dl = 0.0
    for smarts, _name, dh_i, dl_i in _FPS_CORRECTIONS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        n = len(mol.GetSubstructMatches(pattern))
        dh += n * dh_i
        dl += n * dl_i
    return dh, dl


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models
# ---------------------------------------------------------------------------

# (smarts, name, homo_shift, lumo_shift, dielectric_contrib, viscosity_contrib)
_GC_FRAGMENTS: list[tuple[str, str, float, float, float, float]] = [
    ("[CX3](=O)[OX2H0]",       "ester",              0.6, -0.4,  2.5,  0.6),
    ("[CX3](=O)[OH]",          "carboxylic_acid",    0.3, -0.6,  4.0,  1.0),
    ("[CX3](=O)[NX3]",         "amide",              0.8, -0.2,  5.0,  0.8),
    ("[CX3](=O)[CX3]",         "ketone",             0.7, -0.8,  3.0,  0.5),
    ("[CH](=O)",               "aldehyde",           0.5, -1.2,  2.5,  0.3),
    ("O=C([OX2])[OX2]",        "carbonate",          1.0, -0.5,  5.0,  0.7),
    ("[OD2]([CX4])[CX4]",      "ether",              0.4,  0.2,  1.5, -0.3),
    ("[OH][CX4]",              "alcohol",            0.3,  0.1,  4.5,  1.2),
    ("[NX3;H2][CX4]",          "primary_amine",      0.8,  0.5,  3.5,  0.5),
    ("[NX3;H1]([CX4])[CX4]",   "secondary_amine",    0.9,  0.4,  2.5,  0.4),
    ("[NX3;H0]([CX4])([CX4])[CX4]", "tertiary_amine", 0.5, 0.3,  1.5,  0.3),
    ("[C]#[N]",                "nitrile",           -0.3, -0.6,  8.0,  0.4),
    ("[CX3]=[CX3]",            "alkene",             0.8,  0.5,  0.5,  0.1),
    ("[CX2]#[CX2]",            "alkyne",             0.6, -0.3,  1.0,  0.2),
    ("[c]",                    "aromatic_carbon",    1.2,  1.0,  0.5,  0.5),
    ("[F]",                    "fluorine",          -0.15, -0.08, 0.0,  0.1),
    ("[Cl]",                   "chlorine",          -0.2, -0.15,  0.5,  0.2),
    ("[Br]",                   "bromine",           -0.15, -0.1,   0.5,  0.3),
    ("S(=O)(=O)[CX4]",         "sulfone",           -0.5, -1.2,  5.0,  0.5),
    ("S(=O)(=O)[OX2]",         "sulfonate",         -0.6, -1.0,  5.5,  0.6),
    ("S(=O)(=O)F",             "sulfonyl_fluoride", -0.7, -1.3,  4.0,  0.4),
    ("[PX4](=O)([OX2])([OX2])[OX2]", "phosphate",   -0.5, -1.0,  4.0,  0.8),
    ("[C](F)(F)F",             "trifluoromethyl",   -0.5, -0.3,  0.5,  0.2),
    ("[C](F)(F)",              "difluoromethylene", -0.3, -0.2,  0.3,  0.1),
    ("[BX3]([OX2])",           "boronate",          -0.4, -1.5,  2.0,  0.7),
    ("[S]([CX4])[CX4]",        "thioether",          0.2, -0.3,  1.0,  0.2),
]

_GC_BASE_HOMO: float = -9.2
_GC_BASE_LUMO: float = 2.8
_GC_BASE_DIELECTRIC: float = 1.9
_GC_BASE_VISCOSITY: float = 0.1


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each fragment SMARTS pattern in a molecule."""
    counts: dict[str, int] = {}
    for _smarts, name, _dh, _dl, _dd, _dv in _GC_FRAGMENTS:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(_smarts))
        counts[name] = len(matches)
    return counts


def predict_fragment_additivity(mol: Chem.Mol) -> tuple[float, float]:
    """Predict HOMO/LUMO using fragment-additivity (group contribution) model.

    This is the fallback when no RF model is available.
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
    """Predict a dielectric constant proxy via fragment-additivity + TPSA cap.

    The TPSA-based cap replaces the legacy sqrt anti-gaming: the maximum
    possible dielectric contribution is bounded by the molecule's polar
    surface area to prevent unrealistic stacking on small molecules.

    Returns:
        Unitless dielectric proxy (typically 1–15 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, _dh, _dl, dd, _dv in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dd

    from rdkit.Chem import rdMolDescriptors
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    value += tpsa * 0.02

    # TPSA-based cap: no molecule can exceed its polar-surface-area limit
    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)

    return max(1.0, value)


def predict_viscosity_proxy(mol: Chem.Mol) -> float:
    """Predict a viscosity proxy via fragment-additivity.

    Returns:
        Unitless viscosity proxy (typically 0.5–5.0 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_VISCOSITY
    for _smarts, _name, _dh, _dl, _dd, dv in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dv

    from rdkit.Chem import rdMolDescriptors
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    value += (mw - 30.0) * 0.005
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    value += n_rot * 0.15
    return max(0.1, value)


# ---------------------------------------------------------------------------
# RF Training
# ---------------------------------------------------------------------------


def train_oracle_rf(save_path: str = DEFAULT_RF_MODEL_PATH) -> str:
    """Train a RandomForest regressor for HOMO/LUMO on the bundled QM9 data.

    Uses ``MoleculeContext`` for featurization.  Saves via joblib.
    """
    from aurelius.data.loaders import load_qm9_homo_lumo_data

    logger.info("Loading QM9 HOMO/LUMO data...")
    data = load_qm9_homo_lumo_data()

    X_list: list[np.ndarray] = []
    y_homo: list[float] = []
    y_lumo: list[float] = []
    skipped = 0
    for smiles, homo, lumo in data:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            skipped += 1
            continue
        try:
            fp = ctx.get_feature_vector()
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


# ---------------------------------------------------------------------------
# PropertyOracle
# ---------------------------------------------------------------------------


class PropertyOracle:
    """Multi-objective property oracle using a hybrid RF + GC + F/P/S correction.

    Predicts four electrolyte-relevant properties:
      - HOMO energy (eV) — RF + F/P/S inductive correction or GC fallback
      - LUMO energy (eV) — RF + F/P/S inductive correction or GC fallback
      - Dielectric proxy — Group Contribution + TPSA-based cap
      - Viscosity proxy — Group Contribution

    **Hybrid workflow:**
    1. If a trained RF .joblib file is found, it is loaded for HOMO/LUMO.
    2. The F/P/S correction layer runs on every molecule, compensating for
       the QM9 blindspot on fluorine, phosphorus, and sulfur.
    3. If no RF file is found, falls back to GC for HOMO/LUMO (still with
       F/P/S corrections embedded in the GC fragment table).
    4. Dielectric and Viscosity always use GC with TPSA-based cap.
    """

    _CACHE: dict[str, dict[str, Any]] | None = None
    _rf_model: RandomForestRegressor | None = None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the PropertyOracle.

        Args:
            model_path: Path to a trained RF .joblib file. If None,
                defaults to DEFAULT_RF_MODEL_PATH.
        """
        global _DATA_SOURCE
        self._rf_model = None

        if model_path is None:
            model_path = DEFAULT_RF_MODEL_PATH

        if os.path.exists(model_path):
            try:
                self._rf_model = joblib.load(model_path)
                _DATA_SOURCE = (
                    "hybrid: RF (QM9-trained) + F/P/S correction for HOMO/LUMO + "
                    "fragment-additivity GC for dielectric/viscosity"
                )
                logger.info("RF model loaded from %s", model_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load RF model from %s: %s. Falling back to GC.",
                    model_path, exc,
                )
                self._rf_model = None
        else:
            logger.warning(
                "RF model not found at %s. Falling back to GC for HOMO/LUMO.",
                model_path,
            )

        if self._rf_model is None:
            _DATA_SOURCE = (
                "fragment-additivity (GC) + embedded F/P/S corrections"
            )

    def _predict_homo_lumo_rf(self, ctx: MoleculeContext) -> tuple[float, float]:
        """Predict HOMO/LUMO using RF + F/P/S inductive correction.

        The RF predicts from the QM9-trained model on ECFP4 + descriptors.
        The F/P/S correction layer then adds inductive shifts for any
        fluorine, sulfone, or phosphate groups present in the molecule
        that the RF cannot account for (QM9 blindspot).

        Returns:
            (homo_eV, lumo_eV) tuple.
        """
        fp = ctx.get_feature_vector()
        pred = self._rf_model.predict(fp.reshape(1, -1))[0]
        homo = float(pred[0])
        lumo = float(pred[1])

        dh, dl = compute_fps_corrections(ctx.mol)
        homo += dh
        lumo += dl

        homo = float(np.clip(homo, -15.0, 5.0))
        lumo = float(np.clip(lumo, -5.0, 10.0))
        return homo, lumo

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Evaluate a molecule and return predicted properties.

        Uses MoleculeContext (not SMILES) to enforce single-point parsing.
        Results are cached by SMILES string.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dictionary with keys:
              - homo_eV (float)
              - lumo_eV (float)
              - gap_eV (float)
              - dielectric_proxy (float)
              - viscosity_proxy (float)
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

        if self._rf_model is not None:
            try:
                homo, lumo = self._predict_homo_lumo_rf(ctx)
                domain_reason = (
                    "RF (QM9-trained) + F/P/S correction for HOMO/LUMO + "
                    "fragment-additivity GC for dielectric/viscosity"
                )
            except Exception:
                logger.warning(
                    "RF prediction failed for %s, falling back to GC.", smiles
                )
                homo, lumo = predict_fragment_additivity(ctx.mol)
                dh, dl = compute_fps_corrections(ctx.mol)
                homo += dh
                lumo += dl
                domain_reason = "fragment-additivity (GC) fallback + F/P/S correction for HOMO/LUMO"
        else:
            homo, lumo = predict_fragment_additivity(ctx.mol)
            domain_reason = "fragment-additivity (GC) model with embedded F/P/S corrections"

        gap = lumo - homo
        dielectric = predict_dielectric_proxy(ctx.mol)
        viscosity = predict_viscosity_proxy(ctx.mol)

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
        """Persist the cache to disk with joblib.

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
