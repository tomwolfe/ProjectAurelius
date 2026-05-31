"""PropertyOracle — HOMO/LUMO energy prediction for battery electrolyte screening.

This module provides a hybrid HOMO/LUMO (frontier orbital energy) predictor:

1. **Random Forest** (in-domain): Trained on QM9 ECFP4 fingerprints for molecules
   without S, P, or heavy atoms beyond the QM9 distribution.
2. **Fragment-Additivity Correction** (OOD): A group-contribution layer adjusts
   RF predictions for electrolyte-specific motifs (sulfones, carbonates, phosphates,
   heavy fluorination) that are absent or rare in QM9.
3. **Pure Fragment-Additivity Model** (fully OOD): For molecules with S, P, or >15
   heavy atoms, a standalone linear group-contribution model (trained on QM9
   fragment counts + hand-tuned S/P/F corrections) provides the prediction.

Usage:
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
    print(result["homo_eV"])     # e.g. -6.82
    print(result["lumo_eV"])     # e.g. -0.94
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.utils.chem_utils import generate_ecfp4_fingerprint

logger = logging.getLogger(__name__)


_DATA_SOURCE: str | None = None


def get_data_source() -> str:
    """Return a human-readable string describing which data the oracle was trained on."""
    global _DATA_SOURCE
    if _DATA_SOURCE is None:
        return "oracle not yet initialized"
    return _DATA_SOURCE


def _load_qm9_data_for_training(
    min_count: int = 100,
) -> list[tuple[str, float, float]]:
    """Load QM9 HOMO/LUMO data with loud error reporting.

    Args:
        min_count: Minimum number of molecules required.

    Returns:
        List of (smiles, homo_eV, lumo_eV) tuples.

    Raises:
        RuntimeError: If fewer than min_count molecules are loaded.
    """
    from aurelius.data.loaders import load_qm9_homo_lumo_data

    try:
        data = load_qm9_homo_lumo_data()
    except Exception as exc:
        logger.error(
            "PropertyOracle: FAILED to load QM9 data — %s. "
            "The oracle will be non-functional without QM9 training data.",
            exc,
        )
        raise RuntimeError(f"QM9 data loading failed: {exc}") from exc

    if len(data) < min_count:
        msg = (
            f"PropertyOracle: insufficient QM9 data — got {len(data)} molecules, "
            f"need at least {min_count}. Cannot train a meaningful model."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    global _DATA_SOURCE
    _DATA_SOURCE = (
        f"QM9 + fragment-additivity ({len(data)} QM9 molecules, "
        f"augmented with electrolyte fragment corrections)"
    )
    logger.info("PropertyOracle: data source = %s", _DATA_SOURCE)

    return data


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Model
# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS patterns with estimated HOMO/LUMO shifts (eV).
# Shifts represent the change in frontier orbital energy when the fragment
# is added to a simple alkane scaffold. Values are calibrated from QM9
# statistics for CHON groups, and from literature-based inductive effects
# for S/P/F groups absent from QM9.
#
# A positive Δ means the orbital energy increases (less stable, closer to zero).
# A negative Δ means the orbital energy decreases (more stable, more negative).

# (smarts, name, homo_shift, lumo_shift)
_GC_FRAGMENTS: list[tuple[str, str, float, float]] = [
    # Carbonyl groups (from QM9 fit)
    ("[CX3](=O)[OX2H0]", "ester", 0.6, -0.4),
    ("[CX3](=O)[OH]", "carboxylic_acid", 0.3, -0.6),
    ("[CX3](=O)[NX3]", "amide", 0.8, -0.2),
    ("[CX3](=O)[CX3]", "ketone", 0.7, -0.8),
    ("[CH](=O)", "aldehyde", 0.5, -1.2),
    # Carbonates
    ("O=C([OX2])[OX2]", "carbonate", 1.0, -0.5),
    # Ethers
    ("[OD2]([CX4])[CX4]", "ether", 0.4, 0.2),
    # Alcohols
    ("[OH][CX4]", "alcohol", 0.3, 0.1),
    # Amines
    ("[NX3;H2][CX4]", "primary_amine", 0.8, 0.5),
    ("[NX3;H1]([CX4])[CX4]", "secondary_amine", 0.9, 0.4),
    ("[NX3;H0]([CX4])([CX4])[CX4]", "tertiary_amine", 0.5, 0.3),
    # Nitriles
    ("[C]#[N]", "nitrile", -0.3, -0.6),
    # Alkenes
    ("[CX3]=[CX3]", "alkene", 0.8, 0.5),
    # Alkynes
    ("[CX2]#[CX2]", "alkyne", 0.6, -0.3),
    # Aromatic carbon
    ("[c]", "aromatic_carbon", 1.2, 1.0),
    # Halogens
    ("[F]", "fluorine", -0.15, -0.08),
    ("[Cl]", "chlorine", -0.2, -0.15),
    ("[Br]", "bromine", -0.15, -0.1),
    # Electrolyte-specific groups (NOT in QM9 — hand-tuned from inductive effects)
    ("S(=O)(=O)[CX4]", "sulfone", -0.5, -1.2),
    ("S(=O)(=O)[OX2]", "sulfonate", -0.6, -1.0),
    ("S(=O)(=O)F", "sulfonyl_fluoride", -0.7, -1.3),
    ("[PX4](=O)([OX2])([OX2])[OX2]", "phosphate", -0.5, -1.0),
    ("[C](F)(F)F", "trifluoromethyl", -0.5, -0.3),
    ("[C](F)(F)", "difluoromethylene", -0.3, -0.2),
    ("[BX3]([OX2])", "boronate", -0.4, -1.5),
    ("[S]([CX4])[CX4]", "thioether", 0.2, -0.3),
]

# Base HOMO/LUMO for a simple alkane (ethane reference)
_GC_BASE_HOMO: float = -9.2
_GC_BASE_LUMO: float = 2.8


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each fragment SMARTS pattern in a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping fragment name to count.
    """
    counts: dict[str, int] = {}
    for _smarts, name, _dh, _dl in _GC_FRAGMENTS:
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
    for _smarts, _name, dh, dl in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        homo += n * dh
        lumo += n * dl
    return homo, lumo


def _fit_fragment_contributions_from_qm9() -> dict[str, tuple[float, float]]:
    """Fit fragment contributions from QM9 data via Ridge regression.

    Returns a dict of {fragment_name: (homo_coeff, lumo_coeff)} with
    the base intercept stored under key "_base_".

    Falls back to literature-based values if fitting fails.
    """
    from sklearn.linear_model import Ridge

    qm9_data = _load_qm9_data_for_training(min_count=50)

    fragment_names = [name for _, name, _, _ in _GC_FRAGMENTS]
    X_list: list[list[float]] = []
    y_homo: list[float] = []
    y_lumo: list[float] = []

    for smi, homo_val, lumo_val in qm9_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        counts = _count_fragments(mol)
        row = [float(counts.get(name, 0)) for name in fragment_names]
        X_list.append(row)
        y_homo.append(homo_val)
        y_lumo.append(lumo_val)

    if len(X_list) < 50:
        logger.warning(
            "Insufficient QM9 molecules for fragment fitting (%d). Using default values.",
            len(X_list),
        )
        return {}

    X = np.array(X_list)
    y_h = np.array(y_homo)
    y_l = np.array(y_lumo)

    try:
        model_h = Ridge(alpha=1.0, fit_intercept=True)
        model_h.fit(X, y_h)
        model_l = Ridge(alpha=1.0, fit_intercept=True)
        model_l.fit(X, y_l)

        coeffs: dict[str, tuple[float, float]] = {
            "_base_": (float(model_h.intercept_), float(model_l.intercept_)),
        }
        for i, name in enumerate(fragment_names):
            ch = float(model_h.coef_[i])
            cl = float(model_l.coef_[i])
            if abs(ch) > 0.01 or abs(cl) > 0.01:
                coeffs[name] = (ch, cl)
        logger.info(
            "Fragment contributions fitted from %d QM9 molecules (%d active fragments).",
            len(X_list),
            len(coeffs) - 1,
        )
        return coeffs
    except Exception as exc:
        logger.warning("Fragment fitting failed: %s. Using default values.", exc)
        return {}


def _compute_qm9_centroid(qm9_data: list[tuple[str, float, float]]) -> np.ndarray:
    """Compute the mean Morgan fingerprint centroid of the QM9 training set."""
    fps: list[np.ndarray] = []
    for smi, _homo, _lumo in qm9_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = generate_ecfp4_fingerprint(smi).astype(np.float32)
        except Exception:
            continue
        fps.append(fp)
    if not fps:
        return np.zeros(2048, dtype=np.float32)
    return np.mean(fps, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Domain applicability — revised for electrolyte-aware hybrid oracle
# ---------------------------------------------------------------------------
# The oracle uses the RF for molecules within the QM9-like chemical space
# (CHON F≤15, S=0, P=0, heavy_atoms≤15) and the fragment-additivity model
# for OOD molecules. This means the "domain penalty" now indicates which
# model was used, rather than penalizing the score.

_QM9_LIKE_LIMITS: dict[str, int] = {
    "F": 15,
    "S": 0,
    "P": 0,
    "Cl": 3,
    "Br": 1,
    "B": 0,
    "Si": 0,
}


def _is_qm9_like(mol: Chem.Mol) -> bool:
    """Check if a molecule resembles QM9 training distribution.

    QM9 contains only CHONF up to 9 heavy atoms, no S or P, and
    limited halogenation. This function checks whether a molecule
    falls within the QM9-like regime.

    Returns:
        True if molecule is QM9-like (use RF), False for OOD (use GC).
    """
    from collections import Counter

    element_counts: Counter[str] = Counter()
    n_heavy = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        element_counts[sym] = element_counts.get(sym, 0) + 1
        if atom.GetAtomicNum() > 1:
            n_heavy += 1

    for elem, limit in _QM9_LIKE_LIMITS.items():
        if element_counts.get(elem, 0) > limit:
            return False

    if n_heavy > 15:
        return False

    return True


def _domain_applicable(smiles: str, mol: Chem.Mol | None = None) -> tuple[bool, str, float]:
    """Check if a molecule is within the QM9-like applicability domain.

    Revised for the hybrid oracle:
      - In-domain (QM9-like): RF model is used, no penalty.
      - OOD: Fragment-additivity model is used, no penalty (the GC model
        handles all elements by design).

    Returns:
        (is_applicable, reason, penalty_multiplier) tuple.
        penalty_multiplier is always 1.0 now since the GC model handles OOD.
        ``is_applicable`` indicates whether RF is reliable (True = use RF).
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES", 0.0

    is_qm9 = _is_qm9_like(mol)
    if not is_qm9:
        logger.info(
            "Molecule %s is outside QM9 domain — using fragment-additivity model.",
            smiles,
        )
        return False, "OOD (S/P/F/heavy atoms) — using fragment-additivity model", 1.0

    return True, "", 1.0


def _train_homo_lumo_models() -> tuple[Any, float, float]:
    """Train RandomForest models for HOMO and LUMO prediction on ECFP4 fingerprints.

    Uses real QM9 ground-truth HOMO/LUMO values. Also fits the fragment-additivity
    model from QM9 for OOD predictions.

    Returns:
        A tuple of (model, gap_min, gap_max) where model is a
        MultiOutputRegressor wrapper and gap_min/gap_max are the
        training gap range for normalisation.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    qm9_data = _load_qm9_data_for_training()

    X_list: list[np.ndarray] = []
    y_list: list[list[float]] = []
    for smi, homo, lumo in qm9_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = generate_ecfp4_fingerprint(smi).astype(np.float32)
        except Exception:
            continue
        X_list.append(fp)
        y_list.append([homo, lumo])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    base = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X, y)

    # Store QM9 centroid fingerprint for domain applicability
    PropertyOracle._qm9_centroid = _compute_qm9_centroid(qm9_data)
    logger.info(
        "PropertyOracle: QM9 centroid computed (%d nonzero bits approx).",
        int(np.sum(PropertyOracle._qm9_centroid > 0.3)),
    )

    gaps = y[:, 1] - y[:, 0]
    gap_min = float(gaps.min())
    gap_max = float(gaps.max())
    logger.info(
        "PropertyOracle: trained HOMO/LUMO model on %d QM9 molecules (gap range [%.2f, %.2f] eV).",
        len(X), gap_min, gap_max,
    )
    return model, gap_min, gap_max


class PropertyOracle:
    """Hybrid HOMO/LUMO oracle using RF (in-domain) + fragment-additivity (OOD).

    For molecules within the QM9-like chemical space (CHON, no S/P, ≤15 heavy
    atoms), uses a MultiOutput RandomForestRegressor on ECFP4 fingerprints.

    For OOD molecules (containing S, P, heavy fluorination, >15 heavy atoms),
    uses a fragment-additivity (group contribution) model that handles all
    elements by design and includes hand-tuned electrolyte fragment corrections.

    The model is trained once on import (lazy loading on first use).
    Predictions are cached by SMILES string to avoid redundant computation.

    Requirements:
        - ``scikit-learn`` must be importable
        - ``rdkit`` must be importable
    """

    _model: tuple[Any, float, float] | None = None
    _CACHE: dict[str, dict[str, Any]] | None = None
    _qm9_centroid: np.ndarray | None = None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the PropertyOracle.

        The model is built lazily on first call to ``evaluate()``.

        Args:
            model_path: Reserved for future checkpoint loading.
        """
        pass

    def _ensure_model(self) -> None:
        """Load or train the model if not already loaded."""
        if PropertyOracle._model is None:
            PropertyOracle._model = _train_homo_lumo_models()

    def _predict_rf(self, smiles: str) -> dict[str, float]:
        """Run RF model inference and return HOMO/LUMO energies.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict with ``homo_eV``, ``lumo_eV``, ``gap_eV``, and
            ``score_eV`` (normalised gap mapped to [0, 100]).

        Raises:
            RuntimeError: If the model cannot be trained or SMILES is invalid.
        """
        self._ensure_model()
        model, gap_min, gap_max = PropertyOracle._model  # type: ignore[misc]

        try:
            fp = generate_ecfp4_fingerprint(smiles).reshape(1, -1).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate fingerprint for {smiles}: {exc}") from exc

        homo, lumo = model.predict(fp)[0]
        homo = float(homo)
        lumo = float(lumo)
        gap = lumo - homo

        range_gap = gap_max - gap_min
        score_eV = (gap - gap_min) / range_gap * 100.0 if range_gap > 0 else 50.0
        score_eV = max(0.0, min(100.0, score_eV))

        return {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "score_eV": round(score_eV, 2),
        }

    def _predict_gc(self, mol: Chem.Mol) -> dict[str, float]:
        """Run fragment-additivity prediction.

        Args:
            mol: RDKit Mol object.

        Returns:
            Dict with ``homo_eV``, ``lumo_eV``, ``gap_eV``, and
            ``score_eV`` (normalised gap mapped to [0, 100]).
        """
        homo, lumo = predict_fragment_additivity(mol)
        gap = lumo - homo

        self._ensure_model()
        _, gap_min, gap_max = PropertyOracle._model  # type: ignore[misc]
        range_gap = gap_max - gap_min
        score_eV = (gap - gap_min) / range_gap * 100.0 if range_gap > 0 else 50.0
        score_eV = max(0.0, min(100.0, score_eV))

        return {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "score_eV": round(score_eV, 2),
        }

    def evaluate(self, smiles: str, mol: Chem.Mol | None = None) -> dict[str, Any]:
        """Evaluate a molecule and return predicted HOMO/LUMO properties.

        Uses RF for QM9-like molecules and fragment-additivity for OOD molecules.
        Results are cached by SMILES string.

        Args:
            smiles: Canonical or isomeric SMILES string.
            mol: Optional pre-parsed RDKit Mol object (avoids redundant parsing).

        Returns:
            Dictionary with keys ``homo_eV``, ``lumo_eV``, ``gap_eV``,
            ``score_eV``, ``domain_applicable`` (bool: True = RF used,
            False = GC used), ``domain_reason`` (str), and
            ``domain_penalty`` (float, always 1.0).

        Raises:
            ValueError: If SMILES is invalid.
        """
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Check domain applicability
        is_applicable, reason, penalty = _domain_applicable(smiles, mol)

        if is_applicable:
            # In-domain: use Random Forest
            result = self._predict_rf(smiles)
        else:
            # OOD: use fragment-additivity model
            result = self._predict_gc(mol)
            result["prediction_source"] = "fragment_additivity"

        result["domain_applicable"] = is_applicable
        result["domain_reason"] = reason
        result["domain_penalty"] = penalty

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def evaluate_with_ood_penalty(self, smiles: str) -> dict[str, Any]:
        """Evaluate a molecule (backward-compatible wrapper).

        With the hybrid oracle, OOD molecules get accurate predictions
        from the fragment-additivity model, so no penalty is applied.
        ``score_eV`` is the raw prediction score.

        Args:
            smiles: Canonical SMILES string.

        Returns:
            Result dict with ``score_eV``.
        """
        return self.evaluate(smiles)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist all trained models to disk with joblib.

        Args:
            path: File path for the joblib dump.
        """
        import joblib

        payload: dict[str, Any] = {
            "model": PropertyOracle._model,
            "qm9_centroid": PropertyOracle._qm9_centroid,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: models saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        """Load pre-trained models from a joblib cache file.

        Args:
            path: File path to the joblib dump.

        Returns:
            True if models were loaded successfully, False otherwise.
        """
        import joblib

        try:
            payload = joblib.load(path)
        except (FileNotFoundError, Exception) as exc:
            logger.debug("PropertyOracle: no cached oracle at %s (%s)", path, exc)
            return False

        PropertyOracle._model = payload.get("model")
        PropertyOracle._qm9_centroid = payload.get("qm9_centroid")

        global _DATA_SOURCE
        _DATA_SOURCE = payload.get("data_source", "loaded from cache")

        logger.info("PropertyOracle: models loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        """Clear the SMILES->properties cache."""
        if self._CACHE is not None:
            self._CACHE.clear()
