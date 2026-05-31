"""PropertyOracle — HOMO/LUMO energy prediction for battery electrolyte screening.

This module provides a QSPR-based HOMO/LUMO (frontier orbital energy) predictor
using a scikit-learn RandomForestRegressor trained on RDKit ECFP4 fingerprints
with real QM9 ground-truth data.

HOMO/LUMO energies and their gap are the primary determinants of electrochemical
stability for battery electrolyte molecules.

The Oracle returns HOMO, LUMO, gap (eV) and a normalized score in [0, 100].

Usage:
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
    print(result["homo_eV"])     # e.g. -6.82
    print(result["lumo_eV"])     # e.g. -0.94
    print(result["gap_eV"])      # e.g. 5.88
    print(result["score_eV"])    # e.g. 62.3
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

    # Log data source
    global _DATA_SOURCE
    _DATA_SOURCE = f"QM9 real data ({len(data)} molecules, HuggingFace / bundled fallback)"
    logger.info("PropertyOracle: data source = %s", _DATA_SOURCE)

    return data


def _train_lumo_rf() -> tuple[Any, float, float]:
    """Train RF for LUMO prediction on ECFP4 fingerprints using real QM9 data."""
    from sklearn.ensemble import RandomForestRegressor

    qm9_data = _load_qm9_data_for_training()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for smi, _homo, lumo in qm9_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = generate_ecfp4_fingerprint(smi)
        except Exception:
            continue
        X_list.append(fp)
        y_list.append(lumo)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    lumo_min = float(y.min())
    lumo_max = float(y.max())
    logger.info(
        "PropertyOracle: trained LUMO RF on %d QM9 molecules (range [%.4f, %.4f] eV).",
        len(X), lumo_min, lumo_max,
    )
    return rf, lumo_min, lumo_max


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


def _domain_applicable(smiles: str) -> tuple[bool, str]:
    """Check if a molecule is Out-Of-Distribution relative to the QM9 training set.

    Uses Tanimoto similarity to the QM9 centroid fingerprint to flag OOD molecules.
    A soft penalty is applied later based on this check.

    Returns:
        (is_applicable, reason) tuple.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    # Check Tanimoto similarity to QM9 centroid
    centroid = getattr(PropertyOracle, "_qm9_centroid", None)
    if centroid is not None:
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import TanimotoSimilarity
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # Convert centroid to ExplicitBitVect for similarity
        centroid_bv = ExplicitBitVect(2048)
        for i, v in enumerate(centroid):
            if v > 0.3:
                centroid_bv.SetBit(i)
        sim = TanimotoSimilarity(fp, centroid_bv)
        if sim < 0.3:
            return False, f"Low Tanimoto similarity ({sim:.3f}) to QM9 centroid — out of distribution"

    return True, ""


def _train_homo_lumo_models() -> tuple[Any, float, float]:
    """Train RandomForest models for HOMO and LUMO prediction on ECFP4 fingerprints.

    Uses real QM9 ground-truth HOMO/LUMO values.

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
    """QSPR-based oracle for HOMO/LUMO frontier orbital energy prediction.

    Uses a scikit-learn MultiOutput RandomForestRegressor trained on
    RDKit molecular descriptors to predict HOMO and LUMO energies in eV.
    The model is trained once on import (lazy loading on first use).
    Predictions are cached by SMILES string to avoid redundant computation.

    Requirements:
        - ``scikit-learn`` must be importable
        - ``rdkit`` must be importable

    Example:
        >>> oracle = PropertyOracle()
        >>> result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
        >>> result["homo_eV"]
        -6.82
    """

    _model: tuple[Any, float, float] | None = None
    _CACHE: dict[str, dict[str, Any]] | None = None
    _lumo_rf: tuple[Any, float, float] | None = None
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

    def _ensure_lumo_model(self) -> None:
        """Load or train the LUMO RF model if not already loaded."""
        if PropertyOracle._lumo_rf is None:
            PropertyOracle._lumo_rf = _train_lumo_rf()

    def predict_normalized_lumo(self, smiles: str) -> float:
        """Predict normalized LUMO score in [0, 100] from ECFP4 fingerprints.

        Uses a fixed LUMO range of [-3.0, 2.0] eV corresponding to the
        span of typical organic molecules in QM9.  Higher score = more
        positive LUMO = better reductive stability.
        """
        self._ensure_lumo_model()
        rf, lumo_min, lumo_max = PropertyOracle._lumo_rf  # type: ignore[misc]
        fp = generate_ecfp4_fingerprint(smiles).reshape(1, -1)
        lumo = float(rf.predict(fp)[0])
        # Use the theoretical QM9 range for meaningful battery scoring
        normalized = (lumo - (-3.0)) / (2.0 - (-3.0)) * 100.0
        normalized = np.clip(normalized, 0.0, 100.0)
        return round(float(normalized), 2)

    def _predict(self, smiles: str) -> dict[str, float]:
        """Run model inference and return HOMO/LUMO energies.

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

    def evaluate(self, smiles: str) -> dict[str, Any]:
        """Evaluate a molecule and return predicted HOMO/LUMO properties.

        Results are cached by SMILES string to avoid redundant computation.
        Also checks QM9 domain applicability.

        Args:
            smiles: Canonical or isomeric SMILES string.

        Returns:
            Dictionary with keys ``homo_eV``, ``lumo_eV``, ``gap_eV``,
            ``score_eV``, ``domain_applicable`` (bool), and
            ``domain_reason`` (str).

        Raises:
            ValueError: If SMILES is invalid.
        """
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        result: dict[str, Any] = self._predict(smiles)

        # Domain applicability
        is_applicable, reason = _domain_applicable(smiles)
        result["domain_applicable"] = is_applicable
        result["domain_reason"] = reason

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist all trained models to disk with joblib.

        Args:
            path: File path for the joblib dump.
        """
        import joblib

        payload: dict[str, Any] = {
            "model": PropertyOracle._model,
            "lumo_rf": PropertyOracle._lumo_rf,
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
        PropertyOracle._lumo_rf = payload.get("lumo_rf")
        PropertyOracle._qm9_centroid = payload.get("qm9_centroid")

        global _DATA_SOURCE
        _DATA_SOURCE = payload.get("data_source", "loaded from cache")

        logger.info("PropertyOracle: models loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        """Clear the SMILES->properties cache."""
        if self._CACHE is not None:
            self._CACHE.clear()
