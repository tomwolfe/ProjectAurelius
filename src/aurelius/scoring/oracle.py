"""PropertyOracle — logS solubility prediction for battery electrolyte screening.

This module provides a QSPR-based logS (log aqueous solubility) predictor
using a scikit-learn RandomForestRegressor trained on RDKit ECFP4 fingerprints
and ESOL dataset solubility values.  logS is a physically meaningful property
for electrolyte screening — it separates good electrolyte candidates from poor
ones and is directly relevant to SEI formation and electrolyte formulation.

The Oracle returns raw logS and a normalized score_eV in [0, 100].

Usage:
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
    print(result["logS"])        # e.g. -1.74
    print(result["score_eV"])    # e.g. 60.2
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def _load_esol_data() -> tuple[list[str], np.ndarray]:
    """Load and deduplicate ESOL solubility data from the bundled CSV.

    Returns:
        (smiles_list, logS_array) with unique SMILES entries.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "esol_fallback.csv")
    csv_path = os.path.abspath(csv_path)

    seen: dict[str, float] = {}
    with open(csv_path) as f:
        header = next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            smi, logS_str = parts
            try:
                logS = float(logS_str)
            except ValueError:
                continue
            seen[smi] = logS

    smiles_list = list(seen.keys())
    logS_array = np.array([seen[s] for s in smiles_list], dtype=np.float32)
    return smiles_list, logS_array


def _generate_ecfp4(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Generate an ECFP4 (Morgan radius=2) fingerprint from SMILES.

    Args:
        smiles: SMILES string of the molecule.
        n_bits: Fingerprint size (default 2048).

    Returns:
        numpy float32 array of shape (n_bits,) with values 0.0 or 1.0.

    Raises:
        RuntimeError: If RDKit fails to parse SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"RDKit failed to parse SMILES '{smiles}'. Invalid molecule structure.",
        )
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    bit_list = fp.ToList()
    arr = np.array(bit_list, dtype=np.float32)
    if len(arr) < n_bits:
        padded = np.zeros(n_bits, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded
    return arr[:n_bits]


def _train_qsp_model() -> Any:
    """Train and return a RandomForestRegressor on ESOL logS data.

    The model learns the mapping from ECFP4 fingerprints to logS
    (log aqueous solubility).  This is a standard QSPR approach.

    Returns:
        A tuple of (fitted RandomForestRegressor, min_logS, max_logS)
        for normalizing predictions.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for QSPR property prediction. "
            "Install it with: pip install scikit-learn"
        ) from exc

    smiles_list, logS_values = _load_esol_data()
    X = np.array([_generate_ecfp4(s) for s in smiles_list])

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, logS_values)

    min_logS = float(logS_values.min())
    max_logS = float(logS_values.max())
    logger.info(
        "PropertyOracle: trained logS model on %d unique ESOL molecules (range [%.2f, %.2f]).",
        len(smiles_list), min_logS, max_logS,
    )
    return (model, min_logS, max_logS)


class PropertyOracle:
    """QSPR-based oracle for logS solubility prediction.

    Uses a scikit-learn RandomForestRegressor trained on RDKit ECFP4
    fingerprints and ESOL solubility values.  Returns raw logS and a
    normalized score_eV in [0, 100].

    The model is trained once on import (lazy loading on first use).
    Predictions are cached by SMILES string to avoid redundant computation.

    Requirements:
        - ``scikit-learn`` must be importable
        - ``rdkit`` must be importable

    Example:
        >>> oracle = PropertyOracle()
        >>> result = oracle.evaluate("CC(=O)OC1=CC=CC=C1")
        >>> result["logS"]
        -1.74
    """

    _model: tuple[Any, float, float] | None = None
    _CACHE: dict[str, dict[str, float]] | None = None

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise the PropertyOracle.

        The QSPR model is built lazily on first call to ``evaluate()``.

        Args:
            model_path: Reserved for future checkpoint loading.
        """
        pass

    def _ensure_model(self) -> None:
        """Load or train the QSPR model if not already loaded."""
        if PropertyOracle._model is None:
            PropertyOracle._model = _train_qsp_model()

    def _predict(self, smiles: str) -> dict[str, float]:
        """Run QSPR model inference and return logS and normalized score.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict with ``logS`` (raw prediction) and ``score_eV``
            (normalised to [0, 100]).

        Raises:
            RuntimeError: If the QSPR model cannot be trained or SMILES
                is invalid.
        """
        self._ensure_model()
        model, min_logS, max_logS = PropertyOracle._model

        try:
            fingerprint = _generate_ecfp4(smiles)
        except Exception as exc:
            raise RuntimeError(f"Failed to generate fingerprint for {smiles}: {exc}") from exc

        logS = float(model.predict(fingerprint.reshape(1, -1))[0])

        range_logS = max_logS - min_logS
        if range_logS > 0:
            score_eV = (logS - min_logS) / range_logS * 100.0
        else:
            score_eV = 50.0
        score_eV = max(0.0, min(100.0, score_eV))

        return {
            "logS": round(logS, 4),
            "score_eV": round(score_eV, 2),
        }

    def evaluate(self, smiles: str) -> dict[str, float]:
        """Evaluate a molecule and return predicted solubility properties.

        Results are cached by SMILES string to avoid redundant computation.

        Args:
            smiles: Canonical or isomeric SMILES string.

        Returns:
            Dictionary with keys ``logS`` and ``score_eV``.

        Raises:
            ValueError: If SMILES is invalid.
        """
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        result = self._predict(smiles)

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def clear_cache(self) -> None:
        """Clear the SMILES→properties cache."""
        if self._CACHE is not None:
            self._CACHE.clear()
