"""SurrogateQuantumOracle — Lightweight ML pre-filter for EA candidates.

Trains a scikit-learn RandomForestRegressor on ECFP4 fingerprints to predict
HOMO/LUMO energies. Used as a "Tier 0.5" filter: if surrogate predicts
HOMO > -5.0 eV (highly unstable), apply a 0.5x multiplicative penalty
to the score before invoking the full xTB/TOM oracle, saving compute.

Training is lazy (first inference triggers training) and uses only
scikit-learn — no deep learning frameworks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_SURROGATE_HOMO_THRESHOLD: float = -5.0
_SURROGATE_PENALTY: float = 0.5
_TRAINING_TIME_LIMIT: float = 2.0


class SurrogateQuantumOracle:
    """RandomForest surrogate for fast HOMO/LUMO estimation.

    Trained lazily on orbital_calibration.json. Training takes < 2s
    for the calibration set (~45 molecules). Inference is < 1ms per molecule.

    Usage:
        surrogate = SurrogateQuantumOracle()
        homo, lumo = surrogate.predict(ctx)
        penalty = surrogate.compute_penalty(homo)  # 0.5x if HOMO > -5.0
    """

    def __init__(
        self,
        calibration_path: str | None = None,
        n_estimators: int = 50,
        max_depth: int = 8,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._calibration_path = calibration_path
        self._data_override: list[dict[str, Any]] | None = None

        self._homo_model: RandomForestRegressor | None = None
        self._lumo_model: RandomForestRegressor | None = None
        self._is_trained = False
        self._train_time_ms: float = 0.0
        self._n_train: int = 0

    def set_training_data(self, data: list[dict[str, Any]]) -> None:
        """Override training data (used for holdout validation)."""
        self._data_override = data

    def _resolve_calibration_path(self) -> str:
        if self._calibration_path is not None:
            return self._calibration_path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(
            module_dir, "..", "..", "..", "..", "src", "aurelius", "data",
            "orbital_calibration.json",
        )

    def _load_data(self) -> list[dict[str, Any]]:
        if self._data_override is not None:
            return self._data_override
        path = self._resolve_calibration_path()
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            alt = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "data", "orbital_calibration.json",
            )
            alt_resolved = os.path.abspath(alt)
            if os.path.exists(alt_resolved):
                resolved = alt_resolved
            else:
                raise FileNotFoundError(
                    f"Orbital calibration not found at {resolved} or {alt_resolved}"
                )
        with open(resolved) as f:
            return json.load(f)

    def _fingerprint_array(self, mol: Chem.Mol) -> np.ndarray:
        from rdkit.Chem import AllChem
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr

    def _ensure_trained(self) -> None:
        if self._is_trained:
            return
        t0 = time.perf_counter()
        data = self._load_data()

        X_list: list[np.ndarray] = []
        y_homo: list[float] = []
        y_lumo: list[float] = []

        for entry in data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            fp_arr = self._fingerprint_array(mol)
            X_list.append(fp_arr)
            y_homo.append(entry["homo_eV"])
            y_lumo.append(entry["lumo_eV"])

        if len(X_list) < 5:
            raise ValueError(
                f"Surrogate training requires >= 5 molecules, got {len(X_list)}"
            )

        X = np.array(X_list, dtype=np.float32)
        y_homo_arr = np.array(y_homo, dtype=np.float32)
        y_lumo_arr = np.array(y_lumo, dtype=np.float32)

        self._homo_model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
            n_jobs=1,
        )
        self._lumo_model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state + 1,
            n_jobs=1,
        )

        self._homo_model.fit(X, y_homo_arr)
        self._lumo_model.fit(X, y_lumo_arr)

        self._is_trained = True
        self._n_train = len(X_list)
        self._train_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SurrogateQuantumOracle: trained on %d molecules in %.1fms",
            self._n_train, self._train_time_ms,
        )

    def predict(self, ctx: MoleculeContext) -> tuple[float, float]:
        """Predict (homo_eV, lumo_eV) using the trained surrogate.

        Training happens lazily on first call. Returns both values.
        """
        self._ensure_trained()
        fp_arr = self._fingerprint_array(ctx.mol).reshape(1, -1)

        t0 = time.perf_counter()
        homo = float(self._homo_model.predict(fp_arr)[0])  # type: ignore[union-attr]
        lumo = float(self._lumo_model.predict(fp_arr)[0])  # type: ignore[union-attr]
        _inference_ms = (time.perf_counter() - t0) * 1000

        return homo, lumo

    def compute_penalty(self, homo_eV: float) -> float:
        """Return 0.5x penalty if surrogate predicts unstable HOMO (> -5.0 eV)."""
        if homo_eV > _SURROGATE_HOMO_THRESHOLD:
            return _SURROGATE_PENALTY
        return 1.0

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def train_time_ms(self) -> float:
        return self._train_time_ms

    @property
    def n_train(self) -> int:
        return self._n_train
