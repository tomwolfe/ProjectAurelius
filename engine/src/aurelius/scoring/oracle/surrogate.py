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
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    x_rank = np.empty(n, dtype=np.float64)
    y_rank = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_rank[i] = 1 + np.sum(x < x[i]) + (np.sum(x == x[i]) - 1) / 2
        y_rank[i] = 1 + np.sum(y < y[i]) + (np.sum(y == y[i]) - 1) / 2
    x_diff = x_rank - np.mean(x_rank)
    y_diff = y_rank - np.mean(y_rank)
    denom = np.sqrt(np.sum(x_diff ** 2)) * np.sqrt(np.sum(y_diff ** 2))
    rho = float(np.sum(x_diff * y_diff) / denom) if denom > 0 else 0.0
    return rho, 0.0

_SURROGATE_HOMO_THRESHOLD: float = -5.0
_SURROGATE_PENALTY: float = 0.5
_TRAINING_TIME_LIMIT: float = 2.0


class SurrogateQuantumOracle:
    """RandomForest surrogate for fast HOMO/LUMO estimation.

    Trained lazily on orbital_calibration.json. Training takes < 2s
    for the calibration set (~200 molecules). Inference is < 1ms per molecule.

    Uses a 2060-dimensional feature vector:
      - [0:2048]  ECFP4 binary fingerprint (Morgan radius=2, 2048 bits)
      - [2048]    Molecular weight
      - [2049]    MolLogP
      - [2050]    TPSA
      - [2051]    Ring count
      - [2052]    Rotatable bonds
      - [2053]    Num F atoms
      - [2054]    Num O atoms
      - [2055]    Num N atoms
      - [2056]    Num S atoms
      - [2057]    Num aromatic rings
      - [2058]    Fraction sp3
      - [2059]    Num H-bond acceptors

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

    def _build_feature_vector(self, ctx: MoleculeContext) -> np.ndarray:
        """Build 2060-dim feature vector from MoleculeContext.

        Layout:
          - [0:2048]  ECFP4 binary fingerprint (Morgan radius=2, 2048 bits)
          - [2048]    Molecular weight
          - [2049]    MolLogP
          - [2050]    TPSA
          - [2051]    Ring count
          - [2052]    Rotatable bonds
          - [2053]    Num F atoms
          - [2054]    Num O atoms
          - [2055]    Num N atoms
          - [2056]    Num S atoms
          - [2057]    Num aromatic rings
          - [2058]    Fraction sp3
          - [2059]    Num H-bond acceptors
        """
        mol = ctx.mol
        fp_arr = self._fingerprint_array(mol)
        arr = np.zeros(2060, dtype=np.float32)
        arr[:2048] = fp_arr
        arr[2048] = float(Descriptors.ExactMolWt(mol))
        arr[2049] = float(Descriptors.MolLogP(mol))
        arr[2050] = float(Descriptors.TPSA(mol))
        arr[2051] = float(Descriptors.RingCount(mol))
        arr[2052] = float(Descriptors.NumRotatableBonds(mol))
        arr[2053] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9))
        arr[2054] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8))
        arr[2055] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7))
        arr[2056] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16))
        arr[2057] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        arr[2058] = float(rdMolDescriptors.CalcFractionCSP3(mol))
        arr[2059] = float(Descriptors.NumHAcceptors(mol))
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
            from aurelius.types import MoleculeContext
            ctx = MoleculeContext(smiles=entry["smiles"], mol=mol)
            fv = self._build_feature_vector(ctx)
            X_list.append(fv)
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

    def predict(self, ctx: MoleculeContext) -> tuple[float, float, float]:
        """Predict (homo_eV, lumo_eV, uncertainty_score) using the trained surrogate.

        Training happens lazily on first call. Returns all three values.
        uncertainty_score is the standard deviation across RF tree predictions,
        serving as an epistemic uncertainty proxy.
        """
        self._ensure_trained()
        fv = self._build_feature_vector(ctx).reshape(1, -1)

        t0 = time.perf_counter()
        homo = float(self._homo_model.predict(fv)[0])  # type: ignore[union-attr]
        lumo = float(self._lumo_model.predict(fv)[0])  # type: ignore[union-attr]
        _inference_ms = (time.perf_counter() - t0) * 1000

        # Epistemic uncertainty: std dev across individual tree predictions
        uncertainty_score = float(
            np.std([tree.predict(fv)[0] for tree in self._homo_model.estimators_], axis=0),
        )

        return homo, lumo, uncertainty_score

    def evaluate_holdout_spearman(self, property: str = "homo") -> float:
        """Compute Spearman rank correlation on a 20% holdout set.

        Parameters
        ----------
        property : str
            "homo" or "lumo" to select which property to evaluate.

        Returns
        -------
        float
            Spearman rho on the holdout set. Returns 0.0 if insufficient data.
        """
        data = self._load_data()
        if len(data) < 10:
            return 0.0

        indices = list(range(len(data)))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=self._random_state,
        )

        train_data = [data[i] for i in train_idx]
        holdout_data = [data[i] for i in test_idx]

        surrogate = SurrogateQuantumOracle(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )
        surrogate.set_training_data(train_data)
        surrogate._ensure_trained()

        key_map = {"homo": "homo_eV", "lumo": "lumo_eV"}
        index_map = {"homo": 0, "lumo": 1}
        key = key_map[property]
        pred_index = index_map[property]
        y_true: list[float] = []
        y_pred: list[float] = []
        for entry in holdout_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            ctx = MoleculeContext(smiles=entry["smiles"], mol=mol)
            try:
                pred = surrogate.predict(ctx)
                y_true.append(entry[key])
                y_pred.append(pred[pred_index])
            except Exception:
                continue

        if len(y_true) < 5:
            return 0.0

        rho, _ = _spearmanr(np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64))
        return float(rho)

    def compute_penalty(self, homo_eV: float, uncertainty_score: float = 0.0) -> float:
        """Return 0.5x penalty if surrogate predicts unstable HOMO (> -5.0 eV).

        If uncertainty_score > 0.5 eV (high epistemic uncertainty), returns 1.0
        (no penalty), forcing the main loop to use the accurate TOM/xTB oracle instead.
        """
        if uncertainty_score > 0.5:
            return 1.0
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
