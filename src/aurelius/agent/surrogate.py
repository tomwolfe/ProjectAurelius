"""Random Forest surrogate model for distance-weighted candidate acquisition.

The acquisition function ranks candidates by predicted score, weighted by
Tanimoto-distance to the nearest training point. This is mathematically
transparent: the RF provides a point estimate, and chemical novelty
(measured via ECFP4 Tanimoto distance) provides the exploration bonus.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from sklearn.ensemble import RandomForestRegressor

from aurelius.types import MoleculeContext

_KNOWN_ELECTROLYTE_FPS: list[ExplicitBitVect] | None = None


def _load_known_electrolytes() -> list[ExplicitBitVect]:
    global _KNOWN_ELECTROLYTE_FPS
    if _KNOWN_ELECTROLYTE_FPS is not None:
        return _KNOWN_ELECTROLYTE_FPS

    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "known_electrolytes.json"
    )
    try:
        with open(json_path) as f:
            smiles_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    fps: list[ExplicitBitVect] = []
    for smi in smiles_list:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is not None:
            fps.append(ctx.get_ecfp4())
    _KNOWN_ELECTROLYTE_FPS = fps
    return fps


def global_novelty_penalty(
    candidate_bv: ExplicitBitVect,
    threshold: float = 0.85,
) -> float:
    known_fps = _load_known_electrolytes()
    if not known_fps:
        return 1.0
    target_size = known_fps[0].GetNumBits()
    if candidate_bv.GetNumBits() != target_size:
        if candidate_bv.GetNumBits() > target_size:
            truncated = ExplicitBitVect(target_size)
            for i in range(target_size):
                if candidate_bv.GetBit(i):
                    truncated.SetBit(i)
            candidate_bv = truncated
        else:
            padded = ExplicitBitVect(target_size)
            for i in range(candidate_bv.GetNumBits()):
                if candidate_bv.GetBit(i):
                    padded.SetBit(i)
            candidate_bv = padded
    try:
        sims = BulkTanimotoSimilarity(candidate_bv, known_fps)
    except Exception:
        return 1.0
    if not sims:
        return 1.0
    max_sim = max(sims)
    if max_sim <= threshold:
        return 1.0
    return float(1.0 - (max_sim - threshold) / (1.0 - threshold))


class RandomForestSurrogate:
    """Random Forest surrogate for distance-weighted acquisition.

    The acquisition function is:

        score = predicted_mean * (1 + alpha * tanimoto_distance_to_training)

    where ``tanimoto_distance_to_training`` is the maximum Tanimoto distance
    to any point in the training set (ECFP4 bits only). This is a pure,
    mathematically transparent exploration bonus — no pseudo-variance tricks.
    """

    def __init__(self, random_state: int = 42, alpha: float = 1.0) -> None:
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._rf: RandomForestRegressor | None = None
        self._random_state = random_state
        self._alpha = alpha
        self._train_bitvects: list[ExplicitBitVect] | None = None

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        if len(y) < 2:
            raise ValueError("At least 2 samples are required to fit the Random Forest surrogate.")

        self._rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=self._random_state,
            n_jobs=1,
        )
        self._rf.fit(X, y)

        self._X = X
        self._y = y

        ecfp_nbits = min(2048, X.shape[1])
        self._train_bitvects = []
        for i in range(X.shape[0]):
            bv = ExplicitBitVect(ecfp_nbits)
            row = X[i, :ecfp_nbits]
            for idx in np.flatnonzero(row > 0.5):
                bv.SetBit(int(idx))
            self._train_bitvects.append(bv)

    def expected_improvement(self, X_candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute distance-weighted acquisition values for candidates.

        Returns ``predicted_mean * (1 + alpha * Tanimoto_distance_to_nearest_train)``,
        with a global-novelty penalty applied.
        """
        if self._X is None or self._y is None:
            raise RuntimeError(
                "RandomForestSurrogate must be fitted before scoring candidates. "
                "Call .fit(X, y) with training data first."
            )
        if self._rf is None:
            raise RuntimeError("Random Forest surrogate is not trained.")

        means = self._rf.predict(X_candidates)
        ecfp_nbits = min(2048, X_candidates.shape[1])

        candidate_bitvects = []
        for i in range(X_candidates.shape[0]):
            bv = ExplicitBitVect(ecfp_nbits)
            for idx in np.flatnonzero(X_candidates[i, :ecfp_nbits] > 0.5):
                bv.SetBit(int(idx))
            candidate_bitvects.append(bv)

        max_tanimoto_dist = np.zeros(X_candidates.shape[0], dtype=np.float64)
        for i, cand_bv in enumerate(candidate_bitvects):
            if sum(cand_bv.ToList()) == 0:
                continue
            sims = BulkTanimotoSimilarity(cand_bv, self._train_bitvects)
            if sims:
                max_tanimoto_dist[i] = 1.0 - max(sims)

        acquisition = means * (1.0 + self._alpha * max_tanimoto_dist)

        for i, cand_bv in enumerate(candidate_bitvects):
            acquisition[i] *= global_novelty_penalty(cand_bv, threshold=0.85)

        return acquisition

    def score_candidates(
        self,
        X_candidates: np.ndarray[Any, Any],
        top_n: int = 10,
    ) -> list[int]:
        nwei = self.expected_improvement(X_candidates)
        top_indices = np.argsort(nwei)[::-1][:top_n]
        return top_indices.tolist()
