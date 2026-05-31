"""Random Forest surrogate model for Bayesian-guided active learning.

Provides Novelty-Weighted Expected Improvement (NWEI) acquisition
function scoring over ECFP4 fingerprints augmented with global RDKit
descriptors, enabling the DiscoveryLoop to select candidates that are
both promising and structurally novel.

The surrogate uses a Random Forest Regressor which natively handles
high-dimensional sparse binary vectors and provides uncertainty
estimates from tree-based variance for the NWEI calculation.
The novelty bonus is computed as the maximum Tanimoto distance of each
candidate to the training set, rewarding out-of-distribution exploration.

A **global novelty penalty** is also applied against a static list of
known commercial electrolytes (EC, DMC, DEC, FEC, etc.) loaded from
``src/aurelius/data/known_electrolytes.json``. Candidates with Tanimoto
similarity > 0.85 to any known electrolyte receive a penalty, preventing
rediscovery of patented or well-known compounds.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from scipy.stats import norm as _norm
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Global known-electrolyte fingerprint database
# ---------------------------------------------------------------------------
_KNOWN_ELECTROLYTE_FPS: list[ExplicitBitVect] | None = None
_KNOWN_ELECTROLYTE_NAMES: list[str] | None = None


def _load_known_electrolytes() -> tuple[list[ExplicitBitVect], list[str]]:
    """Load the global known-electrolyte fingerprint database.

    Returns:
        (fingerprints, names) tuple.
    """
    global _KNOWN_ELECTROLYTE_FPS, _KNOWN_ELECTROLYTE_NAMES
    if _KNOWN_ELECTROLYTE_FPS is not None:
        return _KNOWN_ELECTROLYTE_FPS, _KNOWN_ELECTROLYTE_NAMES or []

    from rdkit import Chem
    from rdkit.Chem import AllChem

    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "known_electrolytes.json"
    )
    try:
        with open(json_path) as f:
            smiles_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []

    fps: list[ExplicitBitVect] = []
    names: list[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
            names.append(smi)
    _KNOWN_ELECTROLYTE_FPS = fps
    _KNOWN_ELECTROLYTE_NAMES = names
    return fps, names


def global_novelty_penalty(
    candidate_bv: ExplicitBitVect,
    threshold: float = 0.85,
) -> float:
    """Compute penalty multiplier for similarity to known electrolytes.

    Args:
        candidate_bv: ECFP4 fingerprint of the candidate molecule.
        threshold: Tanimoto similarity threshold above which penalty applies.

    Returns:
        Penalty multiplier in [0, 1]. 1.0 = no similarity to known set,
        values < 1.0 mean the candidate resembles a known electrolyte.
    """
    known_fps, _ = _load_known_electrolytes()
    if not known_fps:
        return 1.0
    # Handle size mismatch: if candidate has fewer bits (e.g. 3-bit test features),
    # upscale to 2048 bits with only the intersection considered
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
    # Linear penalty from threshold to 0 at perfect similarity
    return float(1.0 - (max_sim - threshold) / (1.0 - threshold))


class RandomForestSurrogate:
    """Random Forest surrogate model for active learning.

    The surrogate is trained on Morgan + global descriptor features ``X``
    and composite ``Aurelius Score`` values ``y``.  During acquisition
    (``expected_improvement``), the RF predicts the mean and variance
    of each candidate, and the acquisition function scores high both
    high predicted scores and high uncertainty.

    The Novelty-Weighted EI (NWEI) multiplies standard EI by
    ``1.0 + alpha * max_tanimoto_distance``, where the Tanimoto distance
    is computed against the training set using only the ECFP4 bits
    (first 2048 features).

    A **global novelty penalty** is also applied to penalize candidates
    that are too similar (Tanimoto > 0.85) to any known commercial electrolyte
    in the static database.
    """

    def __init__(self, random_state: int = 42, alpha: float = 1.0) -> None:
        """Initialise the surrogate.

        Args:
            random_state: Random seed for reproducibility.
            alpha: Novelty bonus strength for NWEI.
        """
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._rf: RandomForestRegressor | None = None
        self._random_state = random_state
        self._alpha = alpha
        self._train_bitvects: list[ExplicitBitVect] | None = None
        # Pre-load known electrolytes
        _load_known_electrolytes()

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """Fit the Random Forest surrogate to (X, y) data.

        Args:
            X: 2-D array of shape (n_samples, n_features) with Morgan
                fingerprints (ECFP4 radius=2).
            y: 1-D array of composite Aurelius scores.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
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

        # Store the original data for Expected Improvement computation
        self._X = X
        self._y = y

        # Pre-compute ExplicitBitVect objects for Tanimoto novelty computation
        ecfp_nbits = min(2048, X.shape[1])
        self._train_bitvects = []
        for i in range(X.shape[0]):
            bv = ExplicitBitVect(ecfp_nbits)
            row = X[i, :ecfp_nbits]
            for idx in np.flatnonzero(row > 0.5):
                bv.SetBit(int(idx))
            self._train_bitvects.append(bv)

    def expected_improvement(self, X_candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute Novelty-Weighted Expected Improvement (NWEI) for candidates.

        NWEI favours candidates that either have high predicted mean,
        high predictive uncertainty, OR high structural novelty relative
        to the training set.

        Inter-tree variance from the Random Forest's ensemble of decision
        trees is used as a proxy for epistemic uncertainty in the EI
        calculation.

        The novelty bonus is computed as the maximum Tanimoto distance
        of each candidate's ECFP4 bits (first 2048 features) to the
        training set, scaled by ``alpha``.

        A **global novelty penalty** is multiplied into the final score
        to discourage rediscovery of known commercial electrolytes
        (Tanimoto similarity > 0.85 against the static known set).

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).

        Returns:
            1-D array of NWEI acquisition values (higher = more acquisition-worthy).

        Raises:
            RuntimeError: If the surrogate has not been fitted yet.
        """
        if self._X is None or self._y is None:
            raise RuntimeError(
                "RandomForestSurrogate must be fitted before scoring candidates. "
                "Call .fit(X, y) with training data first."
            )

        if self._rf is None:
            raise RuntimeError("Random Forest surrogate is not trained.")

        best = self._y.max() if len(self._y) > 0 else 0.0

        # Vectorized: get predictions from all trees at once
        tree_preds = np.array([tree.predict(X_candidates) for tree in self._rf.estimators_])
        means = tree_preds.mean(axis=0)
        variances = tree_preds.var(axis=0)

        # Standard Expected Improvement using scipy.stats.norm
        s = np.sqrt(np.maximum(variances, 1e-8))
        ei = np.zeros_like(means)
        mask = s >= 1e-8
        z = (means[mask] - best) / s[mask]
        ei[mask] = (means[mask] - best) * _norm.cdf(z) + s[mask] * _norm.pdf(z)

        # Novelty bonus: max Tanimoto distance to training set (ECFP4 bits only)
        # Uses RDKit BulkTanimotoSimilarity on ExplicitBitVect for C++-speed bulk operation
        ecfp_nbits = min(2048, X_candidates.shape[1])
        max_tanimoto_dist = np.zeros(X_candidates.shape[0], dtype=np.float64)
        candidate_bitvects: list[ExplicitBitVect] = []
        for i in range(X_candidates.shape[0]):
            bv = ExplicitBitVect(ecfp_nbits)
            row = X_candidates[i, :ecfp_nbits]
            for idx in np.flatnonzero(row > 0.5):
                bv.SetBit(int(idx))
            candidate_bitvects.append(bv)

        for i, cand_bv in enumerate(candidate_bitvects):
            if sum(cand_bv.ToList()) == 0:
                continue
            sims = BulkTanimotoSimilarity(cand_bv, self._train_bitvects)
            if sims:
                max_tanimoto_dist[i] = 1.0 - max(sims)

        # Novelty-weighted EI
        nwei = ei * (1.0 + self._alpha * max_tanimoto_dist)

        # Global novelty penalty: penalize similarity to known commercial electrolytes
        for i, cand_bv in enumerate(candidate_bitvects):
            penalty = global_novelty_penalty(cand_bv, threshold=0.85)
            nwei[i] *= penalty

        return nwei

    def score_candidates(
        self,
        X_candidates: np.ndarray[Any, Any],
        top_n: int = 10,
    ) -> list[int]:
        """Return indices of top-N candidates by Novelty-Weighted EI.

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).
            top_n: Number of top candidates to return.

        Returns:
            List of indices into ``X_candidates`` sorted by descending NWEI.

        Raises:
            RuntimeError: If the surrogate has not been fitted yet.
        """
        nwei = self.expected_improvement(X_candidates)
        top_indices = np.argsort(nwei)[::-1][:top_n]
        return top_indices.tolist()
