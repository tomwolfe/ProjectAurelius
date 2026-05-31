"""Agent state management: checkpointing, convergence tracking, and feedback."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aurelius.agent.surrogate import RandomForestSurrogate
from aurelius.utils.chem_utils import generate_ecfp4_fingerprint

if TYPE_CHECKING:
    from aurelius.agent.loop import ScreeningResult  # noqa: F401


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    """Resolve a relative output path against a base directory.

    If ``output_dir`` is provided, ``path`` is joined to it.
    Otherwise ``path`` is returned unchanged (backward-compatible).

    Args:
        path: Output file path (relative).
        output_dir: Base directory to resolve against.

    Returns:
        Resolved absolute path.
    """
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


class CheckpointManager:
    """Manages agent state using atomic JSON writes.

    Stores only aggregate stats and top-N discoveries to keep JSON
    files small.  Uses atomic writes (write to .tmp, then os.replace)
    to prevent corruption during crashes.
    """

    _MAX_DISCOVERIES = 100

    def __init__(self, path: str = "agent_state.json", output_dir: str | Path | None = None) -> None:
        """Initialize the checkpoint manager.

        Args:
            path: Path to the JSON state file (relative to output_dir).
            output_dir: Directory to write to. If None, uses current working directory.
        """
        self.path = _resolve_output_path(path, output_dir)
        self._batch = 0
        self._screened_count = 0
        self._best_score = 0.0
        self._viable_count = 0
        self._total_generated = 0
        self._invalid_discarded = 0
        self._started_at = datetime.now(UTC).isoformat()
        self._last_updated: str | None = None

        self._discoveries: list[dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        """Load checkpoint state from the JSON file.

        If the file does not exist, initializes with default values.
        """
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._batch = data.get("batch", 0)
                self._screened_count = data.get("screened_count", 0)
                self._best_score = data.get("best_score", 0.0)
                self._viable_count = data.get("viable_count", 0)
                self._total_generated = data.get("total_generated", 0)
                self._invalid_discarded = data.get("invalid_discarded", 0)
                self._started_at = data.get("started_at", self._started_at)
                self._last_updated = data.get("last_updated")
                self._discoveries = data.get("discoveries", [])
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

    def _save(self) -> None:
        """Save checkpoint state to the JSON file atomically."""
        data = {
            "batch": self._batch,
            "screened_count": self._screened_count,
            "best_score": self._best_score,
            "viable_count": self._viable_count,
            "total_generated": self._total_generated,
            "invalid_discarded": self._invalid_discarded,
            "started_at": self._started_at,
            "last_updated": self._last_updated,
            "discoveries": self._discoveries,
        }

        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def load(self) -> dict[str, Any]:
        """Load checkpoint state from disk.

        Returns:
            Dict of checkpoint state.  Returns empty state if file
            cannot be read.
        """
        self._load()
        return self._get_state_dict()

    def save(self) -> None:
        """Save checkpoint state to the JSON file atomically."""
        self._save()

    def add_discovery(self, discovery: dict[str, Any] | ScreeningResult) -> None:
        """Add a discovery to the checkpoint.

        Only the top ``_MAX_DISCOVERIES`` discoveries are kept, sorted
        by ``total_score`` descending.

        Args:
            discovery: Dict or ScreeningResult with discovery data.
        """
        if hasattr(discovery, "smiles"):
            discovery = {
                "smiles": discovery.smiles,  # type: ignore[union-attr]
                "total_score": discovery.total_score,  # type: ignore[union-attr]
                "is_viable": discovery.is_viable,  # type: ignore[union-attr]
                "rejection_reasons": discovery.rejection_reasons,  # type: ignore[union-attr]
            }
        self._discoveries.append(discovery)
        # Keep only top N by score
        self._discoveries.sort(key=lambda d: d.get("total_score", 0), reverse=True)
        self._discoveries = self._discoveries[: self._MAX_DISCOVERIES]

    def update_stats(
        self,
        batch_smiles: list[str],
        batch_scores: list[float],
        viable_count: int,
        invalid_count: int,
    ) -> None:
        """Update checkpoint stats for a batch.

        Args:
            batch_smiles: List of SMILES in the batch.
            batch_scores: List of scores for the batch.
            viable_count: Number of viable molecules.
            invalid_count: Number of invalid discarded molecules.
        """
        self._batch += 1
        self._screened_count += len(batch_smiles)
        self._total_generated += len(batch_smiles)
        self._invalid_discarded += invalid_count
        self._viable_count += viable_count
        if batch_scores:
            best = max(batch_scores)
            if best > self._best_score:
                self._best_score = best

    def clear(self) -> None:
        """Clear all discoveries and agent state."""
        self._batch = 0
        self._screened_count = 0
        self._best_score = 0.0
        self._viable_count = 0
        self._total_generated = 0
        self._invalid_discarded = 0
        self._discoveries.clear()
        self._save()

    @property
    def state(self) -> dict[str, Any]:
        """Expose current checkpoint state as a dict (for API compatibility).

        Returns:
            Dict with all checkpoint state values.
        """
        return self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        """Return current state as a dict (for API compatibility)."""
        return {
            "batch": self._batch,
            "screened_count": self._screened_count,
            "best_score": self._best_score,
            "viable_count": self._viable_count,
            "total_generated": self._total_generated,
            "invalid_discarded": self._invalid_discarded,
            "discoveries": self._discoveries,
            "started_at": self._started_at,
            "last_updated": self._last_updated,
        }


class ConvergenceChecker:
    """Evaluates whether the screening loop should terminate.

    Uses Welford's online algorithm for computing running statistics
    without storing all individual scores.  Only per-batch aggregate
    values are stored for plateau and saturation detection.
    """

    def __init__(self) -> None:
        """Initialize the convergence checker."""
        self.new_clusters_per_batch: list[int] = []
        self._batch_means: list[float] = []
        self.viable_count = 0
        self.total_screened = 0
        self.generations = 0
        self.seed_pool_size: int = 0

        # Welford's online statistics
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0

    def record_batch(
        self,
        scores: list[float],
        viable_count: int,
        new_clusters: int,
    ) -> None:
        """Record batch results for convergence tracking.

        Args:
            scores: List of total scores for this batch.
            viable_count: Number of viable molecules in the batch.
            new_clusters: Number of new clusters discovered.
        """
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1

        self.new_clusters_per_batch.append(new_clusters)

        # Update Welford's online statistics and record batch mean
        for score in scores:
            self._welford_n += 1
            delta = score - self._welford_mean
            self._welford_mean += delta / self._welford_n
            self._welford_m2 += delta * (score - self._welford_mean)
        self._batch_means.append(self._welford_mean)

    def check_score_plateau(self) -> bool:
        """Check if running mean changes < 1.0% over 3 consecutive batches.

        Uses the per-batch running mean from Welford's algorithm.

        Returns:
            True if the score has plateaued.
        """
        if len(self._batch_means) < 3:
            return False
        last_three = self._batch_means[-3:]
        for i in range(1, 3):
            ref = last_three[i - 1]
            if ref == 0:
                return False
            change = abs(last_three[i] - ref) / abs(ref)
            if change >= 0.01:
                return False
        return True

    def check_structural_saturation(self) -> bool:
        """Check if < 3 new clusters over last 2 batches.

        Returns:
            True if structural saturation is detected.
        """
        if len(self.new_clusters_per_batch) < 2:
            return False
        return self.new_clusters_per_batch[-1] < 3 and self.new_clusters_per_batch[-2] < 3

    def should_terminate(self) -> tuple[bool, str]:
        """Determine if the screening loop should terminate.

        Returns:
            Tuple of (should_terminate, reason_string).
        """
        plateau = self.check_score_plateau()
        saturation = self.check_structural_saturation()
        if plateau and saturation:
            return True, "All convergence criteria met"
        reasons = []
        if not plateau:
            reasons.append("score plateau")
        if not saturation:
            reasons.append("structural saturation")
        return False, f"Volume met but not all criteria: {', '.join(reasons)}"

    def final_score_variance(self) -> float:
        """Compute the variance of all recorded scores using Welford's algorithm.

        Returns:
            Variance of all scores.
        """
        if self._welford_n < 2:
            return 0.0
        return float(self._welford_m2 / (self._welford_n - 1))


class FeedbackAdapter:
    """Adapts mutation strategy based on rejection patterns using RF active learning.

    The adapter maintains a Random Forest surrogate that scores candidate
    molecules via Expected Improvement acquisition, enabling the discovery
    loop to intelligently select high-value candidates from the mutation pool.
    """

    def __init__(self) -> None:
        """Initialize the feedback adapter."""
        self._total_screened = 0
        self._surrogate: RandomForestSurrogate | None = None
        self._X_history: list[np.ndarray[Any, Any]] = []
        self._y_history: list[float] = []
        self._rng = np.random.default_rng(42)

    def record(self, result: ScreeningResult) -> None:
        """Record screening result for feedback analysis.

        Args:
            result: Typed ScreeningResult with score and viability information.
        """
        self._total_screened += 1

        fp = result.fingerprint
        if fp is None:
            try:
                fp = generate_ecfp4_fingerprint(result.smiles)
                # Append zero-valued global descriptors to match 2053-dim feature vector
                if len(fp) == 2048:
                    fp = np.concatenate([fp, np.zeros(5, dtype=np.float32)])
            except Exception:
                fp = np.zeros(2053, dtype=np.float32)
        self._X_history.append(fp)
        self._y_history.append(result.total_score)

        self.maybe_fit_from_history()

    def maybe_fit_from_history(self, min_samples: int = 10) -> None:
        """Fit the surrogate from accumulated history if enough samples exist.

        Args:
            min_samples: Minimum number of samples required to fit.
        """
        if self._surrogate is not None:
            return
        if len(self._X_history) < min_samples:
            return
        X = np.array(self._X_history, dtype=np.float32)
        y = np.array(self._y_history, dtype=np.float32)
        self._surrogate = RandomForestSurrogate()
        self._surrogate.fit(X, y)

    def update(self, X_new: np.ndarray[Any, Any], y_new: np.ndarray[Any, Any]) -> None:
        """Retrain the RF surrogate with newly screened data, accumulating history.

        Combines:
        1. Any in-memory history from individual ``record()`` calls since last update.
        2. Previously stored surrogate training data (all prior batches).
        3. The new batch data ``X_new`` / ``y_new``.

        After fitting, the in-memory history is cleared (it is now part of the
        surrogate's stored training set).

        Args:
            X_new: 2-D array of Morgan fingerprints for new candidates.
            y_new: 1-D or 2-D array of Aurelius scores for new candidates.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
        if self._surrogate is None:
            self._surrogate = RandomForestSurrogate()

        X_parts = [np.asarray(x, dtype=np.float32) for x in self._X_history]
        y_parts = [np.atleast_1d(np.asarray(y, dtype=np.float32)) for y in self._y_history]

        # Include previously stored surrogate training data (all prior batches)
        prev_X = self._surrogate._X
        prev_y = self._surrogate._y
        if prev_X is not None and prev_y is not None:
            X_parts.append(prev_X)
            y_prev_flat = prev_y.ravel() if prev_y.ndim > 1 else prev_y
            y_parts.append(y_prev_flat)

        # Include new batch data
        X_parts.append(np.asarray(X_new, dtype=np.float32))
        y_parts.append(np.asarray(y_new, dtype=np.float32).ravel())

        X_full = np.vstack(X_parts)
        y_full = np.concatenate(y_parts)

        self._surrogate.fit(X_full, y_full)
        self._X_history.clear()
        self._y_history.clear()

    def get_adaptation_strategy(self) -> dict[str, Any]:
        """Return current mutation adaptation recommendations.

        Returns:
            Dict with strategy recommendation and fail rates.
        """
        strategy: dict[str, Any] = {
            "total_screened": self.total_screened,
        }
        if self._y_history:
            avg_score = np.mean(self._y_history)
            strategy["average_score"] = float(avg_score)
            if avg_score < 50.0:
                strategy["recommendation"] = "Increase exploration diversity — current scores are low"
            elif avg_score < 65.0:
                strategy["recommendation"] = "Focus on high-score candidates from the mutation pool"
            else:
                strategy["recommendation"] = "Continue current mutation strategy"
        else:
            strategy["recommendation"] = "Insufficient data for adaptation recommendations"
        return strategy

    def write_rationale_log(self, path: str = "mutation_rationale.md", output_dir: str | Path | None = None) -> None:
        """Write current adaptation strategy to markdown file.

        Args:
            path: Output file path (relative to output_dir).
            output_dir: Directory to write to. If None, uses current working directory.
        """
        log = logging.getLogger("aurelius_agent")

        path = _resolve_output_path(path, output_dir)
        strategy = self.get_adaptation_strategy()

        with open(path, "w") as f:
            f.write("# Mutation Rationale Log\n\n")
            f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")
            f.write("## Strategy Summary\n\n")
            f.write(f"- **Recommendation:** {strategy['recommendation']}\n")
            if "average_score" in strategy:
                f.write(f"- **Average Score:** {strategy['average_score']:.2f}\n")
        log.info("Mutation rationale log written to %s", path)

    @property
    def total_screened(self) -> int:
        """Return total number of screened molecules."""
        return self._total_screened

