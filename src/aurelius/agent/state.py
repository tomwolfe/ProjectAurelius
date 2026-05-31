"""Agent state management: checkpointing, convergence tracking, and feedback.

FeedbackAdapter is a pure data buffer that just stores (X, y) history.
Surrogate fitting is owned by DiscoveryLoop in loop.py.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aurelius.types import ScreeningResult


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


class CheckpointManager:
    """Manages agent state using atomic JSON writes."""

    _MAX_DISCOVERIES = 100

    def __init__(self, path: str = "agent_state.json", output_dir: str | Path | None = None) -> None:
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
        self._load()
        return self._get_state_dict()

    def save(self) -> None:
        self._save()

    def add_discovery(self, discovery: dict[str, Any] | ScreeningResult) -> None:
        if hasattr(discovery, "smiles"):
            discovery = {
                "smiles": discovery.smiles,
                "total_score": discovery.total_score,
                "is_viable": discovery.is_viable,
                "rejection_reasons": discovery.rejection_reasons,
            }
        self._discoveries.append(discovery)
        self._discoveries.sort(key=lambda d: d.get("total_score", 0), reverse=True)
        self._discoveries = self._discoveries[: self._MAX_DISCOVERIES]

    def update_stats(
        self,
        batch_smiles: list[str],
        batch_scores: list[float],
        viable_count: int,
        invalid_count: int,
    ) -> None:
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
        return self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
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
    """Evaluates whether the screening loop should terminate."""

    def __init__(self) -> None:
        self.new_clusters_per_batch: list[int] = []
        self._batch_means: list[float] = []
        self._all_scores: list[float] = []
        self.viable_count = 0
        self.total_screened = 0
        self.generations = 0
        self.seed_pool_size: int = 0

    def record_batch(
        self,
        scores: list[float],
        viable_count: int,
        new_clusters: int,
    ) -> None:
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1
        self.new_clusters_per_batch.append(new_clusters)
        self._all_scores.extend(scores)
        self._batch_means.append(float(np.mean(scores)) if scores else 0.0)

    def check_score_plateau(self) -> bool:
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
        if len(self.new_clusters_per_batch) < 2:
            return False
        return self.new_clusters_per_batch[-1] < 3 and self.new_clusters_per_batch[-2] < 3

    def should_terminate(self) -> tuple[bool, str]:
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
        if len(self._all_scores) < 2:
            return 0.0
        return float(np.var(self._all_scores, ddof=1))


class FeedbackAdapter:
    """Pure data buffer storing (X, y) history for surrogate training.

    Surrogate fitting is owned by DiscoveryLoop; this class just stores data.
    """

    def __init__(self) -> None:
        self._total_screened = 0
        self._X_history: list[np.ndarray[Any, Any]] = []
        self._y_history: list[float] = []

    def record(self, result: ScreeningResult) -> None:
        if result.fingerprint is None:
            raise ValueError(
                "ScreeningResult.fingerprint must be a numpy array. "
                "Use MoleculeContext.get_feature_vector() to generate it."
            )
        self._total_screened += 1
        self._X_history.append(result.fingerprint)
        self._y_history.append(result.total_score)

    def get_history(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return (X, y) arrays of all accumulated data."""
        return np.array(self._X_history, dtype=np.float32), np.array(self._y_history, dtype=np.float32)

    def clear(self) -> None:
        self._X_history.clear()
        self._y_history.clear()

    def get_adaptation_strategy(self) -> dict[str, Any]:
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
        return self._total_screened
