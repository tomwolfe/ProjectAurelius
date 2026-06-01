"""Unified loop state: history, convergence, and checkpointing.

Single source of truth for the screening loop's accumulated state.
Replaces the former ``FeedbackAdapter``, ``ConvergenceChecker``, and
``CheckpointManager`` with a single ``LoopState`` dataclass.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aurelius.types import ScreeningResult

_MAX_DISCOVERIES = 100


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


@dataclass
class LoopState:
    """Unified agent loop state: history, convergence, and checkpointing.

    Stores (X, y) history for surrogate training, convergence metrics for
    termination decisions, and handles atomic JSON checkpointing for resume.

    This replaces ``FeedbackAdapter``, ``ConvergenceChecker``, and
    ``CheckpointManager``.
    """

    # --- History (formerly FeedbackAdapter) ---
    X_history: list[np.ndarray[Any, Any]] = field(default_factory=list)
    y_history: list[float] = field(default_factory=list)

    # --- Convergence (formerly ConvergenceChecker) ---
    new_clusters_per_batch: list[int] = field(default_factory=list)
    batch_means: list[float] = field(default_factory=list)
    _all_scores: list[float] = field(default_factory=list)
    viable_count: int = 0
    total_screened: int = 0
    generations: int = 0
    seed_pool_size: int = 0

    # --- Checkpointing (formerly CheckpointManager) ---
    batch: int = 0
    best_score: float = 0.0
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    total_generated: int = 0
    invalid_discarded: int = 0
    started_at: str = ""
    last_updated: str | None = None
    path: str = "agent_state.json"
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.path = _resolve_output_path(self.path, self.output_dir)
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()
        self._load()

    # ------------------------------------------------------------------
    # History (formerly FeedbackAdapter)
    # ------------------------------------------------------------------

    def record(self, result: ScreeningResult) -> None:
        if result.fingerprint is None:
            raise ValueError(
                "ScreeningResult.fingerprint must be a numpy array. "
                "Use MoleculeContext.get_feature_vector() to generate it."
            )
        self.X_history.append(result.fingerprint)
        self.y_history.append(result.total_score)

    def get_history(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return np.array(self.X_history, dtype=np.float32), np.array(self.y_history, dtype=np.float32)

    def get_adaptation_strategy(self) -> dict[str, Any]:
        strategy: dict[str, Any] = {
            "total_screened": len(self.X_history),
        }
        if self.y_history:
            avg_score = np.mean(self.y_history)
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

    # ------------------------------------------------------------------
    # Convergence (formerly ConvergenceChecker)
    # ------------------------------------------------------------------

    def record_batch(self, scores: list[float], viable_count: int, new_clusters: int) -> None:
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1
        self.new_clusters_per_batch.append(new_clusters)
        self._all_scores.extend(scores)
        self.batch_means.append(float(np.mean(scores)) if scores else 0.0)

    def check_score_plateau(self) -> bool:
        if len(self.batch_means) < 3:
            return False
        last_three = self.batch_means[-3:]
        ref = last_three[0]
        if ref == 0:
            return False
        for i in range(1, 3):
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
            reasons.append("score plateau not reached")
        if not saturation:
            reasons.append("structural saturation not reached")
        return False, f"Criteria not yet met: {', '.join(reasons)}"

    def final_score_variance(self) -> float:
        if len(self._all_scores) < 2:
            return 0.0
        return float(np.var(self._all_scores, ddof=1))

    # ------------------------------------------------------------------
    # Checkpointing (formerly CheckpointManager)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.batch = data.get("batch", 0)
                self.total_screened = data.get("total_screened", 0)
                self.best_score = data.get("best_score", 0.0)
                self.viable_count = data.get("viable_count", 0)
                self.total_generated = data.get("total_generated", 0)
                self.invalid_discarded = data.get("invalid_discarded", 0)
                self.started_at = data.get("started_at", self.started_at)
                self.last_updated = data.get("last_updated")
                self.discoveries = data.get("discoveries", [])
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

    def save(self) -> None:
        data = {
            "batch": self.batch,
            "total_screened": self.total_screened,
            "best_score": self.best_score,
            "viable_count": self.viable_count,
            "total_generated": self.total_generated,
            "invalid_discarded": self.invalid_discarded,
            "started_at": self.started_at,
            "last_updated": datetime.now(UTC).isoformat(),
            "discoveries": self.discoveries,
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def add_discovery(self, discovery: dict[str, Any] | ScreeningResult) -> None:
        if hasattr(discovery, "smiles"):
            discovery = {
                "smiles": discovery.smiles,
                "total_score": discovery.total_score,
                "is_viable": discovery.is_viable,
                "rejection_reasons": discovery.rejection_reasons,
            }
        self.discoveries.append(discovery)
        self.discoveries.sort(key=lambda d: d.get("total_score", 0), reverse=True)
        self.discoveries = self.discoveries[:_MAX_DISCOVERIES]

    def update_stats(
        self,
        batch_smiles: list[str],
        batch_scores: list[float],
        viable_count: int,
        invalid_count: int,
    ) -> None:
        self.batch += 1
        self.total_screened += len(batch_smiles)
        self.total_generated += len(batch_smiles)
        self.invalid_discarded += invalid_count
        self.viable_count += viable_count
        if batch_scores:
            best = max(batch_scores)
            if best > self.best_score:
                self.best_score = best

    def clear(self) -> None:
        self.X_history.clear()
        self.y_history.clear()
        self.new_clusters_per_batch.clear()
        self.batch_means.clear()
        self._all_scores.clear()
        self.generations = 0
        self.seed_pool_size = 0
        self.batch = 0
        self.total_screened = 0
        self.best_score = 0.0
        self.viable_count = 0
        self.total_generated = 0
        self.invalid_discarded = 0
        self.discoveries.clear()
        self.save()


# Backward-compatible aliases
FeedbackAdapter = LoopState
ConvergenceChecker = LoopState
CheckpointManager = LoopState

__all__ = [
    "LoopState",
    "FeedbackAdapter",
    "ConvergenceChecker",
    "CheckpointManager",
]
