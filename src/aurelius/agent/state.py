"""Unified loop state: checkpointing and simple metric tracking.

Single source of truth for the screening loop's accumulated state.
Focuses on checkpointing, discovery tracking, and simple batch-level
metrics. Convergence logic lives in standalone helper functions.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aurelius.types import ScreeningResult

_MAX_DISCOVERIES = 100
log = logging.getLogger(__name__)


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


def check_score_plateau(batch_means: list[float], n_batches: int = 3, tolerance: float = 0.01) -> bool:
    """Check if mean scores have plateaued over the last n_batches."""
    if len(batch_means) < n_batches:
        return False
    recent = batch_means[-n_batches:]
    ref = recent[0]
    if ref == 0:
        return False
    return all(abs(v - ref) / abs(ref) < tolerance for v in recent[1:])


def check_structural_saturation(new_scaffolds_per_batch: list[list[str]], n_batches: int = 2, threshold: int = 3) -> bool:
    """Check if the number of new scaffolds per batch has saturated."""
    if len(new_scaffolds_per_batch) < n_batches:
        return False
    recent = new_scaffolds_per_batch[-n_batches:]
    return all(len(scaffolds) < threshold for scaffolds in recent)


@dataclass
class LoopState:
    """Unified agent loop state: checkpointing and metric tracking.

    Stores batch-level metrics for convergence detection and handles
    atomic JSON checkpointing for resume capability.
    """

    # --- Batch metrics ---
    batch_means: list[float] = field(default_factory=list)
    _all_scores: list[float] = field(default_factory=list)
    viable_count: int = 0
    total_screened: int = 0
    generations: int = 0
    seed_pool_size: int = 0

    # --- Scaffold tracking ---
    scaffolds_per_batch: list[list[str]] = field(default_factory=list)

    # --- Checkpointing ---
    best_score: float = 0.0
    discoveries: list[dict[str, Any]] = field(default_factory=list)
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
    # Convergence
    # ------------------------------------------------------------------

    def record_batch(self, scores: list[float], viable_count: int) -> None:
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1
        self._all_scores.extend(scores)
        self.batch_means.append(float(np.mean(scores)) if scores else 0.0)

    def should_terminate(self) -> tuple[bool, str]:
        plateau = check_score_plateau(self.batch_means)
        saturation = check_structural_saturation(self.scaffolds_per_batch)
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
    # Scaffold tracking
    # ------------------------------------------------------------------

    def record_scaffolds(self, scaffolds: list[str]) -> None:
        self.scaffolds_per_batch.append(list(set(scaffolds)))

    def has_scaffold_stagnation(self, n_batches: int = 3) -> bool:
        if len(self.scaffolds_per_batch) < n_batches:
            return False
        recent = self.scaffolds_per_batch[-n_batches:]
        all_scaffolds: list[str] = []
        for batch in recent:
            all_scaffolds.extend(batch)
        if not all_scaffolds:
            return False
        counts = Counter(all_scaffolds)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count >= n_batches

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.total_screened = data.get("total_screened", 0)
                self.best_score = data.get("best_score", 0.0)
                self.viable_count = data.get("viable_count", 0)
                self.invalid_discarded = data.get("invalid_discarded", 0)
                self.started_at = data.get("started_at", self.started_at)
                self.last_updated = data.get("last_updated")
                self.discoveries = data.get("discoveries", [])
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

    def save(self) -> None:
        data = {
            "total_screened": self.total_screened,
            "best_score": self.best_score,
            "viable_count": self.viable_count,
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

    def clear(self) -> None:
        self.batch_means.clear()
        self._all_scores.clear()
        self.generations = 0
        self.seed_pool_size = 0
        self.total_screened = 0
        self.best_score = 0.0
        self.viable_count = 0
        self.invalid_discarded = 0
        self.discoveries.clear()
        self.save()


__all__ = [
    "LoopState",
    "check_score_plateau",
    "check_structural_saturation",
]
