"""Unified loop state: checkpointing and simple metric tracking.

Single source of truth for the screening loop's accumulated state.
Focuses on checkpointing, discovery tracking, and simple batch-level
metrics. Convergence logic lives in standalone helper functions.

ADR-2026-06-01: Changed datetime.UTC → datetime.timezone.utc for Python 3.9
compatibility (datetime.UTC was added in 3.11). No behavioral change.

ADR-2026-06-07: Dynamic score re-weighting. When empirical wet-lab feedback
is available, LoopState tracks the correlation between each predicted property
(dielectric, viscosity, Li solvation) and the empirical target metric (e.g.,
cycle_life). Weights are adjusted proportionally to correlation strength,
allowing the EA to upweight properties that empirically matter most.

ADR-2026-06-26: Added ``_cached_fingerprints`` mirror list to keep
``find_nearest_screened`` from rebuilding a list comprehension on every
call. Use ``add_screened_fingerprint()`` instead of directly appending to
``screened_fingerprints`` to keep the cache in sync.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aurelius.types import MoleculeContext

if TYPE_CHECKING:
    from aurelius.types import ScreeningResult

import numpy as np

_MAX_DISCOVERIES = 100
log = logging.getLogger(__name__)

# Default score weight constants (mirrored from constants.py for re-weighting)
_WEIGHT_KEYS: list[str] = [
    "SCORE_WEIGHT_LUMO",
    "SCORE_WEIGHT_HOMO",
    "SCORE_WEIGHT_DIELECTRIC",
    "SCORE_WEIGHT_VISCOSITY",
    "SCORE_WEIGHT_LI_SOLVATION",
    "SCORE_WEIGHT_CED",
    "SCORE_WEIGHT_SA",
]


def _resolve_output_path(path: str, output_dir: str | Path | None = None) -> str:
    if output_dir is not None:
        return str(Path(output_dir) / path)
    return path


def check_score_plateau(
    batch_means: list[float], n_batches: int = 3, tolerance: float = 0.01
) -> bool:
    """Check if mean scores have plateaued over the last n_batches."""
    if len(batch_means) < n_batches:
        return False
    recent = batch_means[-n_batches:]
    ref = recent[0]
    if ref == 0:
        return False
    return all(abs(v - ref) / abs(ref) < tolerance for v in recent[1:])


def check_structural_saturation(
    new_scaffolds_per_batch: list[list[str]],
    n_batches: int = 2,
    threshold: int = 3,
) -> bool:
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

    # --- Similarity caching ---
    _fingerprint_dict: dict[str, tuple[Any, dict[str, Any]]] = field(
        default_factory=dict,
    )
    """SMILES → (fingerprint, result_dict) mapping for O(1) exact lookup.
    Prefer :meth:`add_screened_fingerprint` over direct mutation so that
    the internal dict stays in sync.
    """

    # Legacy list kept for backward compatibility with callers that iterate.
    screened_fingerprints: list[tuple[str, Any, dict[str, Any]]] = field(
        default_factory=list
    )

    # --- Batch metrics ---
    batch_means: list[float] = field(default_factory=list)
    _all_scores: list[float] = field(default_factory=list)
    viable_count: int = 0
    total_screened: int = 0
    generations: int = 0
    seed_pool_size: int = 0

    # --- Result tracking (single source of truth) ---
    _all_results: list[dict[str, Any]] = field(default_factory=list)
    _seen_smiles: set[str] = field(default_factory=set)

    # --- Scaffold tracking ---
    scaffolds_per_batch: list[list[str]] = field(default_factory=list)
    _seen_scaffolds: set[str] = field(default_factory=set)

    # --- Checkpointing ---
    best_score: float = 0.0
    discoveries: list[dict[str, Any]] = field(default_factory=list)
    invalid_discarded: int = 0
    started_at: str = ""
    last_updated: str | None = None
    path: str = "agent_state.json"
    output_dir: str | Path | None = None

    # --- Thread-safety locks (initialised in __post_init__) ---
    _state_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _cache_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _al_queue_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.path = _resolve_output_path(self.path, self.output_dir)
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._al_queue_lock = threading.Lock()
        # Mirror list of fingerprints for fast lookup in find_nearest_screened
        self._cached_fingerprints: list[Any] = []
        self._load()

    # ------------------------------------------------------------------
    # Similarity-cache helpers
    # ------------------------------------------------------------------

    def add_screened_fingerprint(
        self, smi: str, fp: Any, result: dict[str, Any]
    ) -> None:
        """Atomically append a screened fingerprint and mirror it in the dict."""
        with self._cache_lock:
            self.screened_fingerprints.append((smi, fp, result))
            self._cached_fingerprints.append(fp)
            self._fingerprint_dict[smi] = (fp, result)

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
    # Similarity-based oracle caching
    # ------------------------------------------------------------------

    def find_nearest_screened(
        self, ctx: MoleculeContext, threshold: float = 0.95
    ) -> dict[str, Any] | None:
        """Return cached screening result if a similar molecule was already screened.

        First checks for an exact SMILES match in O(1) time via the internal
        ``_fingerprint_dict``.  If no exact match, falls back to
        ``BulkTanimotoSimilarity`` on ``_cached_fingerprints`` to find the
        nearest neighbour above *threshold*.

        Thread-safe via ``_cache_lock``.
        """
        with self._cache_lock:
            exact = self._fingerprint_dict.get(ctx.smiles)
            if exact is not None:
                _, result = exact
                return result

            if not self._cached_fingerprints:
                return None
            stored_fps = self._cached_fingerprints

        fp = ctx.get_ecfp4()
        from rdkit.DataStructs import BulkTanimotoSimilarity

        sims = BulkTanimotoSimilarity(fp, stored_fps)
        max_sim = max(sims)
        if max_sim >= threshold:
            best_idx = sims.index(max_sim)
            with self._cache_lock:
                return self.screened_fingerprints[best_idx][2]
        return None

    # ------------------------------------------------------------------
    # Scaffold tracking
    # ------------------------------------------------------------------

    def record_scaffolds(self, scaffolds: list[str]) -> None:
        self.scaffolds_per_batch.append(list(set(scaffolds)))
        self._seen_scaffolds.update(scaffolds)

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

    @property
    def scientific_yield(self) -> float:
        if self.total_screened <= 0:
            return 0.0
        return len(self._seen_scaffolds) / self.total_screened

    # ------------------------------------------------------------------
    # Empirical feedback tracking for dynamic re-weighting
    # ------------------------------------------------------------------

    _empirical_feedback: list[dict[str, Any]] = field(default_factory=list)
    active_learning_queue: list[str] = field(default_factory=list)

    def record_empirical_feedback(self, feedback: list[dict[str, Any]]) -> None:
        self._empirical_feedback.extend(feedback)

    def _compute_property_correlations(self) -> dict[str, float]:
        if len(self._empirical_feedback) < 3:
            return {
                "dielectric_proxy": 0.0,
                "viscosity_proxy": 0.0,
                "li_solvation_proxy": 0.0,
            }

        props = ["dielectric_proxy", "viscosity_proxy", "li_solvation_proxy"]
        correlations: dict[str, float] = {}
        cycle_lives = np.array(
            [e.get("cycle_life", 0.0) for e in self._empirical_feedback]
        )

        for prop in props:
            values = np.array(
                [e.get(prop, 0.0) for e in self._empirical_feedback]
            )
            if np.std(values) < 1e-6 or np.std(cycle_lives) < 1e-6:
                correlations[prop] = 0.0
            else:
                corr = np.corrcoef(values, cycle_lives)[0, 1]
                correlations[prop] = 0.0 if np.isnan(corr) else float(corr)

        return correlations

    def compute_adjusted_weights(
        self,
        base_weights: dict[str, float] | None = None,
        learning_rate: float = 0.1,
    ) -> dict[str, float]:
        if base_weights is None:
            from aurelius.constants import (
                SCORE_WEIGHT_CED,
                SCORE_WEIGHT_DIELECTRIC,
                SCORE_WEIGHT_HOMO,
                SCORE_WEIGHT_LI_SOLVATION,
                SCORE_WEIGHT_LUMO,
                SCORE_WEIGHT_SA,
                SCORE_WEIGHT_VISCOSITY,
            )
            base_weights = {
                "SCORE_WEIGHT_LUMO": SCORE_WEIGHT_LUMO,
                "SCORE_WEIGHT_HOMO": SCORE_WEIGHT_HOMO,
                "SCORE_WEIGHT_DIELECTRIC": SCORE_WEIGHT_DIELECTRIC,
                "SCORE_WEIGHT_VISCOSITY": SCORE_WEIGHT_VISCOSITY,
                "SCORE_WEIGHT_LI_SOLVATION": SCORE_WEIGHT_LI_SOLVATION,
                "SCORE_WEIGHT_CED": SCORE_WEIGHT_CED,
                "SCORE_WEIGHT_SA": SCORE_WEIGHT_SA,
            }

        correlations = self._compute_property_correlations()
        prop_to_weight: dict[str, str] = {
            "dielectric_proxy": "SCORE_WEIGHT_DIELECTRIC",
            "viscosity_proxy": "SCORE_WEIGHT_VISCOSITY",
            "li_solvation_proxy": "SCORE_WEIGHT_LI_SOLVATION",
        }

        adjusted = dict(base_weights)
        for prop_name, weight_key in prop_to_weight.items():
            r = correlations.get(prop_name, 0.0)
            importance = abs(r)
            adjusted[weight_key] = base_weights.get(weight_key, 0.1) * (
                1.0 + learning_rate * importance
            )

        total = sum(adjusted.values())
        if total > 0.0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def apply_dynamic_weights(self, pipeline: Any) -> None:
        adjusted = self.compute_adjusted_weights()

        import aurelius.constants as consts

        for key, value in adjusted.items():
            if hasattr(consts, key):
                setattr(consts, key, value)

        if hasattr(pipeline, "_OBJECTIVES"):
            for obj in pipeline._OBJECTIVES:
                weight_key = {
                    "lumo_reward": "SCORE_WEIGHT_LUMO",
                    "homo_penalty": "SCORE_WEIGHT_HOMO",
                    "dielectric_reward": "SCORE_WEIGHT_DIELECTRIC",
                    "viscosity_penalty": "SCORE_WEIGHT_VISCOSITY",
                    "li_solvation_reward": "SCORE_WEIGHT_LI_SOLVATION",
                    "ced_reward": "SCORE_WEIGHT_CED",
                    "sa_penalty": "SCORE_WEIGHT_SA",
                }.get(obj.name, "")
                if weight_key and weight_key in adjusted:
                    obj.weight = adjusted[weight_key]

        log.info(
            "Dynamic weights applied: %s",
            {k: f"{v:.4f}" for k, v in adjusted.items()},
        )

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
                self._all_results = data.get("_all_results", [])
                self._seen_smiles = set(
                    r.get("smiles", "") for r in data.get("_all_results", [])
                )
                self._empirical_feedback = data.get(
                    "_empirical_feedback", []
                )
                self.active_learning_queue = data.get(
                    "active_learning_queue", []
                )
                # Rebuild _fingerprint_dict from stored fingerprints
                for entry in data.get("screened_fingerprints", []):
                    smi, fp, result = entry
                    self._fingerprint_dict[smi] = (fp, result)
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

    def save(self) -> None:
        with self._state_lock:
            data = {
                "total_screened": self.total_screened,
                "best_score": self.best_score,
                "viable_count": self.viable_count,
                "invalid_discarded": self.invalid_discarded,
                "started_at": self.started_at,
                "last_updated": datetime.now(UTC).isoformat(),
                "discoveries": self.discoveries,
                "_all_results": self._all_results,
                "active_learning_queue": self.active_learning_queue,
                "screened_fingerprints": [
                    (smi, fp, result)
                    for smi, (fp, result) in self._fingerprint_dict.items()
                ],
            }
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.path)

    def add_discovery(
        self, discovery: dict[str, Any] | ScreeningResult
    ) -> None:
        with self._state_lock:
            if isinstance(discovery, dict):
                self.discoveries.append(discovery)
            else:
                self.discoveries.append(
                    {
                        "smiles": discovery.smiles,
                        "total_score": discovery.total_score,
                        "is_viable": discovery.is_viable,
                        "rejection_reasons": discovery.rejection_reasons,
                    }
                )
            self.discoveries.sort(
                key=lambda d: d.get("total_score", 0), reverse=True
            )
            self.discoveries = self.discoveries[:_MAX_DISCOVERIES]

    def add_result(self, sr: Any) -> None:
        with self._state_lock:
            self._seen_smiles.add(sr.smiles)
            self._all_results.append(
                {
                    "smiles": sr.smiles,
                    "total_score": sr.total_score,
                    "is_viable": sr.is_viable,
                    "rejection_reasons": sr.rejection_reasons,
                    "novelty_to_seed": sr.novelty_to_seed,
                    "homo_eV": sr.homo_eV,
                    "lumo_eV": sr.lumo_eV,
                    "dielectric_proxy": sr.dielectric_proxy,
                    "viscosity_proxy": sr.viscosity_proxy,
                    "li_solvation_proxy": sr.li_solvation_proxy,
                    "sa_score": sr.sa_score,
                    "sub_scores": sr.sub_scores,
                }
            )

    def top_scored_smiles(self, divisor: int = 5) -> list[str]:
        scored = [
            (r["total_score"], r["smiles"])
            for r in self._all_results
            if r["total_score"] > 0
        ]
        scored.sort(key=lambda x: -x[0])
        n = max(5, len(scored) // divisor)
        return [s for _, s in scored[:n]]

    def export_active_learning_queue(self, path: str) -> None:
        with self._al_queue_lock:
            snapshot = list(self.active_learning_queue)
        resolved = _resolve_output_path(path, self.output_dir)
        with open(resolved, "w") as f:
            json.dump(snapshot, f, indent=2)
        log.info(
            "Exported active learning queue (%d SMILES) to %s",
            len(snapshot),
            resolved,
        )

    def clear(self) -> None:
        with self._state_lock:
            self.batch_means.clear()
            self._all_scores.clear()
            self.generations = 0
            self.seed_pool_size = 0
            self.total_screened = 0
            self.best_score = 0.0
            self.viable_count = 0
            self.invalid_discarded = 0
            self.discoveries.clear()
            self._all_results.clear()
            self._seen_smiles.clear()
            self._seen_scaffolds.clear()
            self.screened_fingerprints.clear()
            self._cached_fingerprints.clear()
            self._fingerprint_dict.clear()
        with self._al_queue_lock:
            self.active_learning_queue.clear()
        MoleculeContext.from_smiles.cache_clear()
        self.save()


__all__ = [
    "LoopState",
    "check_score_plateau",
    "check_structural_saturation",
]
