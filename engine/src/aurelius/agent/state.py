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

import numpy as np

if TYPE_CHECKING:
    from aurelius.types import MoleculeContext, ScreeningResult

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

    # --- Similarity caching ---
    screened_fingerprints: list[tuple[str, Any, dict[str, Any]]] = field(default_factory=list)
    """SMILES, ECFP4 fingerprint, and screening result for each screened molecule.
    Used by find_nearest_screened to skip redundant oracle evaluations."""

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

    def __post_init__(self) -> None:
        self.path = _resolve_output_path(self.path, self.output_dir)
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()
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
    # Similarity-based oracle caching
    # ------------------------------------------------------------------

    def find_nearest_screened(
        self, ctx: MoleculeContext, threshold: float = 0.95
    ) -> dict[str, Any] | None:
        """Return cached screening result if a similar molecule was already screened.

        Uses BulkTanimotoSimilarity on ECFP4 fingerprints to find the nearest
        match above the given threshold.

        Thread-safe via _cache_lock.
        """
        if not self.screened_fingerprints:
            return None
        fp = ctx.get_ecfp4()
        with self._cache_lock:
            stored_fps = [entry[1] for entry in self.screened_fingerprints]
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
        """Fraction of screened molecules with novel scaffolds.

        Computed as the number of unique Murcko scaffolds observed across
        all batches divided by the total number of molecules screened.
        A higher value indicates the EA is exploring diverse chemical
        space rather than rediscovering known scaffolds.

        Returns:
            Float in [0.0, 1.0]; 0.0 if no molecules have been screened.
        """
        if self.total_screened <= 0:
            return 0.0
        return len(self._seen_scaffolds) / self.total_screened

    # ------------------------------------------------------------------
    # Empirical feedback tracking for dynamic re-weighting
    # ------------------------------------------------------------------

    _empirical_feedback: list[dict[str, Any]] = field(default_factory=list)
    """Accumulated empirical wet-lab feedback for dynamic weight adjustment.
    Each entry: {smiles, cycle_life, coulombic_efficiency, dielectric_proxy,
    viscosity_proxy, li_solvation_proxy}"""

    active_learning_queue: list[str] = field(default_factory=list)
    """SMILES strings of high-uncertainty molecules queued for active learning
    (real QuantumOracle evaluation instead of surrogate)."""

    def record_empirical_feedback(self, feedback: list[dict[str, Any]]) -> None:
        """Record empirical wet-lab feedback for dynamic weight adjustment.

        Each entry should contain at minimum 'smiles' and 'cycle_life'.
        Optionally includes predicted properties for correlation analysis.
        """
        self._empirical_feedback.extend(feedback)

    def _compute_property_correlations(
        self,
    ) -> dict[str, float]:
        """Compute Pearson correlation between each predicted property
        and empirical cycle_life.

        Returns a dict mapping property_key -> correlation coefficient
        (r in [-1, 1]). Higher absolute r means the property is a stronger
        predictor of empirical cycle_life.
        """
        if len(self._empirical_feedback) < 3:
            return {
                "dielectric_proxy": 0.0,
                "viscosity_proxy": 0.0,
                "li_solvation_proxy": 0.0,
            }

        props = ["dielectric_proxy", "viscosity_proxy", "li_solvation_proxy"]
        correlations: dict[str, float] = {}
        cycle_lives = np.array([e.get("cycle_life", 0.0) for e in self._empirical_feedback])

        for prop in props:
            values = np.array([e.get(prop, 0.0) for e in self._empirical_feedback])
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
        """Compute dynamically adjusted score weights based on empirical feedback.

        Physical justification: If empirical data shows viscosity is strongly
        correlated with cycle_life (r > 0.5), the SCORE_WEIGHT_VISCOSITY
        should increase because viscosity is a more important predictor of
        real-world battery performance than initially calibrated. Conversely,
        a property with weak correlation (r < 0.1) has its weight slightly
        decreased.

        The adjustment uses a soft learning rate to prevent oscillation:
          new_weight = base_weight + learning_rate * |correlation| * base_weight

        Args:
            base_weights: Dict of weight_name -> base_value. If None, uses
                the current aurelius.constants values.
            learning_rate: Fractional adjustment per feedback cycle (default 0.1).

        Returns:
            Dict of adjusted weight_name -> new_value. Total still sums to 1.0.
        """
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

        # Map empirical property names to weight keys
        prop_to_weight: dict[str, str] = {
            "dielectric_proxy": "SCORE_WEIGHT_DIELECTRIC",
            "viscosity_proxy": "SCORE_WEIGHT_VISCOSITY",
            "li_solvation_proxy": "SCORE_WEIGHT_LI_SOLVATION",
        }

        adjusted = dict(base_weights)
        for prop_name, weight_key in prop_to_weight.items():
            r = correlations.get(prop_name, 0.0)
            # Use absolute correlation as importance signal
            importance = abs(r)
            adjusted[weight_key] = base_weights.get(weight_key, 0.1) * (
                1.0 + learning_rate * importance
            )

        # Normalise to sum to 1.0
        total = sum(adjusted.values())
        if total > 0.0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def apply_dynamic_weights(self, pipeline: Any) -> None:
        """Apply dynamically adjusted weights to the pipeline's objectives.

        Updates the SCORE_WEIGHT_* constants in the aurelius.constants module
        and refreshes the pipeline's _OBJECTIVES list so subsequent scoring
        uses the adjusted weights.

        Args:
            pipeline: An AureliusPipeline instance whose objectives will be updated.
        """
        adjusted = self.compute_adjusted_weights()

        # Update module-level constants
        import aurelius.constants as consts
        for key, value in adjusted.items():
            if hasattr(consts, key):
                setattr(consts, key, value)

        # Update pipeline's _OBJECTIVES weights
        if hasattr(pipeline, '_OBJECTIVES'):
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
                self._seen_smiles = set(r.get("smiles", "") for r in data.get("_all_results", []))
                self._empirical_feedback = data.get("_empirical_feedback", [])
                self.active_learning_queue = data.get("active_learning_queue", [])
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
            "_all_results": self._all_results,
            "active_learning_queue": self.active_learning_queue,
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def add_discovery(self, discovery: dict[str, Any] | ScreeningResult) -> None:
        with self._state_lock:
            if isinstance(discovery, dict):
                self.discoveries.append(discovery)
            else:
                self.discoveries.append({
                    "smiles": discovery.smiles,
                    "total_score": discovery.total_score,
                    "is_viable": discovery.is_viable,
                    "rejection_reasons": discovery.rejection_reasons,
                })
            self.discoveries.sort(key=lambda d: d.get("total_score", 0), reverse=True)
            self.discoveries = self.discoveries[:_MAX_DISCOVERIES]

    def add_result(self, sr: Any) -> None:
        with self._state_lock:
            self._seen_smiles.add(sr.smiles)
            self._all_results.append({
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
            })

    def top_scored_smiles(self, divisor: int = 5) -> list[str]:
        scored = [(r["total_score"], r["smiles"]) for r in self._all_results if r["total_score"] > 0]
        scored.sort(key=lambda x: -x[0])
        n = max(5, len(scored) // divisor)
        return [s for _, s in scored[:n]]

    def export_active_learning_queue(self, path: str) -> None:
        """Save the active learning queue SMILES list to a JSON file.

        Args:
            path: Destination file path for the JSON export.
        """
        resolved = _resolve_output_path(path, self.output_dir)
        with open(resolved, "w") as f:
            json.dump(self.active_learning_queue, f, indent=2)
        log.info("Exported active learning queue (%d SMILES) to %s", len(self.active_learning_queue), resolved)

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
        self._all_results.clear()
        self._seen_smiles.clear()
        self._seen_scaffolds.clear()
        self.screened_fingerprints.clear()
        self.active_learning_queue.clear()
        MoleculeContext.from_smiles.cache_clear()
        self.save()


__all__ = [
    "LoopState",
    "check_score_plateau",
    "check_structural_saturation",
]
