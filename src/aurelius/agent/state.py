"""Agent state management: checkpointing, convergence tracking, and feedback."""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


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

    Replaces the previous SQLite-based implementation with a simpler
    dictionary-backed approach that writes to a JSON file.

    Uses atomic writes (write to .tmp, then os.replace) to prevent
    corruption during crashes.
    """

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

        self._screened_smiles: set[str] = set()
        self._screened_entries: dict[str, tuple[float, str]] = {}
        self._fingerprints: set[str] = set()
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
                self._screened_smiles = set(data.get("screened_smiles", []))
                self._screened_entries = {
                    entry["smiles"]: (entry["score"], entry["tier_status"])
                    for entry in data.get("screened_molecules", [])
                }
                self._fingerprints = set(data.get("fingerprints", []))
                self._discoveries = data.get("discoveries", [])
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

    def _save(self) -> None:
        """Save checkpoint state to the JSON file atomically.

        Writes to a temporary file first, then uses os.replace() for
        atomic safety.
        """
        data = {
            "batch": self._batch,
            "screened_count": self._screened_count,
            "best_score": self._best_score,
            "viable_count": self._viable_count,
            "total_generated": self._total_generated,
            "invalid_discarded": self._invalid_discarded,
            "started_at": self._started_at,
            "last_updated": self._last_updated,
            "screened_smiles": list(self._screened_smiles),
            "screened_molecules": [
                {"smiles": smi, "score": score, "tier_status": status}
                for smi, (score, status) in self._screened_entries.items()
            ],
            "fingerprints": list(self._fingerprints),
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
        """Save checkpoint state to the JSON file atomically.

        Writes to a temporary file first, then uses os.replace() for
        atomic safety.
        """
        self._save()

    def add_discovery(self, discovery: dict[str, Any]) -> None:
        """Add a discovery to the checkpoint.

        Args:
            discovery: Dict with discovery data.
        """
        self._discoveries.append(discovery)

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

    def is_screened(self, smiles: str) -> bool:
        """Check if a SMILES string has already been screened.

        Args:
            smiles: SMILES string to check.

        Returns:
            True if the molecule has already been screened.
        """
        return smiles in self._screened_smiles

    def add_screened_molecule(self, smiles: str, score: float, tier_status: str) -> None:
        """Record a screened molecule.

        Args:
            smiles: SMILES string.
            score: Molecule score.
            tier_status: Status string (e.g. "viable", "rejected").
        """
        self._screened_smiles.add(smiles)
        self._screened_entries[smiles] = (score, tier_status)

    def fps_hex_list(self) -> list[str]:
        """Return list of known fingerprint hex strings."""
        return list(self._fingerprints)

    def add_fps_hex(self, hex_str: str, smiles: str | None = None) -> None:
        """Add a fingerprint hex string to the known list.

        Args:
            hex_str: Serialized fingerprint string.
            smiles: Associated SMILES (for index lookup).
        """
        self._fingerprints.add(hex_str)

    def is_known_fps(self, hex_str: str) -> bool:
        """Check if a fingerprint hex has already been recorded.

        Args:
            hex_str: Serialized fingerprint hex string.

        Returns:
            True if the fingerprint is already known.
        """
        return hex_str in self._fingerprints

    def clear(self) -> None:
        """Clear all screened molecules, fingerprints, and agent state."""
        self._batch = 0
        self._screened_count = 0
        self._best_score = 0.0
        self._viable_count = 0
        self._total_generated = 0
        self._invalid_discarded = 0
        self._screened_smiles.clear()
        self._screened_entries.clear()
        self._fingerprints.clear()
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

    Uses Welford's online algorithm for computing running variance
    without storing all scores, bounding memory usage.
    """

    def __init__(self) -> None:
        """Initialize the convergence checker."""
        self.all_scores: deque[float] = deque(maxlen=10000)
        self.batch_scores: list[list[float]] = []
        self.viability_rates: list[float] = []
        self.new_clusters_per_batch: list[int] = []
        self.viable_count = 0
        self.total_screened = 0
        self.generations = 0

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
        self.all_scores.extend(scores)
        self.batch_scores.append(scores)
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1

        viable_in_batch = sum(1 for s in scores if s >= 65.0)
        self.viability_rates.append(viable_in_batch / max(len(scores), 1))
        self.new_clusters_per_batch.append(new_clusters)

        # Update Welford's online statistics
        for score in scores:
            self._welford_n += 1
            delta = score - self._welford_mean
            self._welford_mean += delta / self._welford_n
            self._welford_m2 += delta * (score - self._welford_mean)

    def compute_rolling_mean(self, batch_size: int = 50) -> list[float]:
        """Rolling mean of total_score over windows of `batch_size`.

        Args:
            batch_size: Window size for the rolling mean.

        Returns:
            List of rolling mean values.
        """
        if len(self.all_scores) < batch_size:
            return []
        n_batches = len(self.all_scores) // batch_size
        scores_list = list(self.all_scores)
        rolling = [float(np.mean(scores_list[i * batch_size : (i + 1) * batch_size])) for i in range(n_batches)]
        return rolling

    def check_score_plateau(self) -> bool:
        """Check if rolling mean changes < 1.0% over 3 consecutive batches.

        Returns:
            True if the score has plateaued.
        """
        rolling = self.compute_rolling_mean(batch_size=50)
        if len(rolling) < 3:
            return False
        last_three = rolling[-3:]
        for i in range(1, 3):
            ref = last_three[i - 1]
            if ref == 0:
                return False
            change = abs(last_three[i] - ref) / abs(ref)
            if change >= 0.01:
                return False
        return True

    def check_pass_rate_collapsed(self) -> bool:
        """Check if viability rate < 3% for 2 consecutive batches.

        Returns:
            True if pass rate has collapsed.
        """
        if len(self.viability_rates) < 2:
            return False
        return self.viability_rates[-1] < 0.03 and self.viability_rates[-2] < 0.03

    def check_structural_saturation(self) -> bool:
        """Check if < 3 new clusters over last 2 batches.

        Returns:
            True if structural saturation is detected.
        """
        if len(self.new_clusters_per_batch) < 2:
            return False
        return self.new_clusters_per_batch[-1] < 3 and self.new_clusters_per_batch[-2] < 3

    def check_volume_requirement(self) -> bool:
        """Check if >= 150 viable OR >= 300 total unique screened.

        Returns:
            True if volume requirement is met.
        """
        return self.viable_count >= 150 or self.total_screened >= 300

    def should_terminate(self) -> tuple[bool, str]:
        """Determine if the screening loop should terminate.

        Returns:
            Tuple of (should_terminate, reason_string).
        """
        if not self.check_volume_requirement():
            return False, "Volume threshold not met"
        plateau = self.check_score_plateau()
        pass_collapsed = self.check_pass_rate_collapsed()
        saturation = self.check_structural_saturation()
        if plateau and pass_collapsed and saturation:
            return True, "All convergence criteria met"
        reasons = []
        if not plateau:
            reasons.append("score plateau")
        if not pass_collapsed:
            reasons.append("pass rate not collapsed")
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


class ActiveLearningOracle:
    """Active learning oracle with basic caching and query capabilities.

    Provides a query interface for selecting the most informative
    molecules to screen next, using uncertainty sampling.

    Raises:
        NotImplementedError: The oracle is not implemented. Use the
            screening pipeline directly for real viability scores.
    """

    def __init__(self) -> None:
        """Initialize the active learning oracle.

        Raises:
            NotImplementedError: The oracle is not implemented.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )

    def query(self, smiles: str) -> float:
        """Query the oracle for a molecule's predicted viability.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )

    def query_batch(self, smiles_list: list[str]) -> list[float]:
        """Query multiple molecules, returning a list of scores.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )

    def add_to_pool(self, smiles: str, score: float) -> None:
        """Add a molecule to the candidate pool.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )

    def select_most_uncertain(self, top_k: int = 10) -> list[str]:
        """Select the k most uncertain molecules from the pool.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )

    def clear(self) -> None:
        """Clear all cached data and pool.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "ActiveLearningOracle is not implemented. Use the screening pipeline directly for real viability scores."
        )


class FeedbackAdapter:
    """Adapts mutation strategy based on rejection patterns."""

    def __init__(self) -> None:
        """Initialize the feedback adapter."""
        self.tier1_fails = 0
        self.tier2_fails = 0
        self.tier3_low_homogeneity = 0
        self.total_screened = 0
        self.rationale_log: list[str] = []

    def record(self, result: dict[str, Any]) -> None:
        """Record screening result for feedback analysis.

        Args:
            result: Dict with score and viability information.
        """
        score = result.get("score")
        if score is None:
            return
        self.total_screened += 1
        if not score.tier1_viable:
            self.tier1_fails += 1
            self.rationale_log.append(
                f"Tier 1 fail for {score.molecule_smiles}: Lower MW, add polar groups, reduce F-density"
            )
        if not score.tier2_viable:
            self.tier2_fails += 1
            self.rationale_log.append(
                f"Tier 2 fail for {score.molecule_smiles}: "
                "Reduce steric bulk near coordination sites, lower desolvation barrier"
            )
        if score.tier3_viable and score.sei_homogeneity_score < 50.0:
            self.tier3_low_homogeneity += 1
            self.rationale_log.append(
                f"Low SEI homogeneity for {score.molecule_smiles}: Add unsaturation/boron, increase F/C ratio"
            )

    def get_adaptation_strategy(self) -> dict[str, Any]:
        """Return current mutation adaptation recommendations.

        Returns:
            Dict with strategy recommendation and fail rates.
        """
        strategy: dict[str, Any] = {
            "total_screened": self.total_screened,
            "tier1_fail_rate": self.tier1_fails / max(self.total_screened, 1),
            "tier2_fail_rate": self.tier2_fails / max(self.total_screened, 1),
            "tier3_low_homogeneity_rate": self.tier3_low_homogeneity / max(self.total_screened, 1),
        }
        if strategy["tier1_fail_rate"] > 0.5:
            strategy["recommendation"] = "Prioritize MW reduction and polar group addition"
        elif strategy["tier2_fail_rate"] > 0.5:
            strategy["recommendation"] = "Reduce steric bulk, focus on small molecules"
        elif strategy["tier3_low_homogeneity_rate"] > 0.5:
            strategy["recommendation"] = "Add unsaturation and boron-containing groups"
        else:
            strategy["recommendation"] = "Continue current mutation strategy"
        return strategy

    def write_rationale_log(self, path: str = "mutation_rationale.md", output_dir: str | Path | None = None) -> None:
        """Write accumulated rationale to markdown file.

        Args:
            path: Output file path (relative to output_dir).
            output_dir: Directory to write to. If None, uses current working directory.
        """
        import logging
        from datetime import UTC

        log = logging.getLogger("aurelius_agent")

        path = _resolve_output_path(path, output_dir)

        with open(path, "w") as f:
            f.write("# Mutation Rationale Log\n\n")
            f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")
            f.write("## Adaptation Decisions\n\n")
            for entry in self.rationale_log:
                f.write(f"- {entry}\n")
            f.write("\n## Strategy Summary\n\n")
            strategy = self.get_adaptation_strategy()
            f.write(f"- **Recommendation:** {strategy['recommendation']}\n")
            f.write(f"- **Tier 1 fail rate:** {strategy['tier1_fail_rate']:.2%}\n")
            f.write(f"- **Tier 2 fail rate:** {strategy['tier2_fail_rate']:.2%}\n")
            f.write(f"- **Tier 3 low homogeneity rate:** {strategy['tier3_low_homogeneity_rate']:.2%}\n")
        log.info("Mutation rationale log written to %s", path)
