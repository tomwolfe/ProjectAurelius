"""Agent state management: checkpointing, convergence tracking, and feedback."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

import numpy as np


class CheckpointManager:
    """Manages agent state using a lightweight embedded SQLite database.

    Replaces the previous JSON-based checkpoint system with indexed
    SQL tables for ``screened_molecules`` and ``fingerprints``, enabling
    O(log N) duplicate detection instead of O(N) list scans.

    Uses atomic writes to prevent corruption during crashes.
    Saves state after every molecule (not only per batch) for granular
    checkpointing.
    """

    def __init__(self, path: str = "aurelius_state.db") -> None:
        """Initialize the checkpoint manager.

        Args:
            path: Path to the SQLite database file.
        """
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS screened_molecules (
                smiles  TEXT PRIMARY KEY,
                score   REAL,
                tier_status TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_smiles ON screened_molecules(smiles)")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                smiles  TEXT,
                fp_hex  TEXT,
                PRIMARY KEY (smiles, fp_hex)
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_hex ON fingerprints(fp_hex)")

        self._batch = 0
        self._screened_count = 0
        self._best_score = 0.0
        self._viable_count = 0
        self._total_generated = 0
        self._invalid_discarded = 0
        self._discoveries: list[dict[str, Any]] = []
        self._started_at = datetime.now(UTC).isoformat()
        self._last_updated: str | None = None

    def _load_state(self) -> dict[str, Any]:
        """Load checkpoint state from SQLite database.

        Returns:
            Dict of checkpoint state.
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

    def load(self) -> dict[str, Any]:
        """Load checkpoint state from disk.

        Returns:
            Dict of checkpoint state.  Returns empty state if file
            cannot be read.
        """
        return self._load_state()

    def save(self) -> None:
        """Save checkpoint state to SQLite database.

        Updates the in-memory state and persists it to the SQLite database.
        """
        state = self._get_state_dict()
        self._conn.execute(
            "INSERT OR REPLACE INTO screened_molecules (smiles, score, tier_status) "
            "VALUES (?, ?, ?)",
            ("__checkpoint__", str(state.get("screened_count", 0)), str(state.get("batch", 0))),
        )
        self._conn.commit()

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
        cursor = self._conn.execute(
            "SELECT 1 FROM screened_molecules WHERE smiles = ?", (smiles,)
        )
        return cursor.fetchone() is not None

    def add_screened_molecule(self, smiles: str, score: float, tier_status: str) -> None:
        """Record a screened molecule.

        Args:
            smiles: SMILES string.
            score: Molecule score.
            tier_status: Status string (e.g. "viable", "rejected").
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO screened_molecules (smiles, score, tier_status) VALUES (?, ?, ?)",
            (smiles, score, tier_status),
        )

    def fps_hex_list(self) -> list[str]:
        """Return list of known fingerprint hex strings."""
        cursor = self._conn.execute("SELECT fp_hex FROM fingerprints")
        return [row[0] for row in cursor]

    def add_fps_hex(self, hex_str: str, smiles: str | None = None) -> None:
        """Add a fingerprint hex string to the known list.

        Args:
            hex_str: Serialized fingerprint string.
            smiles: Associated SMILES (for index lookup).
        """
        if smiles is None:
            self._conn.execute(
                "INSERT OR IGNORE INTO fingerprints (fp_hex) VALUES (?)",
                (hex_str,),
            )
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO fingerprints (smiles, fp_hex) VALUES (?, ?)",
                (smiles, hex_str),
            )

    def is_known_fps(self, hex_str: str) -> bool:
        """Check if a fingerprint hex has already been recorded.

        Args:
            hex_str: Serialized fingerprint hex string.

        Returns:
            True if the fingerprint is already known.
        """
        cursor = self._conn.execute(
            "SELECT 1 FROM fingerprints WHERE fp_hex = ?", (hex_str,)
        )
        return cursor.fetchone() is not None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def clear(self) -> None:
        """Clear all screened molecules and fingerprints."""
        self._conn.execute("DELETE FROM screened_molecules")
        self._conn.execute("DELETE FROM fingerprints")
        self._conn.commit()


class ConvergenceChecker:
    """Evaluates whether the screening loop should terminate."""

    def __init__(self) -> None:
        """Initialize the convergence checker."""
        self.all_scores: list[float] = []
        self.batch_scores: list[list[float]] = []
        self.viability_rates: list[float] = []
        self.new_clusters_per_batch: list[int] = []
        self.viable_count = 0
        self.total_screened = 0
        self.generations = 0

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

    def compute_rolling_mean(self, batch_size: int = 50) -> list[float]:
        """Rolling mean of total_score over windows of `batch_size`.

        Args:
            batch_size: Window size for the rolling mean.

        Returns:
            List of rolling mean values.
        """
        if len(self.all_scores) < batch_size:
            return []
        rolling: list[float] = []
        for i in range(batch_size, len(self.all_scores) + 1, batch_size):
            window = self.all_scores[i - batch_size : i]
            rolling.append(float(np.mean(window)))
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
        """Compute the variance of all recorded scores.

        Returns:
            Variance of all scores.
        """
        if len(self.all_scores) < 2:
            return 0.0
        return float(np.var(self.all_scores))


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

    def write_rationale_log(self, path: str = "mutation_rationale.md") -> None:
        """Write accumulated rationale to markdown file.

        Args:
            path: Path to write the rationale log.
        """
        import logging
        from datetime import UTC

        log = logging.getLogger("aurelius_agent")

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
