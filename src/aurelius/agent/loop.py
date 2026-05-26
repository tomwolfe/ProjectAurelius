"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), screening, feedback, convergence checking, and
checkpointing.  It is designed as a pure, testable component that
receives its dependencies (pipeline, mutation engine, checkpoint manager,
etc.) rather than constructing them internally.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class DiscoveryLoop:
    """Main autonomous screening loop.

    The loop runs for *max_generations* iterations (or until a time
    cap / convergence condition is met).  Each generation:
    1. Mutates seed molecules via the mutation engine
    2. Filters invalid / duplicate candidates
    3. Screens candidates through the pipeline
    4. Records results / feedback / convergence state
    5. Saves checkpoint at the end of each generation
    """

    def __init__(
        self,
        pipeline: Any,
        engine: Any,
        checkpoint: Any,
        max_generations: int = 50,
        batch_size: int = 50,
        max_wall_time: float = 43200.0,  # 12 hours
    ) -> None:
        """Initialise the discovery loop.

        Args:
            pipeline: AureliusPipeline (or equivalent) instance.
            engine: MutationEngine instance.
            checkpoint: CheckpointManager instance.
            max_generations: Maximum number of generations to run.
            batch_size: Number of candidates per batch.
            max_wall_time: Wall-clock time cap in seconds.
        """
        self.pipeline = pipeline
        self.engine = engine
        self.checkpoint = checkpoint
        self.max_generations = max_generations
        self.batch_size = batch_size
        self.max_wall_time = max_wall_time

        # Internal state accumulated during the loop
        self.total_screened = 0
        self.total_viable = 0
        self.total_invalid = 0
        self.all_results: list[dict[str, Any]] = []
        self.discoveries: list[dict[str, Any]] = []
        self.convergence = _ConvergenceTracker()
        self.feedback = _FeedbackAdapter()
        self.screened_smiles: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        """Run the discovery loop and return accumulated results.

        Returns:
            A dict with keys ``all_results`` and ``discoveries``.
        """
        wall_start = time.time()
        generation = 0

        while generation < self.max_generations:
            elapsed = time.time() - wall_start
            if elapsed > self.max_wall_time:
                log.info("Time cap reached (%.0fs). Exiting loop.", elapsed)
                break

            generation += 1

            # ---- Generation ----
            candidates = self._generate_candidates(generation)

            # ---- Filtering ----
            valid_candidates, invalid_count = self._filter_candidates(candidates)

            if not valid_candidates:
                log.info("Generation %d: No valid candidates. Skipping.", generation)
                continue

            log.info(
                "Generation %d: Screening %d candidates (%d invalid discarded)",
                generation,
                len(valid_candidates),
                invalid_count,
            )

            # ---- Screening ----
            batch_scores: list[float] = []
            batch_viable = 0
            batch_discoveries: list[dict[str, Any]] = []

            for smi in valid_candidates:
                result = self._screen_molecule(smi)
                if result is None:
                    continue

                score = result.get("score")
                if score is None:
                    continue

                self.screened_smiles.add(smi)
                self.engine.add_to_db(smi)

                total_score = score.total_score
                batch_scores.append(total_score)

                is_discovery = (
                    total_score >= 65.0
                    and score.tier1_viable
                    and score.tier2_viable
                    and score.tier3_viable
                    and len(score.rejection_reasons) == 0
                )

                if is_discovery:
                    batch_viable += 1
                    discovery_entry = {
                        "smiles": smi,
                        "total_score": total_score,
                        "sigma": score.sigma_score,
                        "desolvation": score.desolvation_score,
                        "sei_homogeneity": score.sei_homogeneity_score,
                        "mx_synthesis": score.mx_synthesis_score,
                        "gwp_penalty": score.gwp_penalty,
                        "is_viable": True,
                        "rejection_reasons": score.rejection_reasons,
                        "components": score.rejection_reasons,
                    }
                    batch_discoveries.append(discovery_entry)
                    self.discoveries.append(discovery_entry)
                    self.checkpoint.add_discovery(discovery_entry)
                    log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

                self.all_results.append(result)
                self.feedback.record(result)

            # ---- Convergence / checkpoint ----
            self.convergence.record_batch(batch_scores, batch_viable, invalid_count)
            self.checkpoint.update_stats(
                valid_candidates, batch_scores, batch_viable, invalid_count
            )
            self.total_screened += len(valid_candidates)
            self.total_viable += batch_viable
            self.total_invalid += invalid_count

            log.info(
                "  Generation %d complete: %d screened, %d viable, best=%.1f",
                generation,
                len(valid_candidates),
                batch_viable,
                max(batch_scores) if batch_scores else 0,
            )

            strategy = self.feedback.get_adaptation_strategy()
            if generation % 5 == 0:
                log.info("  [Feedback] Strategy: %s", strategy.get("recommendation", ""))

            should_stop, reason = self.convergence.should_terminate()
            if should_stop:
                log.info("Convergence reached: %s", reason)
                break

            # ---- Atomic checkpoint save ----
            self.checkpoint.save()

        # ---- Post-loop: build summary ----
        return {
            "all_results": self.all_results,
            "discoveries": self.discoveries,
            "total_screened": self.total_screened,
            "total_viable": self.total_viable,
            "total_invalid": self.total_invalid,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidates(self, generation: int) -> list[str]:
        """Generate candidate SMILES for this generation.

        First generation uses all seed molecules; later generations use
        the top-scoring seeds from previous results.
        """
        top_seeds = (
            self.engine.seed_pool if generation == 1 else self._top_seeds_from_results()
        )
        return self.engine.mutate_batch(top_seeds, self.batch_size * 3)

    def _top_seeds_from_results(self) -> list[str]:
        """Return the top N seeds based on all_results scores."""
        scored = [
            (r["score"].total_score, r["score"].molecule_smiles)
            for r in self.all_results
            if r.get("score")
        ]
        scored.sort(key=lambda x: -x[0])
        n = max(5, len(scored) // 5)
        return [s for _, s in scored[:n]]

    def _filter_candidates(self, candidates: list[str]) -> tuple[list[str], int]:
        """Filter out invalid / already-screened candidates.

        Returns:
            (valid_candidates, invalid_count) tuple.
        """
        valid: list[str] = []
        invalid_count = 0

        from aurelius.utils.chem_utils import _is_valid_mol, _safe_mol_from_smiles

        for smi in candidates:
            if smi in self.screened_smiles:
                invalid_count += 1
                continue
            mol = _safe_mol_from_smiles(smi)
            if mol is None:
                invalid_count += 1
                continue
            if not _is_valid_mol(mol):
                invalid_count += 1
                continue
            valid.append(smi)

        if len(valid) > self.batch_size:
            valid = valid[: self.batch_size]

        return valid, invalid_count

    def _screen_molecule(self, smiles: str) -> dict[str, Any] | None:
        """Run a single molecule through the screening pipeline.

        Returns the result dict or None on error.
        """
        try:
            result = self.pipeline.screen_molecule(smiles)
        except Exception as e:
            log.error("Pipeline error for %s: %s", smiles, e, exc_info=True)
            return None
        return result


# ------------------------------------------------------------------
# Lightweight convergence & feedback helpers (internal to loop.py)
# ------------------------------------------------------------------

class _ConvergenceTracker:
    """Tracks convergence metrics across generations."""

    def __init__(self) -> None:
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
        self.all_scores.extend(scores)
        self.batch_scores.append(scores)
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1
        viable_in_batch = sum(1 for s in scores if s >= 65.0)
        self.viability_rates.append(viable_in_batch / max(len(scores), 1))
        self.new_clusters_per_batch.append(new_clusters)

    def should_terminate(self) -> tuple[bool, str]:
        """Determine if the loop should terminate.

        Returns:
            (should_terminate, reason) tuple.
        """
        if self.viable_count < 150 and self.total_screened < 300:
            return False, "Volume threshold not met"
        plateau = self._check_plateau()
        pass_collapsed = self._check_pass_rate_collapsed()
        saturation = self._check_saturation()
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

    def _check_plateau(self) -> bool:
        rolling = self._rolling_mean()
        if len(rolling) < 3:
            return False
        last = rolling[-3:]
        for i in range(1, 3):
            ref = last[i - 1]
            if ref == 0:
                return False
            change = abs(last[i] - ref) / abs(ref)
            if change >= 0.01:
                return False
        return True

    def _check_pass_rate_collapsed(self) -> bool:
        if len(self.viability_rates) < 2:
            return False
        return (
            self.viability_rates[-1] < 0.03
            and self.viability_rates[-2] < 0.03
        )

    def _check_saturation(self) -> bool:
        if len(self.new_clusters_per_batch) < 2:
            return False
        return (
            self.new_clusters_per_batch[-1] < 3
            and self.new_clusters_per_batch[-2] < 3
        )

    def _rolling_mean(self, window: int = 50) -> list[float]:
        if len(self.all_scores) < window:
            return []
        n = len(self.all_scores) // window
        return [
            sum(self.all_scores[i * window : (i + 1) * window]) / window
            for i in range(n)
        ]


class _FeedbackAdapter:
    """Tracks rejection patterns and returns adaptation strategy."""

    def __init__(self) -> None:
        self.tier1_fails = 0
        self.tier2_fails = 0
        self.tier3_low_homogeneity = 0
        self.total_screened = 0

    def record(self, result: dict[str, Any]) -> None:
        score = result.get("score")
        if score is None:
            return
        self.total_screened += 1
        if not score.tier1_viable:
            self.tier1_fails += 1
        if not score.tier2_viable:
            self.tier2_fails += 1
        if score.tier3_viable and score.sei_homogeneity_score < 50.0:
            self.tier3_low_homogeneity += 1

    def get_adaptation_strategy(self) -> dict[str, Any]:
        t1_rate = self.tier1_fails / max(self.total_screened, 1)
        t2_rate = self.tier2_fails / max(self.total_screened, 1)
        t3_rate = self.tier3_low_homogeneity / max(self.total_screened, 1)
        strategy: dict[str, Any] = {
            "total_screened": self.total_screened,
            "tier1_fail_rate": t1_rate,
            "tier2_fail_rate": t2_rate,
            "tier3_low_homogeneity_rate": t3_rate,
        }
        if t1_rate > 0.5:
            strategy["recommendation"] = "Prioritize MW reduction and polar group addition"
        elif t2_rate > 0.5:
            strategy["recommendation"] = "Reduce steric bulk, focus on small molecules"
        elif t3_rate > 0.5:
            strategy["recommendation"] = "Add unsaturation and boron-containing groups"
        else:
            strategy["recommendation"] = "Continue current mutation strategy"
        return strategy
