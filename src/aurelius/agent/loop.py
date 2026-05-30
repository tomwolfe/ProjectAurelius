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
from dataclasses import dataclass
from typing import Any

import numpy as np

from aurelius.agent.state import ConvergenceChecker, FeedbackAdapter
from aurelius.utils.chem_utils import _is_valid_mol, _safe_mol_from_smiles

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreeningResult:
    """Typed result from a single molecule screening.

    This replaces the previous ``dict[str, Any]`` pattern for internal
    tracking within the discovery loop, providing static type checking
    while still allowing the pipeline's ``dict[str, Any]`` return type
    to remain unchanged at the public API boundary.
    """

    smiles: str
    total_score: float
    sigma_score: float
    desolvation_score: float
    sei_homogeneity_score: float
    mx_synthesis_score: float
    gwp_penalty: float
    is_viable: bool
    rejection_reasons: list[str]


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
        self.all_results: list[ScreeningResult] = []
        self.discoveries: list[ScreeningResult] = []
        self.convergence = ConvergenceChecker()
        self.feedback = FeedbackAdapter()
        self.screened_smiles: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        """Run the discovery loop and return accumulated results.

        The loop implements a Bayesian active-learning cycle:
        1. Propose a large candidate pool via the mutation engine.
        2. Featurise the pool into ECFP4 fingerprints.
        3. If a GP surrogate is already fitted, use Expected Improvement
           to select the top ``batch_size`` candidates.  If not fitted,
           select randomly for the first batch.
        4. Screen ONLY the selected top candidates.
        5. Update the surrogate with the new (X, y) observations.
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

            # ---- Bayesian selection ----
            selected_indices, batch_smiles = self._select_candidates_for_screening(
                valid_candidates, generation
            )

            log.info(
                "Generation %d: Screening %d candidates (%d invalid discarded, %d selected via BO)",
                generation,
                len(valid_candidates),
                invalid_count,
                len(batch_smiles),
            )

            if not batch_smiles:
                continue

            # ---- Screening ----
            batch_scores: list[float] = []
            batch_viable = 0
            batch_discoveries: list[ScreeningResult] = []

            for smi in batch_smiles:
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
                    and score.get("is_viable", False)
                    and len(score.get("rejection_reasons", [])) == 0
                )

                score_data = result.get("score")
                if score_data is None:
                    continue

                screening_result = ScreeningResult(
                    smiles=smi,
                    total_score=score_data.get("total_score", 0.0),
                    sigma_score=score_data.get("sigma_score", 0.0),
                    desolvation_score=score_data.get("desolvation_score", 0.0),
                    sei_homogeneity_score=score_data.get("sei_homogeneity_score", 0.0),
                    mx_synthesis_score=score_data.get("mx_synthesis_score", 0.0),
                    gwp_penalty=score_data.get("gwp_penalty", 0.0),
                    is_viable=score_data.get("is_viable", False),
                    rejection_reasons=score_data.get("rejection_reasons", []),
                )

                if is_discovery:
                    batch_viable += 1
                    discovery_entry = ScreeningResult(
                        smiles=smi,
                        total_score=total_score,
                        sigma_score=score_data.get("sigma_score", 0.0),
                        desolvation_score=score_data.get("desolvation_score", 0.0),
                        sei_homogeneity_score=score_data.get("sei_homogeneity_score", 0.0),
                        mx_synthesis_score=score_data.get("mx_synthesis_score", 0.0),
                        gwp_penalty=score_data.get("gwp_penalty", 0.0),
                        is_viable=True,
                        rejection_reasons=score_data.get("rejection_reasons", []),
                    )
                    batch_discoveries.append(discovery_entry)
                    self.discoveries.append(discovery_entry)
                    self.checkpoint.add_discovery(discovery_entry)
                    log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

                self.all_results.append(screening_result)
                self.feedback.record(screening_result)

            # ---- Convergence / checkpoint ----
            new_clusters = self._count_new_clusters(valid_candidates)
            self.convergence.record_batch(batch_scores, batch_viable, new_clusters)
            self.checkpoint.update_stats(valid_candidates, batch_scores, batch_viable, invalid_count)
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
        top_seeds = self.engine.seed_pool if generation == 1 else self._top_seeds_from_results()
        candidates = self.engine.mutate_batch(top_seeds, self.batch_size * 3)
        return list(candidates)

    def _top_seeds_from_results(self) -> list[str]:
        """Return the top N seeds based on all_results scores."""
        scored = [(r.total_score, r.smiles) for r in self.all_results if r.total_score > 0]
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

        from aurelius.utils.chem_utils import _safe_mol_from_smiles

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

    def _select_candidates_for_screening(
        self,
        valid_candidates: list[str],
        generation: int,
    ) -> tuple[list[int], list[str]]:
        """Select candidates for screening using Bayesian Active Learning.

        If the GP surrogate is already fitted, use Expected Improvement
        to pick the top ``batch_size`` candidates.  If not fitted
        (first generation), select randomly.

        Returns:
            (indices, smiles) — indices into valid_candidates and
            the corresponding SMILES strings.
        """
        if len(valid_candidates) == 0:
            return [], []

        # Featurise the pool
        X_pool = self._featurise_molecules(valid_candidates)

        if self.feedback._surrogate is not None:
            # Fitted surrogate — use Expected Improvement
            ei_scores = self.feedback._surrogate.expected_improvement(X_pool)
            top_indices = np.argsort(ei_scores)[::-1][: self.batch_size]
        else:
            # First batch: random selection
            n = min(self.batch_size, len(valid_candidates))
            indices = self.feedback._rng.choice(len(valid_candidates), size=n, replace=False)
            top_indices = sorted(indices)

        batch_smiles = [valid_candidates[i] for i in top_indices]
        return top_indices, batch_smiles

    def _featurise_molecules(self, smiles_list: list[str]) -> np.ndarray:
        """Convert a list of SMILES to a 2-D fingerprint array.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Array of shape (n, 2048) with ECFP4 fingerprints.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        X = np.zeros((len(smiles_list), 2048), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for idx in fp.GetNonzeroElements():
                X[i][idx] = 1.0
        return X

    def _screen_molecule(self, smiles: str) -> dict[str, Any] | None:
        """Run a single molecule through the screening pipeline.

        Returns the result dict or None on error.
        """
        try:
            result = self.pipeline.screen_molecule(smiles)
        except (ImportError, ValueError, RuntimeError) as e:
            log.warning("Pipeline error for %s: %s", smiles, e)
            return None
        return result if result is not None else None

    def _count_new_clusters(self, candidates: list[str]) -> int:
        """Count how many candidates represent new structural clusters.

        Uses Tanimoto similarity thresholds to count unique scaffolds.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        unique_scaffolds: set[str] = set()
        for smi in candidates:
            mol = _safe_mol_from_smiles(smi)
            if mol is None:
                continue
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitIdToSize=2048)  # type: ignore[attr-defined, unused-ignore]
            except (AttributeError, RuntimeError):
                continue
            scaffold = Chem.MolFragmentToSmiles(mol, atomsToUse=list(range(mol.GetNumAtoms())), isomericSmiles=False)
            has_new = False
            for existing in self.screened_smiles:
                existing_mol = _safe_mol_from_smiles(existing)
                if existing_mol is None:
                    continue
                try:
                    existing_fp = AllChem.GetMorganFingerprintAsBitVec(existing_mol, 2, bitIdToSize=2048)  # type: ignore[attr-defined, unused-ignore]
                    from rdkit.DataStructs import TanimotoSimilarity

                    if TanimotoSimilarity(fp, existing_fp) > 0.75:
                        has_new = True
                        break
                except (AttributeError, RuntimeError):
                    continue
            if not has_new:
                unique_scaffolds.add(scaffold)
        return len(unique_scaffolds)
