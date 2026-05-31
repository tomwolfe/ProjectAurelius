"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), screening, feedback, convergence checking, and
checkpointing.  It is designed as a pure, testable component that
receives its dependencies (pipeline, mutation engine, checkpoint manager,
etc.) rather than constructing them internally.

SMILES strings are parsed into RDKit Mol objects **exactly once** per
molecule per generation via ``MoleculeContext``, and the parsed object
is reused across Filter, Oracle, and Featurizer stages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.spatial.distance import jaccard
from sklearn.cluster import MiniBatchKMeans

from aurelius.agent.state import ConvergenceChecker, FeedbackAdapter
from aurelius.constants import DISCOVERY_THRESHOLD
from aurelius.types import MoleculeContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreeningResult:
    """Result from a single molecule screening.

    Contains the unified ``total_score`` and multi-objective sub-scores
    for downstream analysis and SDF export.
    """

    smiles: str
    total_score: float
    is_viable: bool
    rejection_reasons: list[str]
    fingerprint: np.ndarray[Any, Any] | None = None
    novelty_to_seed: float | None = None
    homo_eV: float | None = None
    lumo_eV: float | None = None
    dielectric_proxy: float | None = None
    viscosity_proxy: float | None = None
    sa_score: float | None = None
    sub_scores: dict[str, float] | None = None


class DiscoveryLoop:
    """Main autonomous screening loop.

    The loop runs for *max_generations* iterations (or until a time
    cap / convergence condition is met).  Each generation:
    1. Mutates seed molecules via the mutation engine
    2. Filters invalid / duplicate candidates (parses to MoleculeContext)
    3. Featurises candidates from MoleculeContext pre-computed vectors
    4. Selects top candidates via Bayesian Expected Improvement
    5. Screens selected candidates through the pipeline (reuses Mol)
    6. Records results / feedback / convergence state
    7. Saves checkpoint at the end of each generation
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
        self._prev_centroids: np.ndarray[Any, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        """Run the discovery loop and return accumulated results.

        The loop implements a Bayesian active-learning cycle:
        1. Propose a large candidate pool via the mutation engine.
        2. Parse SMILES -> MoleculeContext (Mol parsed ONCE here).
        3. If a RF surrogate is already fitted, use Expected Improvement
           to select the top ``batch_size`` candidates.  If not fitted,
           select randomly for the first batch.
        4. Screen ONLY the selected top candidates (reuses pre-parsed Mol).
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

            # ---- Parse SMILES to Mol ONCE, filter invalid ----
            valid_contexts, invalid_count = self._filter_candidates(candidates)

            if not valid_contexts:
                log.info("Generation %d: No valid candidates. Skipping.", generation)
                continue

            # ---- Bayesian selection (uses ECFP4 from MoleculeContext) ----
            selected_indices, batch_contexts = self._select_candidates_for_screening(
                valid_contexts, generation
            )

            log.info(
                "Generation %d: Screening %d candidates (%d invalid discarded, %d selected via BO)",
                generation,
                len(valid_contexts),
                invalid_count,
                len(batch_contexts),
            )

            if not batch_contexts:
                continue

            # ---- Screening (passes MoleculeContext to avoid re-parsing) ----
            batch_scores: list[float] = []
            batch_viable = 0
            batch_discoveries: list[ScreeningResult] = []
            batch_feature_vectors: list[np.ndarray] = []

            for ctx in batch_contexts:
                result = self._screen_molecule(ctx)
                if result is None:
                    continue

                score_data = result.get("score")
                if score_data is None:
                    continue

                smi = ctx.smiles
                self.screened_smiles.add(smi)
                self.engine.add_to_db(smi)

                total_score = score_data.get("total_score", 0.0)
                batch_scores.append(total_score)

                # Feature vector from MoleculeContext (already computed)
                fv = ctx.get_feature_vector()
                batch_feature_vectors.append(fv)

                # Novelty to seed
                novelty: float | None = None
                rdkit_fp = ctx.get_ecfp4()
                seed_fps = getattr(self.engine, "seed_fingerprints", None)
                if isinstance(seed_fps, list) and seed_fps:
                    sims = BulkTanimotoSimilarity(rdkit_fp, seed_fps)
                    novelty = 1.0 - max(sims) if sims else None

                # Multi-objective sub-scores from pipeline result
                t2 = result.get("tier2", {}) or {}
                sub_scores = score_data.get("sub_scores", {})
                screening_result = ScreeningResult(
                    smiles=smi,
                    total_score=total_score,
                    is_viable=score_data.get("is_viable", False),
                    rejection_reasons=score_data.get("rejection_reasons", []),
                    fingerprint=fv,
                    novelty_to_seed=novelty,
                    homo_eV=t2.get("homo_eV"),
                    lumo_eV=t2.get("lumo_eV"),
                    dielectric_proxy=t2.get("dielectric_proxy"),
                    viscosity_proxy=t2.get("viscosity_proxy"),
                    sa_score=score_data.get("sa_score"),
                    sub_scores=sub_scores,
                )

                is_discovery = (
                    total_score >= DISCOVERY_THRESHOLD
                    and score_data.get("is_viable", False)
                    and len(score_data.get("rejection_reasons", [])) == 0
                )

                if is_discovery:
                    batch_viable += 1
                    discovery_entry = ScreeningResult(
                        smiles=smi,
                        total_score=total_score,
                        is_viable=True,
                        rejection_reasons=score_data.get("rejection_reasons", []),
                        fingerprint=fv,
                        novelty_to_seed=novelty,
                        homo_eV=t2.get("homo_eV"),
                        lumo_eV=t2.get("lumo_eV"),
                        dielectric_proxy=t2.get("dielectric_proxy"),
                        viscosity_proxy=t2.get("viscosity_proxy"),
                        sa_score=score_data.get("sa_score"),
                        sub_scores=sub_scores,
                    )
                    batch_discoveries.append(discovery_entry)
                    self.discoveries.append(discovery_entry)
                    self.checkpoint.add_discovery(discovery_entry)
                    log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

                self.all_results.append(screening_result)
                self.feedback.record(screening_result)

            # ---- Close the active-learning loop: retrain RF surrogate ----
            self.feedback.finalize_batch()

            # ---- Seed pool evolution: feed back high-scoring molecules ----
            new_seeds = [
                ctx.smiles for ctx, sc in zip(batch_contexts, batch_scores, strict=False)
                if sc >= 65.0
            ]
            if new_seeds:
                existing = set(self.engine.seed_pool)
                for smi in new_seeds:
                    if smi not in existing:
                        self.engine.seed_pool.append(smi)
                        existing.add(smi)
                # Cap at 200: keep the most recent (highest-scoring) seeds
                if len(self.engine.seed_pool) > 200:
                    self.engine.seed_pool = self.engine.seed_pool[-200:]
            self.convergence.seed_pool_size = len(self.engine.seed_pool)

            # ---- Convergence / checkpoint ----
            new_clusters = self._count_new_clusters_from_contexts(valid_contexts)
            self.convergence.record_batch(batch_scores, batch_viable, new_clusters)
            self.checkpoint.update_stats(
                [c.smiles for c in batch_contexts],
                batch_scores, batch_viable, invalid_count,
            )
            self.total_screened += len(valid_contexts)
            self.total_viable += batch_viable
            self.total_invalid += invalid_count

            # ---- Diversity tracking ----
            mean_div = self._compute_mean_pairwise_tanimoto(batch_feature_vectors)
            if mean_div is not None:
                log.info("  Generation %d: mean pairwise Tanimoto diversity = %.4f", generation, mean_div)

            log.info(
                "  Generation %d complete: %d screened, %d viable, best=%.1f",
                generation,
                len(valid_contexts),
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

    def _filter_candidates(
        self,
        candidates: list[str],
    ) -> tuple[list[MoleculeContext], int]:
        """Parse SMILES into MoleculeContext and filter invalid/duplicate.

        This is the single point where SMILES -> Mol parsing occurs per
        generation. The returned MoleculeContext objects are reused by
        all subsequent pipeline stages.

        Returns:
            (valid_contexts, invalid_count) tuple.
        """
        valid_contexts: list[MoleculeContext] = []
        invalid_count = 0

        for smi in candidates:
            if smi in self.screened_smiles:
                invalid_count += 1
                continue
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                invalid_count += 1
                continue
            from aurelius.utils.chem_utils import _is_valid_mol
            if not _is_valid_mol(ctx.mol):
                invalid_count += 1
                continue
            valid_contexts.append(ctx)

        if len(valid_contexts) > self.batch_size:
            valid_contexts = valid_contexts[: self.batch_size]

        return valid_contexts, invalid_count

    def _select_candidates_for_screening(
        self,
        valid_contexts: list[MoleculeContext],
        generation: int,
    ) -> tuple[list[int], list[MoleculeContext]]:
        """Select candidates for screening using Bayesian Active Learning.

        If the RF surrogate is already fitted, use Expected Improvement
        to pick the top ``batch_size`` candidates.  If not fitted
        (first generation), select randomly.

        Featurisation uses the pre-computed feature vectors from
        MoleculeContext (no redundant SMILES -> Mol parsing).

        Returns:
            (indices, contexts) — indices into valid_contexts and
            the corresponding MoleculeContext objects.
        """
        if len(valid_contexts) == 0:
            return [], []

        # Featurise from MoleculeContext pre-computed vectors
        X_pool = self._featurise_from_contexts(valid_contexts)

        if self.feedback._surrogate is not None:
            ei_scores = self.feedback._surrogate.expected_improvement(X_pool)
            top_indices: list[int] = np.argsort(ei_scores)[::-1][: self.batch_size].tolist()
        else:
            # First batch: random selection
            n = min(self.batch_size, len(valid_contexts))
            indices = self.feedback._rng.choice(len(valid_contexts), size=n, replace=False)
            top_indices = sorted(indices.tolist())

        batch_contexts = [valid_contexts[i] for i in top_indices]
        return top_indices, batch_contexts

    def _featurise_from_contexts(self, contexts: list[MoleculeContext]) -> np.ndarray:
        """Convert a list of MoleculeContexts to a 2-D feature array.

        Feature vector of shape (n, 2053):
        - 2048 bits: ECFP4 (Morgan radius=2) binary fingerprint
        - 5 values: global RDKit 2D descriptors (MW, LogP, TPSA,
          RingCount, NumRotatableBonds)

        Uses pre-computed feature vectors from MoleculeContext to avoid
        redundant SMILES parsing and descriptor computation.

        Args:
            contexts: List of MoleculeContext objects.

        Returns:
            Array of shape (n, 2053) with fingerprints + descriptors.
        """
        X = np.zeros((len(contexts), 2053), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            try:
                X[i] = ctx.get_feature_vector()
            except Exception:
                continue
        return X

    def _screen_molecule(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        """Run a single molecule through the screening pipeline.

        Passes the pre-parsed MoleculeContext to avoid re-parsing.
        Returns the result dict or None on error.
        """
        try:
            result = self.pipeline.screen_molecule(ctx)
        except (ImportError, ValueError, RuntimeError) as e:
            log.warning("Pipeline error for %s: %s", ctx.smiles, e)
            return None
        return result if result is not None else None

    def _count_new_clusters(self, candidates: list[str]) -> int:
        """Legacy: parse SMILES to count clusters.

        Deprecated in favor of ``_count_new_clusters_from_contexts``.
        """
        contexts = [MoleculeContext.from_smiles(s) for s in candidates]
        contexts = [c for c in contexts if c is not None]
        return self._count_new_clusters_from_contexts(contexts) if contexts else 0

    def _count_new_clusters_from_contexts(self, contexts: list[MoleculeContext]) -> int:
        """Count new structural clusters using MiniBatchKMeans on ECFP4 fingerprints.

        Clusters all viable screened molecules, fits KMeans, and counts
        centroids that are not within Tanimoto distance 0.4 of any
        centroid from the previous batch.
        """
        viable_contexts = [
            r for r in self.all_results
            if r.is_viable and r.total_score >= 50.0
        ]
        n_viable = len(viable_contexts)
        if n_viable < 2:
            return 0

        viable_smiles = [r.smiles for r in viable_contexts]
        viable_ctxs = [MoleculeContext.from_smiles(s) for s in viable_smiles]
        viable_ctxs = [c for c in viable_ctxs if c is not None]
        if len(viable_ctxs) < 2:
            return 0

        X = self._featurise_from_contexts(viable_ctxs)
        n_clusters = min(10, len(viable_ctxs))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(X)
        current_centroids = kmeans.cluster_centers_

        if self._prev_centroids is None:
            self._prev_centroids = current_centroids
            return n_clusters

        new_count = 0
        for c in current_centroids:
            binary_c = (c[:2048] > 0.5).astype(np.uint8)
            is_new = True
            for p in self._prev_centroids:
                binary_p = (p[:2048] > 0.5).astype(np.uint8)
                dist = jaccard(binary_c, binary_p)
                if dist < 0.4:
                    is_new = False
                    break
            if is_new:
                new_count += 1

        self._prev_centroids = current_centroids
        return new_count

    @staticmethod
    def _compute_mean_pairwise_tanimoto(fingerprints: list[np.ndarray]) -> float | None:
        """Compute mean pairwise Tanimoto diversity among a list of fingerprint arrays.

        Uses only the first 2048 bits (ECFP4 portion) of each feature vector.
        Uses RDKit BulkTanimotoSimilarity on a random subset of up to 100 fingerprints
        for performance.

        Args:
            fingerprints: List of feature arrays (2053-dim; only first 2048 bits used).

        Returns:
            Mean 1 - Tanimoto similarity, or None if fewer than 2 fingerprints.
        """
        if len(fingerprints) < 2:
            return None

        from rdkit.DataStructs import ExplicitBitVect

        fps = fingerprints
        if len(fps) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(fps), size=100, replace=False)
            fps = [fps[i] for i in idx]

        rdkit_fps: list[ExplicitBitVect] = []
        for arr in fps:
            bv = ExplicitBitVect(2048)
            for i, val in enumerate(arr[:2048]):
                if val > 0.5:
                    bv.SetBit(i)
            rdkit_fps.append(bv)

        similarities: list[float] = []
        for i, fp_i in enumerate(rdkit_fps):
            sims = BulkTanimotoSimilarity(fp_i, rdkit_fps[i + 1:])
            similarities.extend(sims)

        if not similarities:
            return None
        mean_sim = float(np.mean(similarities))
        return 1.0 - mean_sim
