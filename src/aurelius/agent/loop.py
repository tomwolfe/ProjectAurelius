"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), screening, feedback, convergence checking, and
checkpointing.  SMILES strings are parsed into RDKit Mol objects **exactly once**
per molecule per generation via ``MoleculeContext``.

``AgentConfig`` and ``run_screening`` are the consolidated entry points for
agent execution — ``__main__.py`` imports these rather than duplicate the logic.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.spatial.distance import jaccard
from sklearn.cluster import MiniBatchKMeans

from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import generate_discoveries_sdf, generate_run_summary
from aurelius.agent.state import LoopState
from aurelius.agent.surrogate import RandomForestSurrogate
from aurelius.constants import DISCOVERY_THRESHOLD
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext, ScreeningResult

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    """Parameters for the autonomous screening agent."""

    max_generations: int = 50
    batch_size: int = 50


# ---------------------------------------------------------------------------
# Consolidated agent entry point
# ---------------------------------------------------------------------------


def run_screening(agent_cfg: AgentConfig) -> dict[str, Any]:
    """Run the autonomous screening loop and generate deliverables.

    This is the single entry point for agent execution, called both from
    the CLI (``aurelius agent``) and programmatic use.  All heavy lifting
    (pipeline construction, checkpointing, loop orchestration) lives here.
    """
    output_dir = None  # could be lifted to AgentConfig in future

    engine = MutationEngine()
    state = LoopState(output_dir=output_dir)

    for h in getattr(state, "known_fps_hex", []):
        with contextlib.suppress(Exception):
            from aurelius.utils.chem_utils import _deserialize_fp
            engine.known_fps.append(_deserialize_fp(h))

    resumed = state.total_screened > 0
    if resumed:
        log.info(
            "Resuming from checkpoint: batch=%d, screened=%d, best_score=%.1f",
            state.batch, state.total_screened, state.best_score,
        )
    else:
        log.info("Fresh start. No checkpoint found.")

    wall_start = time.time()

    pipeline = AureliusPipeline()
    pipeline.initialize()

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        state=state,
        max_generations=agent_cfg.max_generations,
        batch_size=agent_cfg.batch_size,
    )
    results = loop.execute()

    all_results = results["all_results"]
    discoveries = results["discoveries"]

    generate_run_summary(loop.state, all_results, discoveries)
    generate_discoveries_sdf(discoveries)
    state.save()

    log.info("=" * 60)
    log.info("  SCREENING COMPLETE")
    log.info("=" * 60)
    log.info("  Total screened:     %d", results["total_screened"])
    log.info("  Generations run:    %d", state.generations)
    log.info("  Viable discoveries: %d", results["total_viable"])
    log.info("  Best score:         %.1f", state.best_score)
    log.info("  Invalid discarded:  %d", results["total_invalid"])
    log.info("  Wall time:          %.0fs", time.time() - wall_start)

    return results


# ---------------------------------------------------------------------------
# Discovery loop
# ---------------------------------------------------------------------------


class DiscoveryLoop:
    """Main autonomous screening loop.

    The loop runs for max_generations iterations.  Each generation:
    1. Mutates seed molecules via the mutation engine
    2. Filters invalid / duplicate candidates (parses to MoleculeContext)
    3. Featurises candidates from MoleculeContext pre-computed vectors
    4. Selects top candidates via Bayesian expected improvement
    5. Screens selected candidates through the pipeline (reuses Mol)
    6. Records results / feedback / convergence state
    7. Refits the RF surrogate from accumulated (X, y) history
    8. Saves checkpoint at the end of each generation
    """

    def __init__(
        self,
        pipeline: Any,
        engine: Any,
        state: Any,
        max_generations: int = 50,
        batch_size: int = 50,
        max_wall_time: float = 43200.0,
    ) -> None:
        self.pipeline = pipeline
        self.engine = engine
        self.state = state
        self.max_generations = max_generations
        self.batch_size = batch_size
        self.max_wall_time = max_wall_time

        self.total_screened = 0
        self.total_viable = 0
        self.total_invalid = 0
        self.all_results: list[ScreeningResult] = []
        self.discoveries: list[ScreeningResult] = []
        self._surrogate: RandomForestSurrogate | None = None
        self.screened_smiles: set[str] = set()
        self._prev_centroids: np.ndarray[Any, Any] | None = None
        self._exploration_boost_active: bool = False
        self._exploration_boost_generations: int = 0

    def execute(self) -> dict[str, Any]:
        wall_start = time.time()
        generation = 0

        while generation < self.max_generations:
            elapsed = time.time() - wall_start
            if elapsed > self.max_wall_time:
                log.info("Time cap reached (%.0fs). Exiting loop.", elapsed)
                break

            generation += 1

            candidates = self._generate_candidates(generation)
            valid_contexts, invalid_count = self._filter_candidates(candidates)

            if not valid_contexts:
                log.info("Generation %d: No valid candidates. Skipping.", generation)
                continue

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

                fv = ctx.get_feature_vector()
                batch_feature_vectors.append(fv)

                novelty: float | None = None
                rdkit_fp = ctx.get_ecfp4()
                seed_fps = getattr(self.engine, "seed_fingerprints", None)
                if isinstance(seed_fps, list) and seed_fps:
                    sims = BulkTanimotoSimilarity(rdkit_fp, seed_fps)
                    novelty = 1.0 - max(sims) if sims else None

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
                    li_solvation_proxy=t2.get("li_solvation_proxy"),
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
                        li_solvation_proxy=t2.get("li_solvation_proxy"),
                        sa_score=score_data.get("sa_score"),
                        sub_scores=sub_scores,
                    )
                    batch_discoveries.append(discovery_entry)
                    self.discoveries.append(discovery_entry)
                    self.state.add_discovery(discovery_entry)
                    log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

                self.all_results.append(screening_result)
                self.state.record(screening_result)

            # Fit the surrogate from accumulated (X, y) history
            X, y = self.state.get_history()
            if len(y) >= 2:
                if self._surrogate is None:
                    self._surrogate = RandomForestSurrogate()
                self._surrogate.fit(X, y)

            # Adaptive diversity: report batch variance to surrogate
            if self._surrogate is not None and batch_scores:
                self._surrogate.update_variance(batch_scores)

            # Scaffold tracking: extract Murcko scaffolds and check for stagnation
            batch_scaffolds: list[str] = []
            if MurckoScaffold is not None:
                for ctx in batch_contexts:
                    try:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                        if scaffold:
                            batch_scaffolds.append(scaffold)
                    except Exception:
                        continue
            if batch_scaffolds:
                self.state.record_scaffolds(batch_scaffolds)

            # Exploration boost: if scaffold stagnation detected, increase xi
            if self.state.has_scaffold_stagnation(n_batches=3):
                if not self._exploration_boost_active:
                    log.info("  ** Exploration boost triggered: scaffold stagnation detected **")
                    self._exploration_boost_active = True
                    self._exploration_boost_generations = 0
                    if self._surrogate is not None:
                        self._surrogate._xi = 0.10  # Increase xi from 0.01 to 0.10
            else:
                if self._exploration_boost_active:
                    self._exploration_boost_generations += 1
                    if self._exploration_boost_generations >= 3:
                        self._exploration_boost_active = False
                        if self._surrogate is not None:
                            self._surrogate._xi = 0.01  # Reset xi
                        log.info("  Exploration boost deactivated after 3 generations")

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
                if len(self.engine.seed_pool) > 200:
                    self.engine.seed_pool = self.engine.seed_pool[-200:]
            self.state.seed_pool_size = len(self.engine.seed_pool)

            new_clusters = self._count_new_clusters_from_contexts(valid_contexts)
            self.state.record_batch(batch_scores, batch_viable, new_clusters)
            self.state.update_stats(
                [c.smiles for c in batch_contexts],
                batch_scores, batch_viable, invalid_count,
            )
            self.total_screened += len(valid_contexts)
            self.total_viable += batch_viable
            self.total_invalid += invalid_count

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

            strategy = self.state.get_adaptation_strategy()
            if generation % 5 == 0:
                log.info("  [Feedback] Strategy: %s", strategy.get("recommendation", ""))

            should_stop, reason = self.state.should_terminate()
            if should_stop:
                log.info("Convergence reached: %s", reason)
                break

            self.state.save()

        return {
            "all_results": self.all_results,
            "discoveries": self.discoveries,
            "total_screened": self.total_screened,
            "total_viable": self.total_viable,
            "total_invalid": self.total_invalid,
        }

    def _generate_candidates(self, generation: int) -> list[str]:
        top_seeds = self.engine.seed_pool if generation == 1 else self._top_seeds_from_results()
        candidates = self.engine.mutate_batch(top_seeds, self.batch_size * 3)
        return list(candidates)

    def _top_seeds_from_results(self) -> list[str]:
        scored = [(r.total_score, r.smiles) for r in self.all_results if r.total_score > 0]
        scored.sort(key=lambda x: -x[0])
        n = max(5, len(scored) // 5)
        return [s for _, s in scored[:n]]

    def _filter_candidates(
        self,
        candidates: list[str],
    ) -> tuple[list[MoleculeContext], int]:
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
            if not ctx.is_valid_electrolyte_mol():
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
        if len(valid_contexts) == 0:
            return [], []

        X_pool = self._featurise_from_contexts(valid_contexts)

        if self._surrogate is not None:
            fps = [ctx.get_ecfp4() for ctx in valid_contexts]

            diversity_lambda = self._surrogate.diversity_lambda
            if self._exploration_boost_active:
                diversity_lambda = min(diversity_lambda + 0.2, 1.0)
                log.info("  Exploration boost active: diversity_lambda=%.2f", diversity_lambda)

            top_indices: list[int] = self._surrogate.score_candidates(
                X_pool, fingerprints=fps, top_n=self.batch_size,
                diversity_lambda=diversity_lambda,
            )
        else:
            n = min(self.batch_size, len(valid_contexts))
            indices = np.random.default_rng(42).choice(len(valid_contexts), size=n, replace=False)
            top_indices = sorted(indices.tolist())

        batch_contexts = [valid_contexts[i] for i in top_indices]
        return top_indices, batch_contexts

    def _featurise_from_contexts(self, contexts: list[MoleculeContext]) -> np.ndarray:
        X = np.zeros((len(contexts), 2053), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            try:
                X[i] = ctx.get_feature_vector()
            except Exception:
                continue
        return X

    def _screen_molecule(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        try:
            result = self.pipeline.screen_molecule(ctx)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for %s: %s", ctx.smiles, e)
            return None
        return result if result is not None else None

    def _count_new_clusters_from_contexts(self, contexts: list[MoleculeContext]) -> int:
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
