"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), filtering, evaluation (Oracle), tournament selection,
and feedback. SMILES strings are parsed into RDKit Mol objects **exactly once**
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

from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import generate_discoveries_sdf, generate_run_summary
from aurelius.agent.selection import compute_pairwise_diversity, tournament_select
from aurelius.agent.state import LoopState
from aurelius.constants import DISCOVERY_THRESHOLD
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext, ScreeningResult

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None

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
    the CLI (``aurelius agent``) and programmatic use.
    """
    output_dir = None

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
    1. Mutate seed molecules via the mutation engine
    2. Filter invalid / duplicate candidates (parse to MoleculeContext)
    3. Evaluate all valid candidates through the Oracle pipeline
    4. Select top candidates via tournament selection + diversity penalty
    5. Record results, evolve seed pool, harvest fragments
    6. Check convergence, save checkpoint
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
        self.screened_smiles: set[str] = set()

    def execute(self) -> dict[str, Any]:
        wall_start = time.time()

        for generation in range(1, self.max_generations + 1):
            elapsed = time.time() - wall_start
            if elapsed > self.max_wall_time:
                log.info("Time cap reached (%.0fs). Exiting loop.", elapsed)
                break

            force_exploration = self.state.has_scaffold_stagnation(3)
            if force_exploration:
                log.info("Generation %d: Scaffold stagnation detected — pivoting to BRICS-only exploration.", generation)
            candidates = self._generate_candidates(generation, force_exploration=force_exploration)
            valid_contexts, invalid_count = self._filter_candidates(candidates)

            if not valid_contexts:
                log.info("Generation %d: No valid candidates. Skipping.", generation)
                continue

            log.info(
                "Generation %d: %d candidates (%d invalid, %d selected for eval)",
                generation,
                len(valid_contexts),
                invalid_count,
                min(self.batch_size, len(valid_contexts)),
            )

            batch_contexts, batch_scores = self._evaluate_and_select(valid_contexts)
            if not batch_contexts:
                continue

            self._record_results(batch_contexts, batch_scores, generation)

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

    def _generate_candidates(self, generation: int, force_exploration: bool = False) -> list[str]:
        top_seeds = self.engine.seed_pool if generation == 1 else self._top_seeds_from_results()
        return list(self.engine.mutate_batch(top_seeds, self.batch_size * 3, force_exploration=force_exploration))

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

        return valid_contexts, invalid_count

    def _evaluate_and_select(
        self,
        valid_contexts: list[MoleculeContext],
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Evaluate all valid candidates through the Oracle and select the top batch."""
        all_scores: list[float] = []
        result_contexts: list[MoleculeContext] = []

        for ctx in valid_contexts:
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
            all_scores.append(total_score)
            result_contexts.append(ctx)

            novelty: float | None = None
            rdkit_fp = ctx.get_ecfp4()
            seed_fps = getattr(self.engine, "seed_fingerprints", None)
            if isinstance(seed_fps, list) and seed_fps:
                from rdkit.DataStructs import BulkTanimotoSimilarity
                sims = BulkTanimotoSimilarity(rdkit_fp, seed_fps)
                novelty = 1.0 - max(sims) if sims else None

            t2 = result.get("tier2", {}) or {}
            sub_scores = score_data.get("sub_scores", {})
            screening_result = ScreeningResult(
                smiles=smi,
                total_score=total_score,
                is_viable=score_data.get("is_viable", False),
                rejection_reasons=score_data.get("rejection_reasons", []),
                fingerprint=ctx.get_feature_vector(),
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
                discovery_entry = ScreeningResult(
                    smiles=smi,
                    total_score=total_score,
                    is_viable=True,
                    rejection_reasons=score_data.get("rejection_reasons", []),
                    fingerprint=ctx.get_feature_vector(),
                    novelty_to_seed=novelty,
                    homo_eV=t2.get("homo_eV"),
                    lumo_eV=t2.get("lumo_eV"),
                    dielectric_proxy=t2.get("dielectric_proxy"),
                    viscosity_proxy=t2.get("viscosity_proxy"),
                    li_solvation_proxy=t2.get("li_solvation_proxy"),
                    sa_score=score_data.get("sa_score"),
                    sub_scores=sub_scores,
                )
                self.discoveries.append(discovery_entry)
                self.state.add_discovery(discovery_entry)
                log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

            self.all_results.append(screening_result)
            self.state.record(screening_result)

        if not result_contexts:
            return [], []

        if len(result_contexts) <= self.batch_size:
            return result_contexts, all_scores

        selected = tournament_select(
            result_contexts,
            all_scores,
            batch_size=self.batch_size,
        )

        selected_scores = [
            all_scores[result_contexts.index(ctx)] for ctx in selected
        ]
        return selected, selected_scores

    def _record_results(
        self,
        batch_contexts: list[MoleculeContext],
        batch_scores: list[float],
        generation: int,
    ) -> None:
        """Record batch results, update state, and evolve seeds/fragments."""
        batch_viable = sum(1 for s in batch_scores if s >= DISCOVERY_THRESHOLD)

        seed_feed = [
            ctx.smiles for ctx, sc in zip(batch_contexts, batch_scores, strict=False)
            if sc >= 65.0
        ]
        if seed_feed:
            existing = set(self.engine.seed_pool)
            for smi in seed_feed:
                if smi not in existing:
                    self.engine.seed_pool.append(smi)
                    existing.add(smi)
            if len(self.engine.seed_pool) > 200:
                self.engine.seed_pool = self.engine.seed_pool[-200:]

        for ctx, sc in zip(batch_contexts, batch_scores, strict=False):
            if sc >= 65.0:
                self.engine.harvest_fragments(ctx.smiles)

        self.state.seed_pool_size = len(self.engine.seed_pool)

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

        self.state.record_batch(batch_scores, batch_viable)

        self.total_screened += len(batch_contexts)
        self.total_viable += batch_viable

        mean_div = compute_pairwise_diversity(batch_contexts)
        log.info(
            "  Generation %d: %d screened, %d viable, best=%.1f, diversity=%.4f",
            generation,
            len(batch_contexts),
            batch_viable,
            max(batch_scores) if batch_scores else 0,
            mean_div,
        )

    def _screen_molecule(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        try:
            result = self.pipeline.screen_molecule(ctx)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for %s: %s", ctx.smiles, e)
            return None
        return result if result is not None else None
