"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), filtering, evaluation (Oracle), tournament selection,
and feedback. SMILES strings are parsed into RDKit Mol objects **exactly once**
per molecule per generation via ``MoleculeContext``.

``AgentConfig`` and ``run_screening`` are the consolidated entry points for
agent execution — ``__main__.py`` imports these rather than duplicate the logic.

ADR-2026-06-01: Reduced scaffold stagnation threshold from 3→2 repeated batches
before force_exploration. Physical justification: in the benchmark, 3 batches of
stagnation means ~15-24 evaluations (3 × batch_size=5-8) before pivoting to BRICS
exploration. Reducing to 2 recovers ~1 generation of wasted exploit-only search.
The benchmark confirms N2 > B2 (+7.6% novel scaffolds). diversity_lambda was kept
at 0.3 because increasing it to 0.4 would deprioritise high-fitness candidates
without proportional novelty gain — tournament_select already applies a diversity
penalty, and the stagnation threshold is the correct tuning knob for triggering
exploration.
"""

from __future__ import annotations

import contextlib
import logging
import random
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

    # Commercial fingerprints are loaded statically from known_electrolytes.json
    # during MutationEngine.__init__; no need to restore from checkpoint.

    resumed = state.total_screened > 0
    if resumed:
        log.info(
            "Resuming from checkpoint: generations=%d, screened=%d, best_score=%.1f",
            state.generations, state.total_screened, state.best_score,
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

    # Post-loop mixture synergy analysis: pairwise score top-10 discoveries
    top_mixtures = None
    top_10 = sorted(discoveries, key=lambda r: -r.total_score)[:10]
    if len(top_10) >= 2:
        mixture_results = []
        for i in range(len(top_10)):
            for j in range(i + 1, len(top_10)):
                ctx_i = MoleculeContext.from_smiles(top_10[i].smiles)
                ctx_j = MoleculeContext.from_smiles(top_10[j].smiles)
                if ctx_i is None or ctx_j is None:
                    continue
                mix = pipeline.screen_mixture(ctx_i, ctx_j, 0.5)
                mix_score = mix.get("score", {}).get("total_score", 0.0)
                synergy = mix.get("mixture_properties", {}).get("synergy_bonus", 0.0)
                mixture_results.append({
                    "component1_smiles": top_10[i].smiles,
                    "component2_smiles": top_10[j].smiles,
                    "component1_score": top_10[i].total_score,
                    "component2_score": top_10[j].total_score,
                    "mixture_score": round(mix_score, 4),
                    "synergy_bonus": round(synergy, 4),
                })
        mixture_results.sort(key=lambda m: -m["mixture_score"])
        top_mixtures = mixture_results[:3]
        log.info("Top-3 mixtures from post-loop analysis:")
        for m in top_mixtures:
            log.info("  %s + %s -> score=%.1f (synergy=%.4f)",
                     m["component1_smiles"], m["component2_smiles"],
                     m["mixture_score"], m["synergy_bonus"])

    generate_run_summary(loop.state, all_results, discoveries, top_mixtures=top_mixtures)
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

            force_exploration = self.state.has_scaffold_stagnation(2)
            if force_exploration:
                log.info("Generation %d: Scaffold stagnation detected — pivoting to BRICS-only exploration.", generation)
                self._inject_tier0_seeds()
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
            "total_screened": self.state.total_screened,
            "total_viable": self.state.viable_count,
            "total_invalid": self.state.invalid_discarded,
        }

    def _inject_tier0_seeds(self) -> None:
        import json
        from pathlib import Path
        tier0_path = Path(__file__).resolve().parent.parent.parent / "data" / "tier0_seed_smiles.json"
        if not tier0_path.exists():
            return
        with open(tier0_path) as f:
            all_seeds = json.load(f)
        existing = set(self.engine.seed_pool)
        unused = [s for s in all_seeds if s not in existing]
        if unused:
            selected = random.sample(unused, min(5, len(unused)))
            self.engine.seed_pool.extend(selected)
            log.info("  Injected %d fresh tier0 seed SMILES into seed pool.", len(selected))

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

    def _compute_novelty(self, ctx: MoleculeContext) -> float | None:
        """Compute novelty score (1 - max Tanimoto to seed pool)."""
        fp = ctx.get_ecfp4()
        seed_fps = getattr(self.engine, "seed_fingerprints", None)
        if not isinstance(seed_fps, list) or not seed_fps:
            return None
        from rdkit.DataStructs import BulkTanimotoSimilarity
        sims = BulkTanimotoSimilarity(fp, seed_fps)
        return 1.0 - max(sims) if sims else None

    @staticmethod
    def _build_screening_result(smi: str, total_score: float, score_data: dict, t2: dict, novelty: float | None, ctx: MoleculeContext, sub_scores: dict) -> ScreeningResult:
        return ScreeningResult(
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

    @staticmethod
    def _is_discovery(total_score: float, score_data: dict) -> bool:
        return (total_score >= DISCOVERY_THRESHOLD
                and score_data.get("is_viable", False)
                and not score_data.get("rejection_reasons", []))

    def _evaluate_mixture_pairs(
        self,
        valid_contexts: list[MoleculeContext],
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Occasionally (~10%) evaluate binary pairs as mixtures during the loop."""
        result_contexts: list[MoleculeContext] = []
        all_scores: list[float] = []

        if len(valid_contexts) < 4:
            return result_contexts, all_scores

        n_pairs = max(1, int(self.batch_size * 0.1))
        indices = list(range(len(valid_contexts)))
        random.shuffle(indices)
        for k in range(0, min(n_pairs * 2, len(indices) - 1), 2):
            ctx1 = valid_contexts[indices[k]]
            ctx2 = valid_contexts[indices[k + 1]]
            try:
                mix = self.pipeline.screen_mixture(ctx1, ctx2, 0.5)
            except Exception:
                continue
            mix_score = mix.get("score", {}).get("total_score", 0.0)
            mix_smi = f"{ctx1.smiles}|{ctx2.smiles}"
            self.screened_smiles.add(mix_smi)
            all_scores.append(mix_score)
            result_contexts.append(ctx1)
            synergy = mix.get("mixture_properties", {}).get("synergy_bonus", 0.0)
            mix_score_data = mix.get("score", {})
            mix_t2 = mix.get("mixture_properties", {})
            novelty = self._compute_novelty(ctx1)
            sr = self._build_screening_result(
                mix_smi, mix_score, mix_score_data, mix_t2, novelty, ctx1,
                mix_score_data.get("sub_scores", {}),
            )
            if sr.total_score >= DISCOVERY_THRESHOLD:
                self.discoveries.append(sr)
                self.state.add_discovery(sr)
            self.all_results.append(sr)
            log.info("  ** MIXTURE ** %s (score=%.1f, synergy=%.4f)", mix_smi, mix_score, synergy)

        return result_contexts, all_scores

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
            self.engine.record_reaction_success(smi, total_score)
            all_scores.append(total_score)
            result_contexts.append(ctx)

            t2 = result.get("tier2", {}) or {}
            novelty = self._compute_novelty(ctx)
            sub_scores = score_data.get("sub_scores", {})

            sr = self._build_screening_result(smi, total_score, score_data, t2, novelty, ctx, sub_scores)
            if self._is_discovery(total_score, score_data):
                self.discoveries.append(sr)
                self.state.add_discovery(sr)
                log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

            self.all_results.append(sr)

        mix_contexts, mix_scores = self._evaluate_mixture_pairs(valid_contexts)
        result_contexts.extend(mix_contexts)
        all_scores.extend(mix_scores)

        if not result_contexts:
            return [], []

        if len(result_contexts) <= self.batch_size:
            return result_contexts, all_scores

        selected = tournament_select(result_contexts, all_scores, batch_size=self.batch_size)
        selected_scores = [all_scores[result_contexts.index(ctx)] for ctx in selected]
        return selected, selected_scores

    def _evolve_seed_pool(self, batch_contexts: list[MoleculeContext], batch_scores: list[float]) -> None:
        """Feed high-scoring molecules back into the seed pool."""
        for ctx, sc in zip(batch_contexts, batch_scores):
            if sc >= 65.0:
                smi = ctx.smiles
                existing = set(self.engine.seed_pool)
                if smi not in existing:
                    self.engine.seed_pool.append(smi)
                    existing.add(smi)
                self.engine.harvest_fragments(smi, score=sc)
        if len(self.engine.seed_pool) > 200:
            self.engine.seed_pool = self.engine.seed_pool[-200:]
        self.state.seed_pool_size = len(self.engine.seed_pool)

    def _record_scaffolds(self, batch_contexts: list[MoleculeContext]) -> None:
        if MurckoScaffold is None:
            return
        scaffolds = []
        for ctx in batch_contexts:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                if scaffold:
                    scaffolds.append(scaffold)
            except Exception:
                continue
        if scaffolds:
            self.state.record_scaffolds(scaffolds)

    def _record_results(
        self,
        batch_contexts: list[MoleculeContext],
        batch_scores: list[float],
        generation: int,
    ) -> None:
        """Record batch results, update state, and evolve seeds/fragments."""
        batch_viable = sum(1 for s in batch_scores if s >= DISCOVERY_THRESHOLD)
        self._evolve_seed_pool(batch_contexts, batch_scores)
        self._record_scaffolds(batch_contexts)
        self.state.record_batch(batch_scores, batch_viable)

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
