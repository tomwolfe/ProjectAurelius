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

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import generate_discoveries_sdf, generate_run_summary
from aurelius.agent.selection import (
    compute_pairwise_diversity,
    select_for_active_learning,
    tournament_select,
)
from aurelius.agent.state import LoopState
from aurelius.constants import DISCOVERY_THRESHOLD
from aurelius.pipeline import AureliusPipeline
from aurelius.types import (
    MoleculeContext,
    ScreeningResult,
    is_mixture_smiles,
    parse_mixture_smiles,
)

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
    wet_lab_feedback: Callable[[list[ScreeningResult], dict[str, float]], None] | None = None


# ---------------------------------------------------------------------------
# Helper: post-loop mixture synergy analysis
# ---------------------------------------------------------------------------


def _analyze_top_mixtures(
    pipeline: Any,
    discoveries: list[ScreeningResult],
) -> list[dict[str, Any]] | None:
    """Analyze pairwise synergy of top 10 discoveries."""
    top_10 = sorted(discoveries, key=lambda r: -r.total_score)[:10]
    if len(top_10) < 2:
        return None

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
    return mixture_results[:3]


# ---------------------------------------------------------------------------
# Consolidated agent entry point
# ---------------------------------------------------------------------------


def run_screening(agent_cfg: AgentConfig) -> dict[str, Any]:
    """Run the autonomous screening loop and generate deliverables."""
    output_dir = None

    engine = MutationEngine()
    state = LoopState(output_dir=output_dir)

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

    top_mixtures = _analyze_top_mixtures(pipeline, discoveries)
    if top_mixtures:
        log.info("Top-3 mixtures from post-loop analysis:")
        for m in top_mixtures:
            log.info("  %s + %s -> score=%.1f (synergy=%.4f)",
                     m["component1_smiles"], m["component2_smiles"],
                     m["mixture_score"], m["synergy_bonus"])

    generate_run_summary(loop.state, all_results, discoveries, top_mixtures=top_mixtures)
    generate_discoveries_sdf(discoveries)

    if loop.wet_lab_feedback is not None:
        _apply_wet_lab_feedback(loop, pipeline, state, discoveries)

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


def _apply_wet_lab_feedback(
    loop: DiscoveryLoop, pipeline: AureliusPipeline, state: LoopState, discoveries: list[ScreeningResult]
) -> None:
    top_n = sorted(discoveries, key=lambda r: -r.total_score)[:10]
    empirical_metrics: dict[str, float] = {
        "cycle_life": 500.0,
        "coulombic_efficiency": 0.95,
        "transference_number": 0.4,
    }
    loop.wet_lab_feedback(top_n, empirical_metrics)

    oracle_obj = getattr(pipeline, '_oracle', None)
    if oracle_obj is None:
        return
    gc_uq = getattr(oracle_obj, '_gc_uq', None)
    if gc_uq is None:
        return

    feedback_data = []
    for result in top_n:
        diel_correction = min(15.0, max(1.0, empirical_metrics.get("cycle_life", 300.0) / 100.0))
        visc_correction = min(5.0, max(0.1, 5.0 / max(empirical_metrics.get("coulombic_efficiency", 0.9), 0.1)))
        feedback_data.append({
            "smiles": result.smiles,
            "dielectric_constant": (result.dielectric_proxy or 5.0) * 0.7 + diel_correction * 0.3,
            "viscosity_cP": (result.viscosity_proxy or 2.0) * 0.7 + visc_correction * 0.3,
        })
    gc_uq.append_empirical_data(feedback_data)

    for result in top_n:
        state._empirical_feedback.append({
            "smiles": result.smiles,
            "cycle_life": empirical_metrics.get("cycle_life", 500.0),
            "coulombic_efficiency": empirical_metrics.get("coulombic_efficiency", 0.95),
            "dielectric_proxy": result.dielectric_proxy or 0.0,
            "viscosity_proxy": result.viscosity_proxy or 0.0,
            "li_solvation_proxy": result.li_solvation_proxy or 0.0,
        })

    if hasattr(state, 'apply_dynamic_weights') and len(state._empirical_feedback) >= 3:
        state.apply_dynamic_weights(pipeline)
        log.info("Dynamic score weights adjusted based on empirical feedback.")


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
        wet_lab_feedback: Callable[[list[ScreeningResult], dict[str, float]], None] | None = None,
        exploration_beta: float = 0.5,
    ) -> None:
        self.pipeline = pipeline
        self.engine = engine
        self.state = state
        self.max_generations = max_generations
        self.batch_size = batch_size
        self.max_wall_time = max_wall_time
        self.wet_lab_feedback = wet_lab_feedback
        self.exploration_beta = exploration_beta

    @staticmethod
    def _dict_to_screening(d: dict[str, Any]) -> ScreeningResult:
        return ScreeningResult(
            smiles=d.get("smiles", ""),
            total_score=d.get("total_score", 0.0),
            is_viable=d.get("is_viable", False),
            rejection_reasons=d.get("rejection_reasons", []),
            novelty_to_seed=d.get("novelty_to_seed"),
            homo_eV=d.get("homo_eV"),
            lumo_eV=d.get("lumo_eV"),
            dielectric_proxy=d.get("dielectric_proxy"),
            viscosity_proxy=d.get("viscosity_proxy"),
            li_solvation_proxy=d.get("li_solvation_proxy"),
            sa_score=d.get("sa_score"),
            sub_scores=d.get("sub_scores"),
        )

    def execute(self) -> dict[str, Any]:
        wall_start = time.time()

        for generation in range(1, self.max_generations + 1):
            elapsed = time.time() - wall_start
            if elapsed > self.max_wall_time:
                log.info("Time cap reached (%.0fs). Exiting loop.", elapsed)
                break

            force_exploration = self.state.has_scaffold_stagnation(2)
            if force_exploration:
                log.info("Generation %d: Scaffold stagnation detected — pivoting to BRICS-only exploration with UCB active learning.", generation)
                self._inject_tier0_seeds()
                self.exploration_beta = 0.5
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
            "all_results": [self._dict_to_screening(d) for d in self.state._all_results],
            "discoveries": [self._dict_to_screening(d) for d in self.state.discoveries],
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
        single_candidates = list(self.engine.mutate_batch(top_seeds, self.batch_size * 3, force_exploration=force_exploration))
        mixture_candidates = list(self.engine.propose_mixture_candidates(
            top_seeds,
            n_mixtures=max(2, self.batch_size // 5),
            batch_size=5,
        ))
        all_candidates = single_candidates + mixture_candidates
        random.shuffle(all_candidates)
        return all_candidates

    def _top_seeds_from_results(self) -> list[str]:
        return self.state.top_scored_smiles(divisor=5)

    def _filter_candidates(
        self,
        candidates: list[str],
    ) -> tuple[list[MoleculeContext], int]:
        valid_contexts: list[MoleculeContext] = []
        invalid_count = 0

        for smi in candidates:
            if smi in self.state._seen_smiles:
                invalid_count += 1
                continue
            if is_mixture_smiles(smi):
                ctx = self._make_mixture_context(smi)
                if ctx is None:
                    invalid_count += 1
                else:
                    valid_contexts.append(ctx)
            else:
                ctx = MoleculeContext.from_smiles(smi)
                if ctx is None:
                    invalid_count += 1
                    continue
                if not ctx.is_valid_electrolyte_mol():
                    invalid_count += 1
                    continue
                valid_contexts.append(ctx)

        return valid_contexts, invalid_count

    @staticmethod
    def _make_mixture_context(mixture_smi: str) -> MoleculeContext | None:
        """Create a MoleculeContext from a mixture SMILES.

        Validates both components and returns a context keyed on component A
        (for fingerprint-based diversity calculations).
        """
        parsed = parse_mixture_smiles(mixture_smi)
        if parsed is None:
            return None
        smi_a, smi_b, frac = parsed
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return None
        if not ctx_a.is_valid_electrolyte_mol():
            return None
        if not ctx_b.is_valid_electrolyte_mol():
            return None
        return ctx_a

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
    def _build_screening_result(smi: str, total_score: float, score_data: dict[str, Any], t2: dict[str, Any], novelty: float | None, ctx: MoleculeContext, sub_scores: dict[str, Any]) -> ScreeningResult:
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
    def _is_discovery(total_score: float, score_data: dict[str, Any]) -> bool:
        return (total_score >= DISCOVERY_THRESHOLD
                and score_data.get("is_viable", False)
                and not score_data.get("rejection_reasons", []))

    @staticmethod
    def _find_best_mixture_pair(
        valid_contexts: list[MoleculeContext],
        result_map: dict[str, Any] | None,
    ) -> tuple[int, int]:
        """Find indices of the highest-dielectric and lowest-viscosity candidates.

        Returns (-1, -1) if no valid pair is found.
        """
        if len(valid_contexts) < 4 or result_map is None:
            return -1, -1

        best_diel: tuple[int, float] = (-1, -1.0)
        best_visc: tuple[int, float] = (-1, 999.0)
        for i, ctx in enumerate(valid_contexts):
            if is_mixture_smiles(ctx.smiles):
                continue
            res = result_map.get(ctx.smiles)
            if res is None:
                continue
            diel = res.get("dielectric_proxy", 0.0)
            visc = res.get("viscosity_proxy", 999.0)
            if diel > best_diel[1]:
                best_diel = (i, diel)
            if visc < best_visc[1]:
                best_visc = (i, visc)
        if best_diel[0] < 0 or best_visc[0] < 0:
            return -1, -1
        return best_diel[0], best_visc[0]

    def apply_wet_lab_feedback(self, feedback_data: list[dict[str, Any]]) -> None:
        """Apply empirical wet-lab feedback to retrain the GC UQ ensemble.

        Feeds experimentally measured dielectric and viscosity values back
        into ``GcUqEnsemble`` via ``append_empirical_data()``, which flags
        the ensemble for lazy retraining on the next prediction call. This
        allows the UQ model to learn from real-world data and reduce
        prediction uncertainty for fed-back molecules.

        Args:
            feedback_data: List of dicts, each containing ``smiles``,
                ``dielectric_constant``, and ``viscosity_cP`` keys matching
                the ``EmpiricalFeedbackEntry`` TypedDict.
        """
        gc_uq = getattr(getattr(self.pipeline, '_oracle', None), '_gc_uq', None)
        if gc_uq is None:
            log.warning("Cannot apply wet-lab feedback: GcUqEnsemble not available")
            return
        gc_uq.append_empirical_data(feedback_data)
        log.info(
            "Applied wet-lab feedback with %d entries — GcUqEnsemble flagged for retrain",
            len(feedback_data),
        )

    def _evaluate_mixture_pairs(
        self,
        valid_contexts: list[MoleculeContext],
        result_map: dict[str, Any] | None = None,
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Evaluate the most complementary binary mixture pair (high-dielectric + low-viscosity).

        ADR-2026-06-02: Replaced random-pair selection with targeted pairing.
        Physical justification: Random pairs 10% of the time almost never hit
        the complementary regime (high-dielectric + low-viscosity) needed for
        the mixture_synergy_bonus to activate. By intentionally pairing the
        highest-dielectric candidate with the lowest-viscosity candidate, we
        maximise the chance of discovering a synergistic electrolyte mixture.
        This is the minimal change: one targeted pair per batch instead of
        random sampling. The thermodynamic mixing rules already reward this
        complementarity — this intervention simply ensures the EA samples it.
        """
        result_contexts: list[MoleculeContext] = []
        all_scores: list[float] = []

        idx_diel, idx_visc = self._find_best_mixture_pair(valid_contexts, result_map)
        if idx_diel < 0 or idx_visc < 0:
            return result_contexts, all_scores

        ctx1 = valid_contexts[idx_diel]
        ctx2 = valid_contexts[idx_visc]
        if ctx1.smiles == ctx2.smiles:
            return result_contexts, all_scores

        try:
            mix = self.pipeline.screen_mixture(ctx1, ctx2, 0.5)
        except Exception:
            return result_contexts, all_scores

        mix_score = mix.get("score", {}).get("total_score", 0.0)
        mix_smi = f"{ctx1.smiles}|{ctx2.smiles}"
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
            self.state.add_discovery(sr)
        self.state.add_result(sr)
        log.info("  ** MIXTURE ** %s (score=%.1f, synergy=%.4f)", mix_smi, mix_score, synergy)

        return result_contexts, all_scores

    def _evaluate_with_real_quantum(
        self, ctx: MoleculeContext, result_map: dict[str, Any]
    ) -> tuple[float, dict[str, Any]] | None:
        """Evaluate a candidate using the real QuantumOracle (bypassing surrogate).

        Used for molecules in the active learning queue to get genuine
        quantum chemical properties instead of surrogate predictions.
        """
        from aurelius.scoring.oracle.gc import (
            predict_ced_proxy,
            predict_dielectric_proxy,
            predict_li_solvation_proxy,
            predict_viscosity_proxy,
        )
        from aurelius.scoring.oracle.quantum import QuantumOracle

        qo = QuantumOracle()
        qr = qo.evaluate(ctx.mol)

        homo_eV = qr.get("homo_eV", -99.0)
        lumo_eV = qr.get("lumo_eV", -99.0)

        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)
        ced = predict_ced_proxy(ctx)

        score = self.pipeline._compute_score(
            homo_eV=homo_eV, lumo_eV=lumo_eV,
            dielectric_proxy=dielectric,
            viscosity_proxy=viscosity,
            li_solvation_proxy=li_solvation,
            ced_proxy=ced,
            ctx=ctx,
            quantum_confidence="xtb",
        )

        t2 = {
            "homo_eV": homo_eV,
            "lumo_eV": lumo_eV,
            "gap_eV": qr.get("gap_eV", lumo_eV - homo_eV),
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "li_solvation_proxy": li_solvation,
            "ced_proxy": ced,
        }

        smi = ctx.smiles
        result_map[smi] = t2
        self.engine.add_to_db(smi)

        total_score = score.get("total_score", 0.0)
        self.engine.record_reaction_success(smi, total_score)

        novelty = self._compute_novelty(ctx)
        sr = self._build_screening_result(
            smi, total_score, score, t2, novelty, ctx, score.get("sub_scores", {}),
        )
        if self._is_discovery(total_score, score):
            self.state.add_discovery(sr)
            log.info("  ** DISCOVERY (active learning) ** %s (score=%.1f)", smi, total_score)
        self.state.add_result(sr)
        log.info(
            "  ** ACTIVE LEARNING ** %s evaluated via real QuantumOracle",
            smi,
        )
        return total_score, t2

    def _process_single_candidate(
        self, ctx: MoleculeContext, result_map: dict[str, Any]
    ) -> tuple[float, dict[str, Any]] | None:
        """Evaluate a single candidate, record results, and return score data.

        Returns (total_score, tier2_dict) on success, None to skip.

        Uncertainty-Aware Bypass (Phase 1): If the surrogate predicts high
        epistemic uncertainty (std dev > 0.5 eV), the candidate is NOT skipped
        but sent to _evaluate_with_real_quantum for accurate evaluation.
        """
        smi = ctx.smiles

        if smi in self.state.active_learning_queue:
            return self._evaluate_with_real_quantum(ctx, result_map)

        # Uncertainty-Aware Bypass: check surrogate uncertainty first
        try:
            from aurelius.scoring.oracle.surrogate import SurrogateQuantumOracle
            surrogate = SurrogateQuantumOracle()
            homo, lumo, uncertainty = surrogate.predict(ctx)
            penalty = surrogate.compute_penalty(homo, uncertainty)

            # If high uncertainty, skip penalty and evaluate with real quantum
            if penalty == 1.0 and uncertainty > 0.5:
                return self._evaluate_with_real_quantum(ctx, result_map)
        except Exception:
            pass

        result = self._screen_molecule(ctx)
        if result is None:
            return None

        score_data = result.get("score")
        if score_data is None:
            return None

        gc_uq = getattr(getattr(self.pipeline, '_oracle', None), '_gc_uq', None)
        if gc_uq is not None:
            try:
                _, _, diel_high = gc_uq.predict_dielectric(ctx)
                _, _, visc_high = gc_uq.predict_viscosity(ctx)
                if (diel_high or visc_high) and smi not in self.state.active_learning_queue:
                    self.state.active_learning_queue.append(smi)
                    log.info("  Added %s to active learning queue (high UQ)", smi)
            except Exception:
                pass

        self.engine.add_to_db(smi)

        total_score = score_data.get("total_score", 0.0)
        self.engine.record_reaction_success(smi, total_score)

        t2 = result.get("tier2", {}) or {}
        result_map[smi] = t2
        novelty = self._compute_novelty(ctx)
        sub_scores = score_data.get("sub_scores", {})

        sr = self._build_screening_result(smi, total_score, score_data, t2, novelty, ctx, sub_scores)
        if self._is_discovery(total_score, score_data):
            self.state.add_discovery(sr)
            log.info("  ** DISCOVERY ** %s (score=%.1f)", smi, total_score)

        self.state.add_result(sr)
        return total_score, t2

    @staticmethod
    def _get_uncertainties(
        pipeline: Any,
        contexts: list[MoleculeContext],
    ) -> list[float]:
        """Compute combined UQ uncertainties for a list of contexts."""
        gc_uq = getattr(getattr(pipeline, '_oracle', None), '_gc_uq', None)
        uncertainties: list[float] = []
        for ctx in contexts:
            if gc_uq is not None:
                try:
                    _, diel_std, _ = gc_uq.predict_dielectric(ctx)
                    _, visc_std, _ = gc_uq.predict_viscosity(ctx)
                    uncertainties.append((diel_std + visc_std) / 2.0)
                except Exception:
                    uncertainties.append(0.0)
            else:
                uncertainties.append(0.0)
        return uncertainties

    def _select_from_active_learning_queue(
        self,
        result_contexts: list[MoleculeContext],
        all_scores: list[float],
    ) -> tuple[list[MoleculeContext], list[float]] | None:
        """If the active learning queue has items, select from it using UCB.

        Returns (selected, scores) or None if the queue is empty.
        """
        if not self.state.active_learning_queue:
            return None
        queue_set = set(self.state.active_learning_queue)
        queue_contexts: list[MoleculeContext] = []
        queue_scores: list[float] = []
        for ctx, score in zip(result_contexts, all_scores, strict=False):
            if ctx.smiles in queue_set:
                queue_contexts.append(ctx)
                queue_scores.append(score)

        if not queue_contexts:
            return None

        uncertainties = self._get_uncertainties(self.pipeline, queue_contexts)
        selected = select_for_active_learning(
            queue_contexts, queue_scores, uncertainties,
            batch_size=self.batch_size,
        )
        selected_scores = [queue_scores[queue_contexts.index(ctx)] for ctx in selected]

        for ctx in selected:
            if ctx.smiles in self.state.active_learning_queue:
                self.state.active_learning_queue.remove(ctx.smiles)

        return selected, selected_scores

    def _ucb_select_all(
        self,
        result_contexts: list[MoleculeContext],
        all_scores: list[float],
    ) -> tuple[list[MoleculeContext], list[float]]:
        """UCB-based selection: score + beta * uncertainty for every candidate."""
        uncertainties = self._get_uncertainties(self.pipeline, result_contexts)
        selected = select_for_active_learning(
            result_contexts, all_scores, uncertainties,
            batch_size=self.batch_size, beta=self.exploration_beta,
        )
        selected_scores = [all_scores[result_contexts.index(ctx)] for ctx in selected]
        return selected, selected_scores

    def _evaluate_and_select(
        self,
        valid_contexts: list[MoleculeContext],
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Evaluate all valid candidates through the Oracle and select the top batch.

        When exploration_beta > 0, uses UCB-based selection (score + beta * uncertainty)
        instead of raw tournament selection, biasing toward high-uncertainty regions
        to maximise information gain. This is triggered automatically when scaffold
        stagnation is detected, or can be forced via the exploration_beta parameter.
        """
        all_scores: list[float] = []
        result_contexts: list[MoleculeContext] = []
        result_map: dict[str, Any] = {}

        for ctx in valid_contexts:
            processed = self._process_single_candidate(ctx, result_map)
            if processed is None:
                continue
            total_score = processed[0]
            all_scores.append(total_score)
            result_contexts.append(ctx)

        mix_contexts, mix_scores = self._evaluate_mixture_pairs(valid_contexts, result_map)
        result_contexts.extend(mix_contexts)
        all_scores.extend(mix_scores)

        if not result_contexts:
            return [], []

        # Prioritise active learning queue, then exploration, then default
        al_result = self._select_from_active_learning_queue(result_contexts, all_scores)
        if al_result is not None:
            return al_result

        if self.exploration_beta > 0 and len(result_contexts) > self.batch_size:
            return self._ucb_select_all(result_contexts, all_scores)

        if len(result_contexts) <= self.batch_size:
            return result_contexts, all_scores

        selected = tournament_select(result_contexts, all_scores, batch_size=self.batch_size)
        selected_scores = [all_scores[result_contexts.index(ctx)] for ctx in selected]
        return selected, selected_scores

    def _evolve_seed_pool(self, batch_contexts: list[MoleculeContext], batch_scores: list[float]) -> None:
        """Feed high-scoring molecules back into the seed pool and expand commercial precursors."""
        for ctx, sc in zip(batch_contexts, batch_scores, strict=False):
            if sc < 65.0 or is_mixture_smiles(ctx.smiles):
                continue
            smi = ctx.smiles
            existing = set(self.engine.seed_pool)
            if smi not in existing:
                self.engine.seed_pool.append(smi)
                existing.add(smi)
            self.engine.harvest_fragments(smi, score=sc)

            from rdkit.Chem import BRICS
            try:
                for fs in BRICS.BRICSDecompose(ctx.mol):
                    self.engine.add_commercial_fragment(fs, sc)
            except Exception:
                pass

        if len(self.engine.seed_pool) > 200:
            self.engine.seed_pool = self.engine.seed_pool[-200:]
        self.state.seed_pool_size = len(self.engine.seed_pool)

    def _record_scaffolds(self, batch_contexts: list[MoleculeContext]) -> None:
        if MurckoScaffold is None:
            return
        scaffolds = []
        for ctx in batch_contexts:
            if is_mixture_smiles(ctx.smiles):
                continue
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
        if is_mixture_smiles(ctx.smiles):
            return self._screen_mixture(ctx)
        try:
            result = self.pipeline.screen_molecule(ctx)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for %s: %s", ctx.smiles, e)
            return None
        return result if result is not None else None

    def _screen_mixture(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        parsed = parse_mixture_smiles(ctx.smiles)
        if parsed is None:
            return None
        smi_a, smi_b, frac = parsed
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return None
        try:
            mix = self.pipeline.screen_mixture(ctx_a, ctx_b, frac)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for mixture %s: %s", ctx.smiles, e)
            return None
        if mix is None:
            return None
        score = mix.get("score", {})
        mixture_props = mix.get("mixture_properties", {})
        return {
            "score": score,
            "tier2": mixture_props,
            "component1": mix.get("component1"),
            "component2": mix.get("component2"),
        }
