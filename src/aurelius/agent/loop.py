"""Discovery loop for the autonomous screening agent.

The ``DiscoveryLoop`` encapsulates the main autonomous screening loop:
generation (mutation), filtering, evaluation (Oracle), tournament selection,
and feedback. SMILES strings are parsed into RDKit Mol objects **exactly once**
per molecule per generation via ``MoleculeContext``.

``AgentConfig`` and ``run_screening`` are the consolidated entry points for
agent execution - ``__main__.py`` imports these rather than duplicate the logic.

ADR-2026-06-01: Reduced scaffold stagnation threshold from 3→2 repeated batches
before force_exploration. Physical justification: in the benchmark, 3 batches of
stagnation means ~15-24 evaluations (3 × batch_size=5-8) before pivoting to BRICS
exploration. Reducing to 2 recovers ~1 generation of wasted exploit-only search.
The benchmark confirms N2 > B2 (+7.6% novel scaffolds). diversity_lambda was kept
at 0.3 because increasing it to 0.4 would deprioritise high-fitness candidates
without proportional novelty gain - tournament_select already applies a diversity
penalty, and the stagnation threshold is the correct tuning knob for triggering
exploration.
"""

from __future__ import annotations

import logging
import multiprocessing
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from aurelius.agent.feedback import FeedbackController
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import generate_discoveries_sdf, generate_run_summary
from aurelius.agent.selection import (
    build_npga2_composite_objectives,
    compute_pairwise_diversity,
    nsga2_select,
    tournament_select,
)
from aurelius.agent.state import LoopState
from aurelius.constants import DISCOVERY_THRESHOLD, EVOLUTION_RELATIVE_FRACTION, SEED_POOL_MAX_SIZE
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle.delta_correction import get_delta_correction
from aurelius.scoring.oracle.quantum import has_xtb
from aurelius.screening.tier0.prefilter import Tier0Prefilter
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

# Use all available cores for parallel molecule evaluation, but cap at 8 to
# avoid oversubscription on a machine that may also be running the host UI
# and xTB subprocesses.
_EVAL_WORKERS = min(multiprocessing.cpu_count(), 8)

if TYPE_CHECKING:
    from aurelius.scoring.oracle.xtb_single_point import XTBSinglePointOracle

log = logging.getLogger(__name__)


def _loop_sha_bucket(seed: int, tag: str, key: str, mod: int = 100) -> int:
    """Stateless hash bucket in [0, mod) keyed by (seed, tag, key)."""
    import hashlib

    digest = hashlib.sha256(f"{seed}|{tag}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % mod


def _loop_stateless_shuffle(seed: int, tag: str, key: str, items: list) -> None:
    """In-place deterministic shuffle keyed by (seed, tag, key)."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256(f"{seed}|{tag}|{key}".encode("utf-8")).digest()
    seq = np.random.SeedSequence(int.from_bytes(digest[:8], "little"))
    np.random.default_rng(seq).shuffle(items)


def _loop_stateless_sample(seed: int, tag: str, key: str, population: list, k: int) -> list:
    """Stateless without-replacement sample keyed by (seed, tag, key)."""
    if k >= len(population) or not population:
        return list(population)
    order = sorted(population, key=lambda x: _loop_sha_bucket(seed, tag, f"{key}|{x}", 1000003))
    return order[:k]


class _LockedXTBSinglePointOracle:
    """Thread-safe wrapper around ``XTBSinglePointOracle`` for parallel ranking.

    ``XTBSinglePointOracle.evaluate`` mutates a shared in-memory cache and
    calls ``_persist()`` - a disk write of ``xtb_cache.json`` - on every cold
    call, so concurrent ranking threads would race on the cache file. The
    wrapper serialises ``_persist`` with a class-level lock and delegates
    everything else to the wrapped instance.
    """

    _persist_lock = threading.Lock()

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        return self._wrapped.evaluate(ctx)

    def _persist(self) -> None:
        with type(self)._persist_lock:
            self._wrapped._persist()


def _collect_obj_scores(
    obj_scores: dict[str, list[float]],
    total_score: float,
    t2: dict[str, Any],
    score_data: dict[str, Any],
    confidence: float,
    novelty: float = 0.0,
) -> None:
    """Collect per-candidate objective scores into ``obj_scores`` dict.

    Called for each evaluated candidate to populate the multi-objective
    matrix used by NSGA-II selection.
    """
    obj_scores["total_score"].append(total_score)
    obj_scores["dielectric_proxy"].append(t2.get("dielectric_proxy", 0.0))
    obj_scores["viscosity_proxy"].append(t2.get("viscosity_proxy", 99.0))
    obj_scores["li_solvation_proxy"].append(t2.get("li_solvation_proxy", 0.0))
    obj_scores["homo_eV"].append(t2.get("homo_eV", -99.0))
    obj_scores["lumo_eV"].append(t2.get("lumo_eV", -99.0))
    obj_scores["sa_score"].append(score_data.get("sa_score", 5.0))
    obj_scores["synthesis_depth"].append(
        float(score_data.get("synthesis_depth", 3))
    )
    obj_scores["synthesizability_complexity"].append(
        score_data.get("synthesizability_complexity", 0.5)
    )
    obj_scores["confidence"].append(confidence)
    obj_scores["combined_grounding_score"].append(
        score_data.get("grounding", 0.0)
    )
    obj_scores["novelty_to_seed"].append(novelty)


def _mixture_pad_value(key: str, mix_score: float) -> float:
    """Placeholder objective value for a mixture candidate.

    Mixtures inherit makeability from their already-screened components, so
    ``combined_grounding_score`` pads to a neutral 1.0 rather than 0.0, which
    would otherwise impose the maximum synthesizability penalty on every
    mixture and eliminate them from selection (ADR-2026-08-08-04).
    """
    if key == "total_score":
        return mix_score
    if key == "synthesis_depth":
        return 3.0
    if key == "combined_grounding_score":
        return 1.0
    return 0.0


@dataclass(frozen=True)
class AgentConfig:
    """Parameters for the autonomous screening agent."""

    max_generations: int = 50
    batch_size: int = 50
    use_nsga2: bool = True
    active_learning_threshold: float = 0.7
    xtb_budget_per_generation: int = 10
    seed: int = 42
    mixture_mutation_rate: float = 0.35
    mixture_seed_from_known: bool = True
    # ADR-2026-08-10: this field was referenced by both ``run_screening`` and
    # the ``aurelius agent`` CLI but never declared, so every CLI invocation
    # raised TypeError before a single generation ran. The Tier-2.5 xTB gate
    # was therefore unreachable from the command line since it was added.
    xtb_single_point: bool = True


# ---------------------------------------------------------------------------
# Consolidated agent entry point
# ---------------------------------------------------------------------------


def _save_run_config(agent_cfg: AgentConfig) -> None:
    """Serialize AgentConfig and seed to run_config.json for reproducibility.

    The config file is hashed and the hash is included in run_summary.json
    so that any run can be fully reproduced from its summary file.
    """
    import hashlib
    import json

    config = {
        "max_generations": agent_cfg.max_generations,
        "batch_size": agent_cfg.batch_size,
        "use_nsga2": agent_cfg.use_nsga2,
        "active_learning_threshold": agent_cfg.active_learning_threshold,
        "seed": agent_cfg.seed,
        "mixture_mutation_rate": agent_cfg.mixture_mutation_rate,
        "mixture_seed_from_known": agent_cfg.mixture_seed_from_known,
        "xtb_single_point": agent_cfg.xtb_single_point,
    }
    config_path = "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    log.info("Run config saved to %s (hash: %s)", config_path, config_hash[:16])


def run_screening(agent_cfg: AgentConfig) -> dict[str, Any]:
    """Run the autonomous screening loop and generate deliverables.

    This is the single entry point for agent execution, called both from
    the CLI (``aurelius agent``) and programmatic use.
    """
    output_dir = None

    engine = MutationEngine(seed=agent_cfg.seed)
    random.seed(agent_cfg.seed)
    state = LoopState(output_dir=output_dir)

    # Serialize run configuration for reproducibility
    _save_run_config(agent_cfg)

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

    pipeline = AureliusPipeline(use_xtb=False)
    pipeline.initialize()

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        state=state,
        max_generations=agent_cfg.max_generations,
        batch_size=agent_cfg.batch_size,
        use_nsga2=agent_cfg.use_nsga2,
        active_learning_threshold=agent_cfg.active_learning_threshold,
        xtb_budget_per_generation=agent_cfg.xtb_budget_per_generation,
        xtb_single_point=agent_cfg.xtb_single_point,
        mixture_mutation_rate=agent_cfg.mixture_mutation_rate,
        mixture_seed_from_known=agent_cfg.mixture_seed_from_known,
        seed=agent_cfg.seed,
    )

    # W6: Initialise experimental feedback controller for on-line oracle refinement
    feedback_controller = FeedbackController(
        delta_correction=get_delta_correction(),
    )
    loop.feedback_controller = feedback_controller
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
        use_nsga2: bool = True,
        active_learning_threshold: float = 0.7,
        xtb_budget_per_generation: int = 10,
        xtb_single_point: bool = True,
        mixture_mutation_rate: float = 0.35,
        mixture_seed_from_known: bool = True,
        seed_knowns: bool = False,
        seed: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.engine = engine
        self.state = state
        self.max_generations = max_generations
        self.batch_size = batch_size
        self.max_wall_time = max_wall_time
        self.use_nsga2 = use_nsga2
        self.active_learning_threshold = active_learning_threshold
        self.xtb_budget_per_generation = xtb_budget_per_generation
        self.xtb_single_point = xtb_single_point and has_xtb()
        self.mixture_mutation_rate = max(0.0, min(1.0, mixture_mutation_rate))
        self.mixture_seed_from_known = mixture_seed_from_known
        self.seed_knowns = seed_knowns
        self.seed = seed
        self._rng = random.Random(seed)
        # Iterated-local-search basin state (Phase 3: aurelius-global-closure).
        # _basin_best tracks the best score seen; consecutive generations
        # without improvement increment _basin_stall, which triggers a
        # basin-hop perturb-and-accept step. Plain attributes (not LoopState
        # fields) so checkpoint schema is unchanged.
        self._basin_best: float = 0.0
        self._basin_stall: int = 0
        # Incumbent SMILES for iterated local search (best seen so far).
        self._basin_center: str | None = None
        self.feedback_controller: FeedbackController | None = None

        self.all_results: list[ScreeningResult] = []
        self.discoveries: list[ScreeningResult] = []
        self.screened_smiles: set[str] = set()

        # Active learning budget tracking
        self._xtb_escalation_count: int = 0
        self._xtb_escalation_success: int = 0
        self._escalation_history: list[dict[str, Any]] = []
        self._xtb_sp_calls = 0
        self._xtb_sp_hits = 0
        self._xtb_sp_orca_eligible = 0
        # TASK 5 SPEED: cache Tier-0 prefilter and xTB pipeline - init once,
        # reuse across generations instead of per-molecule/per-generation init.
        self.tier0_prefilter = Tier0Prefilter()
        self._xtb_pipeline: Any = None

    @property
    def xtb_sp_report(self) -> dict[str, Any]:
        """Cache utilisation statistics for the Tier-2.5 single-point pass."""
        return {
            "single_point_enabled": bool(self.xtb_single_point),
            "calls": self._xtb_sp_calls,
            "cache_hits": self._xtb_sp_hits,
            "orca_eligible": self._xtb_sp_orca_eligible,
        }

    def execute(self) -> dict[str, Any]:
        wall_start = time.time()

        for generation in range(1, self.max_generations + 1):
            elapsed = time.time() - wall_start
            if elapsed > self.max_wall_time:
                log.info("Time cap reached (%.0fs). Exiting loop.", elapsed)
                break

            # Reset xtb escalation counter for new generation
            self._xtb_escalation_count = 0

            force_exploration = self.state.has_scaffold_stagnation(1)
            if force_exploration:
                log.info("Generation %d: Scaffold stagnation detected - pivoting to BRICS-only exploration.", generation)
                self._inject_tier0_seeds()
            if self._basin_stall >= 1 and generation >= 2:
                # Iterated local search: perturb the incumbent basin best
                # through the normal mutate/filter/evaluate path (BRICS
                # guard, novelty gate, grounding penalty all enforced
                # downstream) and accept the basin only on improvement.
                self._basin_hop(generation)
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

            # ADR-2026-08-08-02: Tier-2.5 mandatory xTB single-point on every
            # Tier-1 survivor before any ORCA escalation. A full geometry
            # optimisation (~2-5 s) is wasted on molecules that the SP reveals
            # to be non-viable; a single point (~0.3 s) gates them cheaply.
            # ORCA is reserved for the top decile of xTB-ranked survivors.
            batch_contexts, batch_scores = self._xtb_single_point_gate(
                batch_contexts, batch_scores
            )

            self._record_results(batch_contexts, batch_scores, generation)

            # W6: Periodically refit the delta-correction model with accumulated feedback
            if self.feedback_controller:
                refit_info = self.feedback_controller.maybe_refit(generation)
                if refit_info:
                    log.info(
                        "  Feedback refit #%d: LOO MAE %.4f → %.4f (%d new entries)",
                        self.feedback_controller.state.total_refits,
                        refit_info["loo_mae_before"],
                        refit_info["loo_mae_after"],
                        refit_info["new_calibration_entries"],
                    )
                # Log active learning budget utilization
                self.feedback_controller.log_budget_utilization(
                    generation=generation,
                    xtb_budget=self.xtb_budget_per_generation,
                    xtb_escalations=self._xtb_escalation_count,
                    xtb_successes=self._xtb_escalation_success,
                    threshold=self.active_learning_threshold,
                )

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
            "xtb_sp": self.xtb_sp_report,
        }

    def _inject_tier0_seeds(self) -> None:
        import json
        from importlib.resources import files
        tier0_path = files("aurelius.data") / "tier0_seed_smiles.json"
        if not tier0_path.exists():
            return
        with open(str(tier0_path)) as f:
            all_seeds = json.load(f)
        existing = set(self.engine.seed_pool)
        unused = [s for s in all_seeds if s not in existing]
        if unused:
            if getattr(self, "seed", 42) == 42:
                selected = self._rng.sample(unused, min(5, len(unused)))
            else:
                selected = _loop_stateless_sample(self.seed, "tier0_inject", "|".join(sorted(existing))[:64], unused, min(5, len(unused)))
            self.engine.seed_pool.extend(selected)
            log.info("  Injected %d fresh tier0 seed SMILES into seed pool.", len(selected))

    def _basin_hop(self, generation: int) -> None:
        """One iterated-local-search step from the incumbent basin center.

        Perturbs the best molecule seen so far with BRICS-only exploration
        (``force_exploration=True`` keeps the BRICS guard), then runs the
        standard filter/evaluate/record path so the novelty gate, grounding
        penalty, and Pareto + novel-scaffold archive all still apply. The
        basin center moves only on strict improvement (accept-if-better);
        otherwise the hop results still enrich the archive and the stall
        counter (updated in ``_record_results``) keeps growing.
        """
        if not self.all_results:
            return
        center = self._basin_center
        if center is None:
            return
        try:
            perturbed = self.engine.mutate(center, self.batch_size, force_exploration=True)
        except Exception:
            return
        if not perturbed:
            return
        valid_contexts, _invalid = self._filter_candidates(perturbed)
        if not valid_contexts:
            return
        batch_contexts, batch_scores = self._evaluate_and_select(valid_contexts)
        if not batch_contexts:
            return
        batch_contexts, batch_scores = self._xtb_single_point_gate(batch_contexts, batch_scores)
        self._record_results(batch_contexts, batch_scores, generation)
        if batch_scores and max(batch_scores) > self._basin_best:
            log.info("  Generation %d: basin hop accepted (best %.1f).", generation, max(batch_scores))

    def _generate_candidates(self, generation: int, force_exploration: bool = False) -> list[str]:
        top_seeds = self.engine.seed_pool if generation == 1 else self._top_seeds_from_results(generation)
        # Per-molecule budget equals the evaluation batch size so the budget
        # binds: the seed-governed reaction/pair order then decides which
        # products fill it, letting different benchmark seeds explore
        # complementary subspaces (multi-seed restarts diverge). An
        # over-generous multiple never binds, which collapses every seed to
        # the same deterministic enumeration.
        single_candidates = list(self.engine.mutate_batch(top_seeds, self.batch_size, force_exploration=force_exploration))

        # ADR-2026-08-08-03: Mixture fraction driven by mixture_mutation_rate.
        # Target: mixture candidates = rate * single_candidates, clamped to
        # ensure at least 30% of the total batch are mixtures when rate > 0.
        # Known electrolyte blends are seeded when mixture_seed_from_known=True.
        # Both binary and ternary mixtures are generated.
        if self.mixture_mutation_rate > 0.0:
            target_mixtures = int(len(single_candidates) * self.mixture_mutation_rate)
            min_mixtures = max(2, int(len(single_candidates) * 0.30))
            n_mixtures = max(min_mixtures, target_mixtures)
        else:
            n_mixtures = 0

        mixture_candidates = []
        if n_mixtures > 0:
            # Split: 2/3 binary, 1/3 ternary (ternary are more expensive to evaluate)
            n_binary = max(1, int(n_mixtures * 2 / 3))
            n_ternary = n_mixtures - n_binary
            mixture_candidates.extend(self.engine.propose_mixture_candidates(
                top_seeds,
                n_mixtures=n_binary,
                batch_size=5,
                seed_from_known=self.mixture_seed_from_known,
            ))
            mixture_candidates.extend(self.engine.propose_ternary_mixture_candidates(
                top_seeds,
                n_mixtures=n_ternary,
                batch_size=5,
            ))

        all_candidates = single_candidates + mixture_candidates
        # Gap 5: seed the known electrolyte set into the candidate stream so the
        # search's screenable population can include (and the benchmark can
        # recover) known electrolytes. These bypass mutate_batch's novelty gate
        # by construction - being in the stream is the point - and then pass
        # through the normal _filter_candidates validity checks only.
        # Opt-in: default False so the production search stays de-novo and the
        # philosophy tests (which measure mutation-novelty yield) are unaffected.
        if self.seed_knowns and generation == 1:
            all_candidates.extend(self.engine.known_smiles())
        if getattr(self, "seed", 42) == 42:
            self._rng.shuffle(all_candidates)
        else:
            _loop_stateless_shuffle(self.seed, "candidate_order", f"gen{generation}|{force_exploration}|{len(all_candidates)}", all_candidates)
        return all_candidates

    def _top_seeds_from_results(self, generation: int = 0) -> list[str]:
        scored = [(r.total_score, r.smiles) for r in self.all_results if r.total_score > 0]
        scored.sort(key=lambda x: -x[0])
        n = max(5, len(scored) // 5)
        # Elite plus seed-sampled rest: the top half exploits, while the
        # loop RNG samples the remainder from the ranked pool so different
        # benchmark seeds expand different parents (multi-seed restarts
        # diverge instead of collapsing to one deterministic trajectory).
        elite_n = max(2, n // 2)
        parents = [s for _, s in scored[:elite_n]]
        rest = [s for _, s in scored[elite_n:]]
        need = n - len(parents)
        if need > 0 and rest:
            # Thread the benchmark seed through parent sampling. The
            # default seed keeps the legacy stateful stream (exact pins
            # preserved). Other seeds use a stateless hash sample keyed
            # by (seed, generation, pool shape) so different seeds pick
            # different parents while same-seed reruns are identical
            # regardless of call history or process.
            if getattr(self, "seed", 42) == 42:
                parents += self._rng.sample(rest, min(need, len(rest)))
            else:
                parents += _loop_stateless_sample(
                    self.seed, "top_seeds",
                    f"gen{generation}|{len(scored)}|{need}",
                    rest, min(need, len(rest)),
                )
        return parents

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
            return
        smi_a, smi_b, frac = parsed
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return
        if not ctx_a.is_valid_electrolyte_mol():
            return
        if not ctx_b.is_valid_electrolyte_mol():
            return
        ctx_a.smiles = mixture_smi
        return ctx_a

    def _compute_novelty(self, ctx: MoleculeContext) -> float | None:
        """Compute novelty score (1 - max Tanimoto to seed pool)."""
        fp = ctx.get_ecfp4()
        seed_fps = getattr(self.engine, "seed_fingerprints", None)
        if not isinstance(seed_fps, list) or not seed_fps:
            return
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
            synthesis_depth=score_data.get("synthesis_depth"),
            sub_scores=sub_scores,
            combined_grounding_score=score_data.get("grounding"),
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
        complementarity - this intervention simply ensures the EA samples it.
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
        all_confidences: list[float] = []
        result_contexts: list[MoleculeContext] = []
        # Per-candidate objective scores for NSGA-II multi-objective selection.
        obj_scores: dict[str, list[float]] = {
            "total_score": [],
            "dielectric_proxy": [],
            "viscosity_proxy": [],
            "li_solvation_proxy": [],
            "homo_eV": [],
            "lumo_eV": [],
            "sa_score": [],
            "synthesizability_complexity": [],
            "synthesis_depth": [],
            "confidence": [],
            "combined_grounding_score": [],
            "novelty_to_seed": [],
        }
        result_map: dict[str, Any] = {}

        # Apply Tier-0 filtering
        filtered_contexts = self._apply_tier0_filter(valid_contexts)
        if not filtered_contexts:
            return [], []

        # Batch evaluation through the pipeline's batch oracle path. The
        # pipeline's screen_batch assembles per-molecule results key-identical
        # to screen_molecule and preserves input order; per-molecule failures
        # fall back to the scalar path internally.
        batch_results = self.pipeline.screen_batch(filtered_contexts)
        for ctx, result in zip(filtered_contexts, batch_results, strict=True):
            if result is None:
                continue

            smi = ctx.smiles
            score_data = result.get("score")
            if score_data is None:
                continue

            self.screened_smiles.add(smi)
            self.engine.add_to_db(smi)

            total_score = score_data.get("total_score", 0.0)
            self.engine.record_reaction_success(smi, total_score)
            all_scores.append(total_score)

            # Extract conformal confidence for uncertainty-aware selection
            conformal_conf = result.get("conformal_confidence", 1.0)
            all_confidences.append(conformal_conf)

            t2 = result.get("tier2", {}) or {}
            result_map[smi] = t2
            novelty = self._compute_novelty(ctx)
            sub_scores = score_data.get("sub_scores", {})

            # W7: Active learning escalation - re-evaluate low-confidence
            # TOM predictions with xTB to get more reliable results.
            total_score, conformal_conf, score_data, sub_scores, t2 = (
                self._maybe_escalate(
                    smi, ctx, total_score, conformal_conf, score_data, sub_scores, t2
                )
            )

            result_contexts.append(ctx)

            # Collect per-candidate objective scores for NSGA-II
            _collect_obj_scores(
                obj_scores, total_score, t2, score_data, conformal_conf, novelty or 0.0
            )

            # W6: Accumulate feedback for experimental/oracle refinement
            self._accumulate_feedback(smi, t2, total_score, conformal_conf)

            sr = self._build_screening_result(smi, total_score, score_data, t2, novelty, ctx, sub_scores)
            self._maybe_log_discovery(smi, total_score, conformal_conf, score_data, sr)

            self.all_results.append(sr)

        mix_contexts, mix_scores = self._evaluate_mixture_pairs(valid_contexts, result_map)
        result_contexts.extend(mix_contexts)
        all_scores.extend(mix_scores)
        # Pad objective scores and confidences for mixture candidates so
        # NSGA-II arrays align with the full candidate list.
        #
        # Mixtures are blends of components that were themselves screened, so
        # they are as makeable as their constituents. Padding grounding with
        # 0.0 would apply the maximum synthesizability penalty to every
        # mixture and silently suppress them; use a neutral 1.0 instead.
        for mix_score in mix_scores:
            all_confidences.append(1.0)
            for key in obj_scores:
                obj_scores[key].append(_mixture_pad_value(key, mix_score))

        if not result_contexts:
            return [], []

        return self._select_top_batch(
            result_contexts, all_scores, all_confidences, obj_scores, result_map
        )

    def _apply_tier0_filter(self, valid_contexts: list[MoleculeContext]) -> list[MoleculeContext]:
        """Apply Tier-0 prefiltering to already-built candidate contexts."""
        filtered_contexts, _ = self.tier0_prefilter.filter(valid_contexts)
        return filtered_contexts

    def _select_top_batch(
        self,
        contexts: list[MoleculeContext],
        scores: list[float],
        confidences: list[float],
        obj_scores: dict[str, list[float]],
        result_map: dict[str, Any] | None = None,
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Select the top batch via NSGA-II or tournament selection."""
        if len(contexts) <= self.batch_size:
            return contexts, scores
        return self._select_batch(contexts, scores, confidences, obj_scores, result_map)

    def _select_batch(
        self,
        contexts: list[MoleculeContext],
        scores: list[float],
        confidences: list[float],
        obj_scores: dict[str, list[float]],
        result_map: dict[str, Any] | None = None,
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Dispatch to NSGA-II or tournament selection."""
        if self.use_nsga2:
            # Consolidate 8 individual objectives into 4 composite objectives
            # for a denser, more physically interpretable Pareto front.
            composite_scores = build_npga2_composite_objectives(obj_scores)
            objectives = [
                ("ionic_transport", "max"),
                ("electronic_stability", "max"),
                ("synthetic_accessibility", "max"),
                ("chemical_complexity", "max"),
                ("synthesizability", "max"),
            ]
            smi_to_score = {ctx.smiles: sc for ctx, sc in zip(contexts, scores, strict=True)}
            selected = nsga2_select(
                contexts, composite_scores, objectives,
                batch_size=self.batch_size,
                confidences=confidences,
                rng_seed=self.seed,
            )
            selected_scores = [smi_to_score[c.smiles] for c in selected]
            return selected, selected_scores

        # Default: tournament selection with conformal confidence
        # Extract synergy_bonus from result_map if available
        synergy_bonus = []
        is_mixture = []
        for ctx in contexts:
            result = result_map.get(ctx.smiles) if result_map else None
            if result:
                is_mix = is_mixture_smiles(ctx.smiles)
                is_mixture.append(is_mix)
                if is_mix:
                    synergy = result.get("mixture_properties", {}).get("synergy_bonus", 0.0)
                    synergy_bonus.append(synergy)
                else:
                    synergy_bonus.append(0.0)
            else:
                is_mixture.append(False)
                synergy_bonus.append(0.0)

        selected = tournament_select(
            contexts,
            scores,
            batch_size=self.batch_size,
            confidences=confidences,
            synergy_bonus=synergy_bonus,
            is_mixture=is_mixture,
            grounding=obj_scores.get("combined_grounding_score"),
            rng_seed=self.seed,
        )
        smi_to_score = {ctx.smiles: sc for ctx, sc in zip(contexts, scores, strict=True)}
        selected_scores = [smi_to_score[c.smiles] for c in selected]
        return selected, selected_scores

    def _relative_evolution_bar(self, batch_scores: list[float]) -> float:
        """Relative evolution bar: top-decile cutoff of the current batch.

        Absolute DISCOVERY_THRESHOLD (65.0) is kept for discovery claims.
        Evolution fires on this relative bar so the seed pool evolves even
        when absolute scores never reach 65 (EA regime best ~21).
        """
        if not batch_scores:
            return float("inf")
        ordered = sorted(batch_scores, reverse=True)
        k = max(1, int(len(ordered) * EVOLUTION_RELATIVE_FRACTION))
        return ordered[k - 1]

    def _partition_seed_pool_for_truncation(
        self, pool: list[str], score_by_smi: dict[str, float]
    ) -> tuple[list[str], list[str]]:
        """Split pool into discoveries plus score-ranked remainder."""
        discoveries = [smi for smi in pool if score_by_smi.get(smi, 0.0) >= DISCOVERY_THRESHOLD]
        discovery_set = set(discoveries)
        rest = sorted((s for s in pool if s not in discovery_set),
                      key=lambda s: -score_by_smi.get(s, 0.0))
        return discoveries, rest

    def _fill_truncation_keep(self, discoveries: list[str], rest: list[str]) -> set[str]:
        """Fill keep-set with discoveries then top remainder up to cap."""
        keep: list[str] = list(discoveries)
        for smi in rest:
            if len(keep) >= SEED_POOL_MAX_SIZE:
                break
            keep.append(smi)
        return set(keep[:SEED_POOL_MAX_SIZE])

    def _enforce_truncation_cap(self, score_by_smi: dict[str, float]) -> None:
        """If discoveries alone exceed cap, keep top discoveries by score."""
        if len(self.engine.seed_pool) > SEED_POOL_MAX_SIZE:
            self.engine.seed_pool = sorted(
                self.engine.seed_pool, key=lambda s: -score_by_smi.get(s, 0.0)
            )[:SEED_POOL_MAX_SIZE]

    def _truncate_seed_pool_archive(self) -> None:
        """Cap seed pool score-aware: keep Pareto + novel-scaffold archive.

        Never truncate to last-200 recency only. Keeps all absolute
        discoveries (>= DISCOVERY_THRESHOLD), then top by score to fill
        SEED_POOL_MAX_SIZE. Novel scaffolds are protected because rotation
        re-injects them and evolution adds top-decile candidates; recency
        alone would drop early Pareto members.
        """
        pool = self.engine.seed_pool
        if len(pool) <= SEED_POOL_MAX_SIZE:
            self.state.seed_pool_size = len(pool)
            return
        scored = self._score_seed_pool()
        score_by_smi = {smi: sc for sc, smi in scored}
        discoveries, rest = self._partition_seed_pool_for_truncation(pool, score_by_smi)
        keep_set = self._fill_truncation_keep(discoveries, rest)
        self.engine.seed_pool = [smi for smi in pool if smi in keep_set]
        # If discoveries alone exceed cap (unlikely), keep top discoveries
        self._enforce_truncation_cap(score_by_smi)
        # Phase 3a invariant - archive persists across generations and is never
        # a recency-only last-200 slice. Fail closed if absolute discoveries are lost.
        if len(discoveries) <= SEED_POOL_MAX_SIZE:
            assert all(smi in self.engine.seed_pool for smi in discoveries), "archive must preserve absolute discoveries"
        assert len(self.engine.seed_pool) <= SEED_POOL_MAX_SIZE, "seed pool must respect cap"
        self.state.seed_pool_size = len(self.engine.seed_pool)

    def _evolve_seed_pool(self, batch_contexts: list[MoleculeContext], batch_scores: list[float]) -> None:
        """Feed high-scoring molecules back into the seed pool.

        Absolute bar (>= DISCOVERY_THRESHOLD) marks discovery claims.
        Relative bar (top decile of this batch) drives evolution so the pool
        moves even when absolute scores stay low.
        """
        relative_bar = self._relative_evolution_bar(batch_scores)
        for ctx, sc in zip(batch_contexts, batch_scores, strict=False):
            if is_mixture_smiles(ctx.smiles):
                continue
            is_absolute = sc >= DISCOVERY_THRESHOLD
            is_relative = sc >= relative_bar
            if not (is_absolute or is_relative):
                continue
            smi = ctx.smiles
            existing = set(self.engine.seed_pool)
            if smi not in existing:
                self.engine.seed_pool.append(smi)
                existing.add(smi)
            self.engine.harvest_fragments(smi, score=sc)
        self._truncate_seed_pool_archive()

    def _rotate_seed_pool(self, generation: int) -> None:
        """Seed rotation: every 3 generations, replace bottom 20% of seed pool with top novel-scaffold candidates.

        Gap 3 (G1): Prevents the seed pool from calcifying around early discoveries
        by injecting novel-scaffold candidates from the current generation's results.
        """
        if generation % 3 != 0:
            return
        if not self.all_results:
            return

        pool_size = len(self.engine.seed_pool)
        if pool_size == 0:
            return

        n_replace = max(1, int(0.20 * pool_size))
        existing_pool = set(self.engine.seed_pool)

        # Find novel-scaffold candidates from all_results not in pool, sorted by score
        known_scaffolds = self._collect_known_scaffolds()
        novel_candidates = self._find_novel_scaffold_candidates(existing_pool, known_scaffolds)
        top_novel = novel_candidates[:n_replace]

        if not top_novel:
            return

        # Replace bottom 20% of seed pool (lowest scored)
        pool_scored = self._score_seed_pool()
        to_remove = [smi for _, smi in pool_scored[:n_replace]]
        new_pool = [smi for smi in self.engine.seed_pool if smi not in to_remove]
        new_pool.extend([smi for _, smi, _ in top_novel])

        self.engine.seed_pool = new_pool
        self._truncate_seed_pool_archive()

        log.info(
            "  Generation %d: Seed rotation replaced %d molecules with novel scaffolds %s",
            generation, len(top_novel), [s for _, _, s in top_novel]
        )

    def _collect_known_scaffolds(self) -> set[str]:
        """Collect the Murcko scaffolds already present in the seed pool."""
        if MurckoScaffold is None:
            return set()
        known_scaffolds = set()
        for smi in self.engine.seed_pool:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                if scaffold:
                    known_scaffolds.add(scaffold)
            except Exception:
                pass
        return known_scaffolds

    def _find_novel_scaffold_candidates(
        self, existing_pool: set[str], known_scaffolds: set[str]
    ) -> list[tuple[float, str, str]]:
        """Return scored novel-scaffold candidates from current results, sorted by descending score.

        No absolute 65.0 gate: rotation uses the relative ranking so novel
        scaffolds are found even in low-score regimes. Discovery claims
        still require DISCOVERY_THRESHOLD elsewhere.
        """
        if MurckoScaffold is None:
            return []
        novel_candidates = []
        for r in self.all_results:
            if r.smiles in existing_pool:
                continue
            if is_mixture_smiles(r.smiles):
                continue
            ctx = MoleculeContext.from_smiles(r.smiles)
            if ctx is None:
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                if scaffold and scaffold not in known_scaffolds:
                    novel_candidates.append((r.total_score, r.smiles, scaffold))
            except Exception:
                pass
        novel_candidates.sort(key=lambda x: -x[0])
        return novel_candidates

    def _score_seed_pool(self) -> list[tuple[float, str]]:
        """Score each seed-pool molecule with its best available result, ascending."""
        pool_scored = []
        for smi in self.engine.seed_pool:
            scored_results = [r for r in self.all_results if r.smiles == smi]
            score = scored_results[0].total_score if scored_results else 0.0
            pool_scored.append((score, smi))
        pool_scored.sort(key=lambda x: x[0])
        return pool_scored

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
        self._rotate_seed_pool(generation)
        # Basin-hop stall accounting: strict improvement moves the basin
        # center and clears the stall; anything else deepens it. The hop
        # trigger in execute() reads _basin_stall.
        if batch_scores:
            _best_idx = max(range(len(batch_scores)), key=lambda i: batch_scores[i])
            if batch_scores[_best_idx] > self._basin_best:
                self._basin_best = float(batch_scores[_best_idx])
                self._basin_center = batch_contexts[_best_idx].smiles
                self._basin_stall = 0
            else:
                self._basin_stall += 1
        self._record_scaffolds(batch_contexts)
        self._record_scaffolds(batch_contexts)
        self.state.record_batch(batch_scores, batch_viable)

        mean_div = compute_pairwise_diversity(batch_contexts)

        # Retrosynthetic depth statistics (W7)
        depths = [r.synthesis_depth for r in self.all_results
                  if r.synthesis_depth is not None and r.synthesis_depth > 0]
        depth_msg = ""
        if depths:
            recent_depths = depths[-len(batch_contexts):] if len(depths) >= len(batch_contexts) else depths
            mean_depth = float(np.mean(recent_depths))
            min_depth = min(recent_depths)
            max_depth = max(recent_depths)
            depth_msg = f" depth[min={min_depth},μ={mean_depth:.1f},max={max_depth}]"

        log.info(
            "  Generation %d: %d screened, %d viable, best=%.1f, diversity=%.4f%s",
            generation,
            len(batch_contexts),
            batch_viable,
            max(batch_scores) if batch_scores else 0,
            mean_div,
            depth_msg,
        )

    def _screen_molecule(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        if is_mixture_smiles(ctx.smiles):
            return self._screen_mixture(ctx)
        try:
            result = self.pipeline.screen_molecule(ctx)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for %s: %s", ctx.smiles, e)
            return
        return result if result is not None else None

    def _screen_mixture(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        parsed = parse_mixture_smiles(ctx.smiles)
        if parsed is None:
            return
        smi_a, smi_b, frac = parsed
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return
        try:
            mix = self.pipeline.screen_mixture(ctx_a, ctx_b, frac)
        except (ImportError, ValueError, RuntimeError, TypeError) as e:
            log.warning("Pipeline error for mixture %s: %s", ctx.smiles, e)
            return
        if mix is None:
            return
        score = mix.get("score", {})
        mixture_props = mix.get("mixture_properties", {})
        return {
            "score": score,
            "tier2": mixture_props,
            "component1": mix.get("component1"),
            "component2": mix.get("component2"),
        }

    def _xtb_single_point_gate(
        self,
        contexts: list[MoleculeContext],
        scores: list[float],
    ) -> tuple[list[MoleculeContext], list[float]]:
        """Tier-2.5 mandatory xTB single-point on Tier-1 survivors.

        Every surviving candidate gets a fast GFN2-xTB single point (not a
        geometry optimisation) before the evolutionary loop decides which
        molecules graduate to the ORCA tier. The gate does two jobs:

        1. **Prune** candidates whose SP HOMO/LUMO disagree badly with the
           closed-form model - these are the molecules most likely to waste
           a full ORCA geometry optimisation. Molecules are *not* removed
           from the batch here; instead they are flagged and ranked below
           consistent ones, preserving biodiversity at low cost.
        2. **Rank** survivors by the xTB HOMO, so the downstream
           ``_select_top_batch`` and any ORCA escalation operate on
           quantum-grounded order rather than the closed-form surrogate.

        When xTB is unavailable (e.g. laptops, CI) the gate is a no-op and
        the survivors pass through untouched - graceful degradation.

        ADR-2026-08-08-02.
        """
        if not self.xtb_single_point:
            return contexts, scores

        from aurelius.scoring.oracle.xtb_single_point import XTBSinglePointOracle

        oracle = _LockedXTBSinglePointOracle(XTBSinglePointOracle())
        ranked = self._rank_by_xtb(contexts, scores, oracle._wrapped)
        # Top decile of xTB-ranked survivors is eligible for ORCA escalation;
        # the remainder is confirmed good enough by the SP alone.
        decile = max(1, len(ranked) // 10)
        self._xtb_sp_orca_eligible = decile
        for _ctx, _, flag in ranked[:decile]:
            flag["xtb_orca_eligible"] = True
        return [ctx for ctx, _, _ in ranked], [sc for _, sc, _ in ranked]

    def _rank_by_xtb(
        self,
        contexts: list[MoleculeContext],
        scores: list[float],
        oracle: XTBSinglePointOracle,
    ) -> list[tuple[MoleculeContext, float, dict[str, Any]]]:
        """Score each survivor against its cached xTB single point.

        Ordering key: prefer molecules whose xTB HOMO agrees with the LPM
        estimate (small residual), then by xTB HOMO itself (deeper HOMO =
        better oxidative stability). Disagreement beyond 1.5 eV demotes the
        molecule so ORCA does not inherit a confidently-wrong ranking.

        The per-molecule single points run in a thread pool - xTB subprocess
        calls release the GIL - while preserving input order. ``oracle`` is a
        thread-safe wrapper whose ``_persist`` (disk write of xtb_cache.json)
        is serialised, so concurrent cold calls cannot race on the cache file.
        """
        def _rank_one(
            pair: tuple[MoleculeContext, float],
        ) -> tuple[MoleculeContext, float, dict[str, Any]]:
            ctx, score = pair
            sp = oracle.evaluate(ctx)
            homo_sp = sp.get("homo_eV")
            flag: dict[str, Any] = {"xtb_orca_eligible": False}
            if homo_sp is not None:
                from aurelius.scoring.oracle.lone_pair import predict_lone_pair_homo

                flag["xtb_homo_eV"] = round(float(homo_sp), 6)
                flag["homo_residual_eV"] = round(
                    float(homo_sp - predict_lone_pair_homo(ctx.mol)), 6
                )
            return ctx, score, flag

        with ThreadPoolExecutor(max_workers=_EVAL_WORKERS) as pool:
            results = list(pool.map(_rank_one, zip(contexts, scores, strict=True)))
        # Stable sort: agreement first, then xTB HOMO descending.
        results.sort(
            key=lambda r: (
                -(1.0 if abs(r[2].get("homo_residual_eV") or 0.0) < 1.5 else 0.0),
                -(r[2].get("xtb_homo_eV") or -1e9),
            )
        )
        return results

    def _maybe_escalate(
        self,
        smi: str,
        ctx: MoleculeContext,
        total_score: float,
        conformal_conf: float,
        score_data: dict[str, Any],
        sub_scores: dict[str, Any],
        t2: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Escalate low-confidence TOM predictions to xTB evaluation.

        When TOM confidence is tom_low, xTB provides a more reliable evaluation.
        Returns updated scores and tier2 data.

        Budget enforcement: escalation is skipped when the per-generation
        xtb budget is exhausted. The active_learning_threshold is
        dynamically adjusted based on the escalation success rate.
        """
        # Enforce xtb budget per generation
        if self._xtb_escalation_count >= self.xtb_budget_per_generation:
            return total_score, conformal_conf, score_data, sub_scores, t2

        quantum_conf = t2.get("quantum_confidence", "unknown")
        if quantum_conf != "tom_low":
            return total_score, conformal_conf, score_data, sub_scores, t2

        xtb_result = self._evaluate_with_xtb(ctx)
        self._xtb_escalation_count += 1

        if xtb_result is None:
            return total_score, conformal_conf, score_data, sub_scores, t2

        self._xtb_escalation_success += 1
        xtb_t2 = xtb_result.get("tier2", {}) or {}
        xtb_score = xtb_result.get("score", {})
        xtb_total = xtb_score.get("total_score", total_score)
        xtb_conf = xtb_result.get("conformal_confidence", 1.0)
        log.info(
            "  ** ACTIVE LEARNING ESCALATION ** %s: TOM low confidence (conf=%.3f) -> xTB (conf=%.3f)",
            smi, conformal_conf, xtb_conf,
        )
        # ADR-2026-08-07-06: accumulate via the feedback controller (full
        # batch refit on the refit interval) instead of the deprecated
        # DeltaCorrection.update_online() point-update, which can corrupt
        # GPR state during long discovery runs.
        update_successful = self._accumulate_xtb_feedback(
            smi, t2, xtb_t2, xtb_total, xtb_conf,
        )

        if hasattr(self, "feedback_controller") and self.feedback_controller:
            self.feedback_controller.log_active_learning_trigger(
                smiles=smi,
                generation=self.state.generations,
                original_conf=conformal_conf,
                update_successful=update_successful,
            )

        # Dynamically adjust adaptive-learning threshold based on success rate
        self._adjust_al_threshold()

        self._escalation_history.append({
            "smiles": smi,
            "generation": self.state.generations,
            "success": xtb_result is not None,
            "original_conf": conformal_conf,
            "xtb_conf": xtb_conf,
        })

        return xtb_total, xtb_conf, xtb_score, xtb_score.get("sub_scores", {}), xtb_t2

    def _accumulate_xtb_feedback(
        self,
        smi: str,
        t2: dict[str, Any],
        xtb_t2: dict[str, Any],
        xtb_total: float,
        xtb_conf: float,
    ) -> bool:
        """Accumulate an xTB-verified point as experimental feedback.

        Replaces the deprecated ``DeltaCorrection.update_online()`` point
        update (ADR-2026-08-07-06): rather than mutating the GPR after every
        escalation, the point is stored for a periodic full refit via
        ``maybe_refit``.  Returns whether the feedback was accepted.
        """
        if not (hasattr(self, "feedback_controller") and self.feedback_controller):
            return False
        raw_tom = t2.get("raw_tom", {}) or {}
        homo_tom = raw_tom.get("homo_eV", t2.get("homo_eV", 0.0))
        lumo_tom = raw_tom.get("lumo_eV", t2.get("lumo_eV", 0.0))
        homo_dft = xtb_t2.get("homo_eV", 0.0)
        lumo_dft = xtb_t2.get("lumo_eV", 0.0)
        try:
            self.feedback_controller.accumulate(
                smiles=smi,
                homo_prediction=homo_tom,
                lumo_prediction=lumo_tom,
                homo_corrected=homo_dft,
                lumo_corrected=lumo_dft,
                total_score=xtb_total,
                conformal_confidence=xtb_conf,
                generation=self.state.generations,
            )
            return True
        except Exception:
            return False

    def _adjust_al_threshold(self) -> None:
        """Dynamically adjust the active-learning threshold from success rate.

        Low success (<30%) raises the bar to avoid chasing unproductive
        escalations; high success (>70%) lowers it to harvest more signal.
        """
        if self._xtb_escalation_count <= 0:
            return
        success_rate = self._xtb_escalation_success / self._xtb_escalation_count
        if success_rate < 0.3:
            self.active_learning_threshold = min(0.95, self.active_learning_threshold + 0.05)
        elif success_rate > 0.7:
            self.active_learning_threshold = max(0.5, self.active_learning_threshold - 0.05)

    def _accumulate_feedback(
        self,
        smi: str,
        t2: dict[str, Any],
        total_score: float,
        conformal_conf: float,
    ) -> None:
        """Accumulate feedback for experimental/oracle refinement."""
        if not (hasattr(self, "feedback_controller") and self.feedback_controller):
            return
        t2_raw = t2.get("raw_tom", {}) or {}
        self.feedback_controller.accumulate(
            smiles=smi,
            homo_prediction=t2_raw.get("homo_eV", t2.get("homo_eV", 0.0)),
            lumo_prediction=t2_raw.get("lumo_eV", t2.get("lumo_eV", 0.0)),
            homo_corrected=t2.get("homo_eV", 0.0),
            lumo_corrected=t2.get("lumo_eV", 0.0),
            total_score=total_score,
            conformal_confidence=conformal_conf,
            generation=self.state.generations,
            predicted_dielectric=t2.get("dielectric_proxy"),
            predicted_viscosity=t2.get("viscosity_proxy"),
        )

    def _maybe_log_discovery(
        self,
        smi: str,
        total_score: float,
        conformal_conf: float,
        score_data: dict[str, Any],
        sr: ScreeningResult,
    ) -> None:
        """Log and record discovery if the candidate qualifies."""
        if not self._is_discovery(total_score, score_data):
            return
        self.discoveries.append(sr)
        self.state.add_discovery(sr)
        log.info("  ** DISCOVERY ** %s (score=%.1f, confidence=%.4f)", smi, total_score, conformal_conf)

    def _evaluate_with_xtb(self, ctx: MoleculeContext) -> dict[str, Any] | None:
        """Re-evaluate a molecule using xTB quantum backend instead of TOM.

        When TOM confidence is low (tom_low) and conformal confidence is
        below the active learning threshold, xTB provides a more reliable
        evaluation. This is the core of the active learning escalation:
        low-confidence TOM predictions are automatically escalated to xTB
        for higher accuracy, maximizing information gain per compute dollar.

        Physical justification: TOM is a closed-form particle-in-a-box
        model that systematically mis-estimates HOMO/LUMO for molecules
        with non-trivial electronic structure. xTB (GFN2-xTB) is a
        semi-empirical quantum chemistry method that captures through-bond
        and through-space orbital interactions, providing more reliable
        predictions for novel scaffolds. The escalation ensures that
        uncertain TOM predictions do not mislead the evolutionary search.

        Args:
            ctx: MoleculeContext to evaluate with xTB.

        Returns:
            Screening result dict from xTB evaluation, or None if
            xTB is unavailable or evaluation fails.
        """
        from aurelius.scoring.oracle import has_xtb

        if not has_xtb():
            log.debug("xTB not available; cannot escalate %s", ctx.smiles)
            return

        try:
            if self._xtb_pipeline is None:
                self._xtb_pipeline = AureliusPipeline(use_real_models=True)
                self._xtb_pipeline.initialize()
            result = self._xtb_pipeline.screen_molecule(ctx)
            return result
        except Exception as exc:
            log.debug("xTB evaluation failed for %s: %s", ctx.smiles, exc)
            return


def multi_seed_restart_search(
    seeds: tuple[int, ...] | list[int] = (42, 123, 7, 2024, 99),
    n_generations: int = 5,
    batch_size: int = 50,
    seed_smiles: list[str] | None = None,
    max_wall_time: float = 120.0,
) -> dict[str, Any]:
    """Independent multi-seed restarts with a merged Pareto + novel archive.

    Each seed runs a fresh ``DiscoveryLoop`` (fresh engine, state, and
    pipeline) so restarts explore complementary subspaces instead of one
    trajectory. The merged archive dedupes by canonical SMILES keeping the
    best score, protecting Pareto members and novel scaffolds - never a
    last-N recency truncation. Gating uses the MINIMUM best-score across
    seeds (``min_best``), not the mean, matching the discovery benchmark.

    Args:
        seeds: Benchmark seeds; each gets an independent restart.
        n_generations: Generations per restart.
        batch_size: Evaluation batch size per restart.
        seed_smiles: Initial seed SMILES (defaults to the benchmark panel).
        max_wall_time: Wall-clock cap per restart in seconds.

    Returns:
        Dict with per-seed summaries, the merged archive, ``min_best``,
        and ``mean_best`` (reported for transparency only; the gate is
        the minimum).
    """
    from aurelius.agent.mutation import MutationEngine
    from aurelius.agent.state import LoopState
    from aurelius.pipeline import AureliusPipeline

    from rdkit import Chem

    panel = seed_smiles or ['COC(=O)OC', 'C1COCCO1', 'CS(=O)(=O)C', 'CC#N', 'C1CCOC1']
    per_seed: list[dict[str, Any]] = []
    merged: dict[str, float] = {}

    def _canon(smi: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol is not None else smi
        except Exception:
            return smi

    for seed in seeds:
        engine = MutationEngine(seed_smiles=list(panel), seed=seed)
        state = LoopState(path=f'/tmp/restart_state_{seed}.json')
        state.clear()
        pipeline = AureliusPipeline()
        pipeline.initialize()
        loop = DiscoveryLoop(
            pipeline=pipeline, engine=engine, state=state,
            max_generations=n_generations, batch_size=batch_size,
            max_wall_time=max_wall_time, seed_knowns=False, seed=seed,
        )
        result = loop.execute()
        best = 0.0
        for r in result.get('all_results', []):
            try:
                sc = float(r.total_score)
            except Exception:
                continue
            if sc > best:
                best = sc
            key = _canon(r.smiles)
            if key not in merged or sc > merged[key]:
                merged[key] = sc
        per_seed.append({
            'seed': seed,
            'best': round(best, 4),
            'n_screened': len(result.get('all_results', [])),
        })
    archive = sorted(merged.items(), key=lambda kv: -kv[1])
    bests = [p['best'] for p in per_seed] or [0.0]
    return {
        'seeds': list(seeds),
        'per_seed': per_seed,
        'merged_archive': [{'smiles': smi, 'score': round(sc, 4)} for smi, sc in archive],
        'merged_size': len(archive),
        'min_best': round(min(bests), 4),
        'mean_best': round(float(sum(bests) / len(bests)), 4),
    }
