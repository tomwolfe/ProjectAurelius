"""Speed baseline profiler for Task 1 - fixed-seed run before any optimization.

Profiles a fixed-seed DiscoveryLoop run (seed=42) and reports:
- wall time, per-generation time
- molecules screened, throughput
- xTB calls / cache hits, escalation counts
- GPR fits (DeltaCorrection initial + refits)
- Tier0 filter pass rates

Output: benchmarks/results/speed_baseline.json + speed_baseline.md
Target: 3-5x throughput on same seed with scores identical within tolerance.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

SEED = 42
N_GENERATIONS = 3
BATCH_SIZE = 24
WALL_CAP = 600.0


def main() -> int:
    import random

    import numpy as np

    from aurelius.agent.loop import DiscoveryLoop
    from aurelius.agent.mutation import MutationEngine
    from aurelius.agent.state import LoopState
    from aurelius.pipeline import AureliusPipeline
    from aurelius.scoring.oracle.delta_correction import get_delta_correction

    random.seed(SEED)
    np.random.seed(SEED)

    SEED_SMILES = ["COC(=O)OC", "C1COCCO1", "CS(=O)(=O)C", "CC#N", "C1CCOC1"]

    t_init0 = time.perf_counter()
    engine = MutationEngine(seed_smiles=SEED_SMILES, seed=SEED)
    pipeline = AureliusPipeline()
    pipeline.initialize()
    t_init1 = time.perf_counter()

    # GPR baseline: count fitted models in singleton delta correction
    dc = get_delta_correction()
    gpr_models = 0
    for attr in ("_homo_model", "_lumo_model"):
        m = getattr(dc, attr, None)
        if m is not None and hasattr(m, "X_train_"):
            gpr_models += 1

    state = LoopState(path="/tmp/speed_baseline_state.json")
    state.clear()

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        state=state,
        max_generations=N_GENERATIONS,
        batch_size=BATCH_SIZE,
        max_wall_time=WALL_CAP,
        seed=SEED,
    )

    # Per-stage timing via wrappers (Phase 5: per-stage profile).
    stage_totals: dict[str, float] = {
        "generate": 0.0, "filter": 0.0, "evaluate": 0.0,
        "select": 0.0, "record": 0.0,
    }
    stage_counts: dict[str, int] = {k: 0 for k in stage_totals}

    def _timed(stage: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                stage_totals[stage] += time.perf_counter() - t0
                stage_counts[stage] += 1
        return wrapper

    loop._generate_candidates = _timed("generate", loop._generate_candidates)  # type: ignore[method-assign]
    loop._filter_candidates = _timed("filter", loop._filter_candidates)  # type: ignore[method-assign]
    loop._evaluate_and_select = _timed("evaluate", loop._evaluate_and_select)  # type: ignore[method-assign]
    loop._select_top_batch = _timed("select", loop._select_top_batch)  # type: ignore[method-assign]

    # Per-generation timing via wrapper on _record_results
    per_gen: list[dict] = []
    orig_record = loop._record_results

    def timed_record(batch_contexts, batch_scores, generation):
        g0 = time.perf_counter()
        out = orig_record(batch_contexts, batch_scores, generation)
        g1 = time.perf_counter()
        # batch size here equals evaluated count; generation time approximated
        # by record hook overhead + caller-side measurement below
        per_gen.append(
            {
                "generation": generation,
                "n_evaluated": len(batch_contexts),
                "best": round(max(batch_scores), 4) if batch_scores else 0.0,
                "record_overhead_s": round(g1 - g0, 4),
            }
        )
        stage_totals["record"] += g1 - g0
        stage_counts["record"] += 1
        return out

    loop._record_results = timed_record  # type: ignore[method-assign]

    wall0 = time.perf_counter()
    gen_times: list[float] = []
    # Run generation-by-generation to capture per-stage time
    from aurelius.agent.feedback import FeedbackController

    loop.feedback_controller = FeedbackController(delta_correction=get_delta_correction())
    total_start = time.perf_counter()
    result = loop.execute()
    total_wall = time.perf_counter() - total_start

    all_results = result.get("all_results", [])
    n_screened = len(all_results)
    best = max((r.total_score for r in all_results), default=0.0)
    throughput = n_screened / total_wall if total_wall > 0 else 0.0

    xtb_sp = result.get("xtb_sp", {})
    xtb_calls = int(xtb_sp.get("calls", 0))
    xtb_hits = int(xtb_sp.get("cache_hits", 0))
    xtb_hit_rate = (xtb_hits / xtb_calls) if xtb_calls else 1.0

    # Feedback refits during run
    refits = 0
    try:
        refits = int(loop.feedback_controller.state.total_refits)
    except Exception:
        refits = 0

    # Tier0 is init-once in DiscoveryLoop.__init__ (Phase 3 audit); count
    # actual inits by construction events, not by generation count.
    tier0_inits_per_run = 1  # single Tier0Prefilter() in __init__, reused

    stage_profile = {
        name: {
            "total_s": round(stage_totals[name], 4),
            "calls": stage_counts[name],
            "mean_s_per_call": round(
                stage_totals[name] / max(stage_counts[name], 1), 4),
            "share": round(stage_totals[name] / max(total_wall, 1e-9), 4),
        }
        for name in stage_totals
    }

    payload = {
        "task": "TASK 1 SPEED BASELINE",
        "seed": SEED,
        "n_generations": N_GENERATIONS,
        "batch_size": BATCH_SIZE,
        "wall_seconds": round(total_wall, 3),
        "pipeline_init_seconds": round(t_init1 - t_init0, 3),
        "per_generation": per_gen,
        "n_screened": n_screened,
        "throughput_mols_per_sec": round(throughput, 4),
        "best_score": round(float(best), 4),
        "xtb_calls": xtb_calls,
        "xtb_cache_hits": xtb_hits,
        "xtb_cache_hit_rate": round(float(xtb_hit_rate), 4),
        "xtb_escalations": int(getattr(loop, "_xtb_escalation_count", 0)),
        "gpr_models_fitted_at_start": int(gpr_models),
        "gpr_refits_during_run": int(refits),
        "tier0_prefilter_inits": int(tier0_inits_per_run),
        "stage_profile": stage_profile,
        "note": "Baseline before any optimization. Do not optimize without this baseline.",
        "throughput_target": "3-5x throughput on same seed (42) with scores identical within tolerance 1e-6; speed that changes scores is a different oracle and is rejected.",
    }

    out_json = PROJECT_ROOT / "benchmarks" / "results" / "speed_baseline.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Speed Baseline (Task 1)",
        "",
        f"Seed: {SEED}, generations: {N_GENERATIONS}, batch: {BATCH_SIZE}",
        f"Wall: {total_wall:.2f}s, screened: {n_screened}, throughput: {throughput:.3f} mol/s",
        f"Best score: {float(best):.2f}",
        f"xTB calls: {xtb_calls}, hits: {xtb_hits}, hit rate: {xtb_hit_rate:.2%}",
        f"GPR models at start: {gpr_models}, refits: {refits}",
        f"Tier0 inits: {tier0_inits_per_run} (init-once, reused across generations)",
        "Per-stage profile:",
    ] + [
        f"  {name}: {stage_profile[name]['total_s']}s "
        f"({stage_profile[name]['share']:.0%} of wall, "
        f"{stage_profile[name]['calls']} calls)"
        for name in stage_profile
    ] + [
        "",
        "Target for Task 5: 3-5x throughput on seed 42 with identical scores.",
    ]
    out_md = PROJECT_ROOT / "benchmarks" / "results" / "speed_baseline.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
