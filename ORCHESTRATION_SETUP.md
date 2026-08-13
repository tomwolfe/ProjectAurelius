# Orchestration Setup — Project Aurelius

Date: 2026-08-12. OpenCode v1.18.17. Provider: OpenCode Zen **free tier** (keyless, shared rate-limit pool).

## What was created

| Path | Purpose |
|---|---|
| `.opencode/agent/orchestrator.md` | Primary agent. Plans, delegates, verifies, reports. |
| `.opencode/agent/analyst.md` | Subagent. Read-only analysis; writes GAP_ANALYSIS.md. |
| `.opencode/agent/worker.md` | Subagent. Implements one task + regression test, runs pytest. |
| `.opencode/agent/verifier.md` | Subagent. Read-only; runs pytest, checks diff scope, reports PASS/FAIL. |
| `.opencode/agent/reviewer.md` | Subagent. Read-only code review vs ADRs/philosophy. |
| `.opencode/opencode.json` | Task-permission gate: orchestrator may only call analyst/worker/verifier/reviewer. |
| `scripts/orchestrate_worktree.sh` | Creates an isolated git worktree per parallel task. |
| `GAP_ANALYSIS.md` | Produced by the analyst in the live test: 3 grounded commercial gaps. |
| `tests/test_reduction_oracle.py` | One regression test appended by the worker during the live test. |
| `.gitignore` | Added `.orchestrate/` (worktree dir). |

## Model assignments (all Zen free)

- orchestrator: `opencode/deepseek-v4-flash-free`
- worker: `opencode/deepseek-v4-flash-free`
- analyst / verifier / reviewer: `opencode/big-pickle`

## Verified end-to-end (live, Zen free tier)

1. **Dry run** — orchestrator invoked `@analyst`; returned a 5-line bottleneck summary; no files touched. PASS.
2. **Full loop** — `@analyst` wrote `GAP_ANALYSIS.md` (3 gaps, file:line evidence) → `@worker` appended a regression test asserting the solution-phase EA calibration is not the `(1.0, 0.0)` placeholder → `@verifier` ran `pytest tests/test_reduction_oracle.py -q`, returned a structured verdict: worker test FAILS as intended (regression signal — Gap 3 open), flagged a pre-existing `KeyError: 'grounding'` as out of scope, and recommended re-delegating to close the src-side fix. Correct orchestration behavior confirmed.

## Launch commands

```bash
# Full autonomous gap-closing run
opencode run --agent orchestrator --auto "Execute GAP_ANALYSIS.md to completion, verifying each gap."

# Parallel worktree worker (independent tasks)
dir=$(./scripts/orchestrate_worktree.sh close-gap3)
opencode run --dir "$dir" --agent worker --auto "<task spec>"

# Merge + cleanup
git worktree remove .orchestrate/close-gap3
git branch -d orchestrate/close-gap3
git worktree prune
```

Use `--auto` so non-interactive runs don't block on permission prompts. Workers never commit/push (denied in agent file); the orchestrator or you commit.

## First live execution (2026-08-12)

The orchestrator was dispatched on Gap 3. Outcome: **it refused to ship a bad fit** — exactly the verification behavior it was built for.

1. `@worker` ran `scripts/calibrate_reduction.py --solution` and fit `_EA_SOLUTION_CALIBRATION = (-0.3695, 4.7726)`.
2. `@verifier` caught the regression: the fit is **rank-reversing** (Spearman ρ = −0.82 vs the validated gas-phase ΔSCF axis ρ = 0.91) because the 10 CV-onset labels are inverted across chemical families. The Born correction cannot reorder across families, so any monotonic fit is anti-correlated with the labels.
3. Decision (best judgement, Option A): reverted the constants, fixed the false-confidence flag so solution-mode never claims `0.85` while uncalibrated, and rewrote the regression test to the honest invariant.
4. Also fixed a pre-existing `KeyError: 'grounding'` at `src/aurelius/pipeline.py:567` (objective read `raw_values["grounding"]` which was never set) — this was keeping the baseline suite red.

**Result:** `pytest tests/test_reduction_oracle.py` → 36 passed; full suite → 351 passed, 1 failed (pre-existing, unrelated: `test_generation_loop_discovers_novel_scaffolds`), 1 skipped. Nothing committed.

## Gap 1 execution (2026-08-12) — COMPLETE

Second orchestrator dispatch, executed on Zen free tier. Gap 1 (batch oracle into the loop) was implemented across `oracle.py`, `pipeline.py`, `loop.py` and verified:

- `predict_batch_properties` now returns the reduction axis (`ea_eV`, `reduction_records`).
- `screen_batch` is the single implementation; `screen_molecule` is a thin wrapper (paths cannot drift).
- Loop uses one `screen_batch` call per generation; `_rank_by_xtb` is threaded with a lock-protected cache.
- Regression test `test_batch_screening_matches_scalar_scores` passes. Full suite: **521 passed, 8 failed (all pre-existing on clean checkout), 2 skipped** — zero new regressions.
- Throughput with reduction axis: ~10.7 mol/s (xTB-bound). The ≥100 mol/s target remains open (needs a cheaper EA surrogate).

**Follow-up open:** Gap 2 (discovery benchmark gates) in `GAP_ANALYSIS.md`; reduction-axis throughput target for Gap 1.

## Gap 2 execution (2026-08-12) — COMPLETE

Third orchestrator dispatch, executed on Zen free tier. Gap 2 (discovery benchmark gates + confound audit) implemented across `benchmark_unified.py`, `audit_label_confound.py`, regenerated result JSONs/MD, and verified:

- `check_tolerances` now hard-fails when the `discovery` section is missing/empty ("discovery benchmark not run") instead of passing by default.
- `donor_number` added to the audited targets; `label_confound.json` regenerated.
- Full benchmark run without `--skip-discovery`; `unified_benchmark.json` discovery section now populated.
- Regression tests `test_discovery_gates_require_results`, `test_donor_number_audited` pass. Full suite: **525 passed, 8 failed (same pre-existing set), 2 skipped** — zero new regressions.

**Honest finding:** with the gates no longer skippable the benchmark **FAILs**: rediscovery 0.0 (0/51, canonical-SMILES verified — the loop genuinely does not reproduce known electrolytes), novelty 54.5% (< 80%), score gap +27.16 (passes). Donor Number audited as **confounded** (citation ρ 0.70, between-source fraction 0.51) — the "weak ρ 0.19" axis is partly a label artifact. Do not "fix" the FAIL by loosening gates; the fix is loop/mutation work.

**All three gaps are now executed.** Gap 3 recorded as BLOCKED (rank-reversing fit, decision A applied), Gap 1 COMPLETE, Gap 2 COMPLETE-with-honest-FAIL. Nothing committed.

## Known conditions

- **Zen free tier is keyless but shared-rate-limited.** `FreeUsageLimitError: rate limit exceeded` appears under contention; retrying succeeds (verified). Long parallel runs may hit daily free-usage caps — keep concurrency modest or add a $20+ Zen balance to raise free-model limits.
- **Local oMLX provider exists in the global config but was not used** (server intentionally stopped). Subagents override the global default model, so the orchestrator runs on Zen regardless.
- **Zero stored credentials.** Zen free tier works without an API key; no `opencode auth login` required.
