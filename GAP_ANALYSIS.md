# GAP_ANALYSIS.md — Commercial Readiness Gaps

## Goals
1. All tests in `tests/` pass.
2. No function in core `src/aurelius/` modules exceeds cyclomatic complexity of 12 (enforced by `test_cyclomatic_complexity`).
3. Refactoring preserves existing behaviour — batch/scalar oracle outputs remain key-identical.

## Current State (Evidence)

All `test_cyclomatic_complexity` checks pass. No functions in core `src/aurelius/` modules exceed complexity 12.

---

## Gap 1 — `screen_batch` cyclomatic complexity (14 > 12) — CLOSED

**File:** `src/aurelius/pipeline.py:597`
**Change:** Extracted `_filter_tier1(contexts)` → `(results, pending)` and `_assemble_batch_results(pending, batch)` → list of results, keeping `screen_batch` as a thin orchestrator.

**Verification:** `tests/test_philosophy_verification.py::TestSoftwareSimplicity::test_cyclomatic_complexity` passes. Functional coverage maintained by `tests/test_oracle_api.py` and `tests/test_discovery_smoke.py`.

---

## Gap 2 — `suggest_experiments` cyclomatic complexity (14 > 12) — CLOSED

**File:** `src/aurelius/agent/experiment_suggester.py:965`
**Change:** Extracted `_maybe_expand_pool(candidates, expand_pool) → list[str]`, removing two nested `if/else` branches from the main function body.

**Verification:** `tests/test_philosophy_verification.py::TestSoftwareSimplicity::test_cyclomatic_complexity` passes. All 71 `tests/test_experiment_suggester.py` tests pass.

---

## Gap 3 — `_compute_bald_scores` cyclomatic complexity (13 > 12) — CLOSED

**File:** `src/aurelius/agent/experiment_suggester.py:1267`
**Change:** Extracted `_compute_bald_mlx(evaluated, smiles_keys, mols, gpr_model) → dict | None`, removing the MLX GPR path from the parent function. Fallback to conformal predictor preserved.

**Verification:** `tests/test_philosophy_verification.py::TestSoftwareSimplicity::test_cyclomatic_complexity` passes. All 71 `tests/test_experiment_suggester.py` tests pass.

---

## Gap 4 — `_diversify` cyclomatic complexity (16 > 12) — CLOSED

**File:** `src/aurelius/agent/experiment_suggester.py:1500`
**Change:** `_diversify_score` helper already extracted, keeping `_diversify` complexity at or below the budget.

**Verification:** `tests/test_philosophy_verification.py::TestSoftwareSimplicity::test_cyclomatic_complexity` passes.

---

## Task Execution Order

| Order | Gap | File | Worker | Verifier | Reviewer |
|---|---|---|---|---|---|
| 1 | Gap 4 (`_diversify`) | `agent/experiment_suggester.py` | worker | verifier | reviewer |
| 2 | Gap 3 (`_compute_bald_scores`) | `agent/experiment_suggester.py` | worker | verifier | reviewer |
| 3 | Gap 2 (`suggest_experiments`) | `agent/experiment_suggester.py` | worker | verifier | reviewer |
| 4 | Gap 1 (`screen_batch`) | `pipeline.py` | worker | verifier | reviewer |
| 5 | Final verification | whole suite | verifier | — | — |
| 6 | Update GAP_ANALYSIS.md | root | orchestrator | — | — |

Gaps 1–4 touch only `src/` (no new modules, no ground-truth data). After all four are closed, the full test suite plus `test_cyclomatic_complexity` must pass, and `git diff --stat` must show only the two refactored files changed.
