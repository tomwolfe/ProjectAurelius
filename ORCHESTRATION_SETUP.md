# Orchestration Setup — Project Aurelius

Date: 2026-08-12. OpenCode v1.18.18. Provider: OpenCode Zen **free tier** (keyless, shared rate-limit pool).

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

## Known conditions

- **Zen free tier is keyless but shared-rate-limited.** `FreeUsageLimitError: rate limit exceeded` appears under contention; retrying succeeds (verified). Long parallel runs may hit daily free-usage caps — keep concurrency modest or add a $20+ Zen balance to raise free-model limits.
- **Local oMLX provider exists in the global config but was not used** (server intentionally stopped). Subagents override the global default model, so the orchestrator runs on Zen regardless.
- **Zero stored credentials.** Zen free tier works without an API key; no `opencode auth login` required.
