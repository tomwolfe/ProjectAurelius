#!/usr/bin/env bash
#
# orchestrate_worktree.sh — Create an isolated git worktree for a parallel
# agent task, so multiple workers can edit the same repo without trampling
# each other's changes.
#
# Usage:
#   ./scripts/orchestrate_worktree.sh <task-name>
#
# Behavior:
#   - Creates branch orchestrate/<task-name> from the current HEAD.
#   - Adds worktree at .orchestrate/<task-name>.
#   - Prints the worktree directory on stdout for the caller to capture.
#
# Example:
#   dir=$(./scripts/orchestrate_worktree.sh fix-lumo-calibration)
#   opencode run --dir "$dir" --agent worker --auto "fix LUMO calibration..."
#
# Cleanup (after merging the branch):
#   git worktree remove .orchestrate/<task-name>
#   git branch -d orchestrate/<task-name>
#   git worktree prune

set -euo pipefail

TASK_NAME="${1:?usage: orchestrate_worktree.sh <task-name>}"

BRANCH="orchestrate/${TASK_NAME}"
WORKTREE=".orchestrate/${TASK_NAME}"

if git worktree list --porcelain | grep -q "worktree .*${WORKTREE}$"; then
  echo "Worktree already exists: ${WORKTREE}" >&2
  echo "${WORKTREE}"
  exit 0
fi

git worktree add -b "${BRANCH}" "${WORKTREE}" HEAD
echo "${WORKTREE}"
