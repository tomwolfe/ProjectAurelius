#!/usr/bin/env python3
"""Auto-generate docs/BENCHMARKS.md from live benchmark executions.

Usage:
    python scripts/update_benchmark_docs.py

Each benchmark is run with a per-process timeout so the script always
completes. Timed-out benchmarks produce a warning note in the output
instead of crashing the entire generation.

The external validation benchmark prints a ``__BENCHMARK_RESULTS__`` JSON
block at the end of stdout. This script parses that block to populate a
structured summary table with pass/fail status and trend arrows (compared
against the previous run stored in ``BENCHMARKS_HISTORY.json``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Per-benchmark timeouts (seconds). The reality check's own wall-time limit
# is set to 30s via AURELIUS_REALITY_WALL_TIME (vs 120s standalone).
_TIMEOUTS: dict[str, int] = {
    "benchmarks.benchmark_external_validation": 60,
    "benchmarks.benchmark_reality_check": 400,
    "benchmarks.benchmark_mixture_synergy": 30,
}

# Property labels and thresholds for the summary table
_PROPERTY_CONFIG: list[tuple[str, str, float]] = [
    ("dielectric_constant", "Dielectric ε", 0.5),
    ("viscosity_cP", "Viscosity η", 0.5),
    ("donor_number", "Donor Number", 0.5),
    ("homo_eV", "HOMO", 0.5),
    ("lumo_eV", "LUMO", 0.5),
]

# When running via subprocess we need src/ on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASE_ENV = os.environ.copy()
_BASE_ENV.setdefault("PYTHONPATH", str(_PROJECT_ROOT / "src"))
_BASE_ENV["AURELIUS_DOCS_MODE"] = "1"
_BASE_ENV["AURELIUS_REALITY_WALL_TIME"] = "210"

# Sentinel marker for structured results in benchmark output
_RESULTS_MARKER = "__BENCHMARK_RESULTS__"


def _capture(module: str) -> str:
    timeout = _TIMEOUTS.get(module, 60)
    try:
        return subprocess.check_output(
            [sys.executable, "-m", module],
            cwd=_PROJECT_ROOT,
            text=True,
            timeout=timeout,
            env=_BASE_ENV,
        )
    except subprocess.TimeoutExpired:
        return (
            f"[Benchmark timed out after {timeout}s — "
            f"rerun manually via `python -m {module}`]"
        )
    except subprocess.CalledProcessError as exc:
        return (
            f"[Benchmark failed with exit code {exc.returncode} — "
            f"{exc.stderr or 'no stderr output'}]"
        )


def _run_script(script_path: str) -> str:
    try:
        return subprocess.check_output(
            [sys.executable, script_path],
            cwd=_PROJECT_ROOT,
            text=True,
            timeout=60,
            env=_BASE_ENV,
        )
    except Exception as exc:
        print(f"Warning: {script_path} failed: {exc}", file=sys.stderr)
        return ""


def _check_module(module: str) -> bool:
    try:
        import importlib
        sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        sys.path.insert(0, str(_PROJECT_ROOT))
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _compute_scientific_yield() -> float:
    np.random.seed(42)
    import random
    random.seed(42)
    try:
        from aurelius.agent.mutation import MutationEngine
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        candidates = engine.propose_candidates(n_candidates=50, batch_size=25)

        seed_scaffolds: set[str] = set()
        for smi in engine.seed_pool:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if s:
                    seed_scaffolds.add(s)

        novel_count = 0
        total_screened = 0
        for smi in candidates:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            total_screened += 1
            try:
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if s and s not in seed_scaffolds:
                    novel_count += 1
            except Exception:
                continue

        if total_screened == 0:
            return 0.0
        return novel_count / total_screened
    except Exception:
        return 0.0


def _parse_benchmark_results(output: str) -> dict[str, Any] | None:
    """Parse ``__BENCHMARK_RESULTS__`` JSON block from benchmark stdout."""
    if _RESULTS_MARKER not in output:
        return None
    _, _, rest = output.partition(_RESULTS_MARKER)
    rest = rest.strip()
    try:
        return json.loads(rest)
    except json.JSONDecodeError:
        return None


def _load_history(history_path: Path) -> dict[str, Any]:
    if history_path.exists():
        try:
            with open(history_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": []}


def _save_history(history_path: Path, history: dict[str, Any], current: dict[str, Any]) -> None:
    history["runs"].append(current)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _compute_trend(
    current_val: float,
    current_key: str,
    history: dict[str, Any],
) -> str:
    """Return trend arrow (↑/↓/→) compared to the previous run's value."""
    runs = history.get("runs", [])
    if len(runs) < 2:
        return "—"
    prev = runs[-2].get("properties", {}).get(current_key, {})
    prev_val = prev.get("rho")
    if prev_val is None:
        return "—"
    diff = current_val - prev_val
    if diff > 0.01:
        return "↑"
    if diff < -0.01:
        return "↓"
    return "→"


def _generate_validation_table(
    results: dict[str, Any],
    history: dict[str, Any],
) -> str:
    """Generate a Markdown summary table for external validation results."""
    lines = [
        "### External Property Validation — Summary",
        "",
        "| Property | Spearman ρ | Threshold | Pass/Fail | Trend |",
        "|----------|-----------:|:----------|:----------|:------|",
    ]
    for exp_key, label, threshold in _PROPERTY_CONFIG:
        prop_data = results.get(exp_key, {})
        rho = prop_data.get("rho", 0.0)
        n = prop_data.get("n", 0)
        if n < 4:
            row = f"| {label} | N/A (n={n}) | ρ > {threshold} | ⚠ Insufficient | — |"
        else:
            passed = rho > threshold
            status = "✅ Pass" if passed else "❌ Fail"
            trend = _compute_trend(rho, exp_key, history)
            row = f"| {label} | {rho:+.4f} | ρ > {threshold} | {status} | {trend} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _generate_content(
    docs_dir: Path,
    brief_file: Path,
    history_path: Path,
) -> tuple[str, str | None]:
    """Generate the full BENCHMARKS.md content without writing it.

    Returns (content, output_file_path_as_str_or_None_for_error).
    """
    available: list[str] = []
    unavailable: list[str] = []
    for module in _TIMEOUTS:
        if _check_module(module):
            available.append(module)
        else:
            unavailable.append(module)

    ext_val_output = _capture("benchmarks.benchmark_external_validation") if "benchmarks.benchmark_external_validation" in available else "*Benchmark module not available — skip*"
    reality = _capture("benchmarks.benchmark_reality_check") if "benchmarks.benchmark_reality_check" in available else "*Benchmark module not available — skip*"
    mixture = _capture("benchmarks.benchmark_mixture_synergy") if "benchmarks.benchmark_mixture_synergy" in available else "*Benchmark module not available — skip*"

    script_path = str(Path(__file__).resolve().parent / "generate_synthesis_brief.py")
    _run_script(script_path)

    model_card_path = str(Path(__file__).resolve().parent / "generate_model_card.py")
    _run_script(model_card_path)

    sci_yield = _compute_scientific_yield()

    history = _load_history(history_path)

    ext_val_results = _parse_benchmark_results(ext_val_output)
    if ext_val_results is not None:
        ext_val_summary = _generate_validation_table(ext_val_results, history)
        current_run = {
            "timestamp": datetime.now(UTC).isoformat(),
            "properties": ext_val_results,
        }
        _save_history(history_path, history, current_run)
    else:
        ext_val_summary = "External validation results not available."

    parts = [
        "# Live Benchmark Results\n",
        "\n",
        "*Do not edit this file manually. It is auto-generated by `scripts/update_benchmark_docs.py`.*\n",
        "\n",
        "## External Property Validation\n",
        "\n",
        ext_val_summary,
        "\n",
        "\n",
        "### External Property Validation — Raw Output\n",
        "```text\n",
        ext_val_output,
        "```\n",
        "\n",
        "## Reality Check: EA Discoveries vs. Known Electrolytes\n",
        "```text\n",
        reality,
        "```\n",
        "\n",
        "## Mixture Synergy Validation\n",
        "```text\n",
        mixture,
        "```\n",
        "\n",
        "## Scientific Yield\n",
        "```text\n",
        f"Scientific yield (novel scaffolds / total screened): {sci_yield:.4f}\n",
        "```\n",
    ]

    if brief_file.exists():
        parts.append("\n## Synthesis Target Brief\n\n")
        parts.append("*Auto-generated — see [synthesis_brief.md](synthesis_brief.md) for full table.*\n")

    note = None
    if unavailable:
        note = f"Note: {len(unavailable)} benchmark(s) were unavailable (check PYTHONPATH):\n" + "\n".join(f"  - {m}" for m in unavailable)

    return "".join(parts), note


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark documentation.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether docs/BENCHMARKS.md is up-to-date. Exit with code 1 if changes are needed.",
    )
    args = parser.parse_args()

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    output_file = docs_dir / "BENCHMARKS.md"
    brief_file = docs_dir / "synthesis_brief.md"
    history_path = docs_dir / "BENCHMARKS_HISTORY.json"

    content, note = _generate_content(docs_dir, brief_file, history_path)

    if args.check:
        if output_file.exists():
            existing = output_file.read_text()
            if existing == content:
                print("Benchmark docs are up-to-date.")
                return
            print("ERROR: docs/BENCHMARKS.md is out of date.", file=sys.stderr)
            print("Run `python scripts/update_benchmark_docs.py` and commit the changes.", file=sys.stderr)
            sys.exit(1)
        else:
            print("ERROR: docs/BENCHMARKS.md does not exist. Run `python scripts/update_benchmark_docs.py` first.", file=sys.stderr)
            sys.exit(1)

    with open(output_file, "w") as f:
        f.write(content)

    print(f"Successfully updated {output_file}")
    if brief_file.exists():
        print(f"Synthesis brief: {brief_file}")

    if note:
        print(f"\n{note}")


if __name__ == "__main__":
    main()
