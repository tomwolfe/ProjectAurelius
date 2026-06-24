#!/usr/bin/env python3
"""Auto-generate docs/benchmarks.md from live benchmark executions.

Usage:
    python scripts/update_benchmark_docs.py

Each benchmark is run with a per-process timeout so the script always
completes. Timed-out benchmarks produce a warning note in the output
instead of crashing the entire generation.
"""

import os
import subprocess
import sys
from pathlib import Path

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

# When running via subprocess we need src/ on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASE_ENV = os.environ.copy()
_BASE_ENV.setdefault("PYTHONPATH", str(_PROJECT_ROOT / "src"))
_BASE_ENV["AURELIUS_DOCS_MODE"] = "1"
_BASE_ENV["AURELIUS_REALITY_WALL_TIME"] = "210"


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
    """Check if a benchmark module exists and is importable."""
    try:
        import importlib
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _compute_scientific_yield() -> float:
    """Compute scientific yield as novel_scaffold_count / total_screened.

    Runs a quick mutation engine proposal and screens a batch of candidates
    to measure how many novel Murcko scaffolds are discovered per molecule
    screened. This is a proxy for the EA's chemical exploration efficiency,
    reported separately from net_progress to decouple scientific from
    code-simplicity metrics.
    """
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


def main() -> None:
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    output_file = docs_dir / "benchmarks.md"
    brief_file = docs_dir / "synthesis_brief.md"

    available: list[str] = []
    unavailable: list[str] = []
    for module in _TIMEOUTS:
        if _check_module(module):
            available.append(module)
        else:
            unavailable.append(module)

    if unavailable:
        print(f"Warning: benchmark modules not found: {', '.join(unavailable)}", file=sys.stderr)

    ext_val = _capture("benchmarks.benchmark_external_validation") if "benchmarks.benchmark_external_validation" in available else "*Benchmark module not available — skip*"
    reality = _capture("benchmarks.benchmark_reality_check") if "benchmarks.benchmark_reality_check" in available else "*Benchmark module not available — skip*"
    mixture = _capture("benchmarks.benchmark_mixture_synergy") if "benchmarks.benchmark_mixture_synergy" in available else "*Benchmark module not available — skip*"

    # Generate synthesis brief
    script_path = str(Path(__file__).resolve().parent / "generate_synthesis_brief.py")
    _run_script(script_path)

    # Auto-regenerate model card so it never goes stale
    model_card_path = str(Path(__file__).resolve().parent / "generate_model_card.py")
    _run_script(model_card_path)

    sci_yield = _compute_scientific_yield()

    parts = [
        "# Live Benchmark Results\n",
        "\n",
        "*Do not edit this file manually. It is auto-generated by `scripts/update_benchmark_docs.py`.*\n",
        "\n",
        "## External Property Validation\n",
        "```text\n",
        ext_val,
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

    with open(output_file, "w") as f:
        f.writelines(parts)

    if brief_file.exists():
        with open(output_file, "a") as f:
            f.write("\n## Synthesis Target Brief\n\n")
            f.write("*Auto-generated — see [synthesis_brief.md](synthesis_brief.md) for full table.*\n")

    print(f"Successfully updated {output_file}")
    if brief_file.exists():
        print(f"Synthesis brief: {brief_file}")

    if unavailable:
        print(f"\nNote: {len(unavailable)} benchmark(s) were unavailable (check PYTHONPATH):")
        for m in unavailable:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
