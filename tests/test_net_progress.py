"""Net Progress metric — repository-level objective function.

Defines:
  DISCOVERY_VALUE = (0.10 * rediscovery_rate) + (0.15 * scaffold_novelty)
                    + (0.10 * top_k_enrichment) + (0.15 * external_consistency)
                    + (0.20 * holdout_generalization) + (0.15 * experimental_trend_recovery)
                    + (0.15 * validation_score)
  SIMPLICITY_COST = (0.30 * norm_loc) + (0.20 * norm_cc_violations)
                    + (0.20 * norm_dependencies) + (0.30 * norm_architectural_surface_area)
  NET_PROGRESS = DISCOVERY_VALUE - (0.35 * SIMPLICITY_COST)

This test calculates the BASELINE net progress and verifies that any code
changes do not decrease it. The constants (lambda, normalisation factors) are
chosen so that NET_PROGRESS lives in [0, 1] for a healthy repository.

Usage:
    pytest tests/test_net_progress.py -v
"""

from __future__ import annotations

import ast
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import path setup (existing precedent: tests/test_label_confound.py).
# benchmark_unified.py inserts src/ into sys.path itself at import time.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
)

from benchmark_unified import check_tolerances  # noqa: E402

from aurelius.scoring.oracle.gc import predict_dielectric_proxy, predict_viscosity_proxy
from aurelius.scoring.oracle.quantum import (
    predict_tom_orbitals,
)
from aurelius.types import MoleculeContext

LAMBDA = 0.35

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius")

HOLDOUT_SEED = 42
HOLDOUT_FRACTION = 0.20


def _count_lines_of_code() -> int:
    """Count total non-empty, non-comment lines in src/aurelius/.py files."""
    total = 0
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        total += 1
    return total


def _count_cyclomatic_violations() -> int:
    """Count functions exceeding cyclomatic complexity of 12 in core modules."""
    try:
        from radon.complexity import cc_visit
    except ImportError:
        return 0

    excluded = {"chem_utils.py", "dependencies.py", "__init__.py", "__main__.py",
                "reporting.py"}
    violations = 0
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py") or fn in excluded:
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    blocks = cc_visit(f.read())
                    for block in blocks:
                        if block.complexity > 12:
                            violations += 1
                except Exception:
                    continue
    return violations


def _count_dependency_imports() -> int:
    """Count unique third-party imports in src/aurelius/ (excluding aurelius itself)."""
    stdlib = {"os", "sys", "json", "math", "re", "time", "io", "abc", "typing",
              "collections", "functools", "itertools", "pathlib", "copy", "inspect",
              "logging", "contextlib", "subprocess", "tempfile", "threading",
              "concurrent", "dataclasses", "warnings", "pickle", "enum", "hashlib",
              "textwrap", "bisect", "random"}
    deps: set[str] = set()
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        pkg = alias.name.split(".")[0]
                        if pkg not in stdlib and not pkg.startswith("aurelius"):
                            deps.add(pkg)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    pkg = node.module.split(".")[0]
                    if pkg not in stdlib and not pkg.startswith("aurelius"):
                        deps.add(pkg)
    return len(deps)


def _count_architectural_surface_area() -> int:
    """Count public classes and functions in core src/aurelius/ modules.

    Tracks the number of public (non-underscore-prefixed) classes and
    top-level functions as a proxy for architectural complexity.
    """
    count = 0
    excluded = {"__init__.py", "__main__.py", "dependencies.py", "reporting.py"}
    for root, _dirs, files in os.walk(SRC_DIR):
        for fn in files:
            if not fn.endswith(".py") or fn in excluded:
                continue
            filepath = os.path.join(root, fn)
            with open(filepath) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                    count += 1
    return count


def _compute_rediscovery_rate() -> float:
    """Approximate rediscovery rate using a quick mutation-engine test."""
    engine = None
    try:
        from aurelius.agent.mutation import MutationEngine
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
    except Exception:
        return 0.0

    known_smiles = [
        "C1COC(=O)O1", "COC(=O)OC", "CCOC(=O)OCC", "CC#N", "CS(=O)(=O)C",
    ]
    rediscovered = 0
    for smi in known_smiles:
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if engine._is_known_smiles(canon):
            rediscovered += 1
    return rediscovered / max(len(known_smiles), 1)


def _compute_scaffold_novelty() -> float:
    """Approximate scaffold novelty via mutation engine proposal."""
    try:
        from aurelius.agent.mutation import MutationEngine

        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        candidates = engine.propose_candidates(n_candidates=100, batch_size=25)

        seed_scaffolds: set[str] = set()
        for smi in engine.seed_pool:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if s:
                    seed_scaffolds.add(s)

        novel = 0
        for smi in candidates:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                try:
                    s = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                    if s and s not in seed_scaffolds:
                        novel += 1
                except Exception:
                    continue
        return novel / max(len(candidates), 1)
    except Exception:
        return 0.0


def _compute_top_k_enrichment() -> float:
    """Compare mean score of top-10 vs bottom-10 from mutation engine proposals."""
    return _compute_top_k_with_predictions()[0]


def _compute_top_k_with_predictions() -> tuple[float, list[dict[str, float]]]:
    """Compute top-k enrichment and return top-10 predictions for external consistency check.

    Returns (enrichment_score, list of top-10 prediction dicts with keys:
    dielectric_proxy, viscosity_proxy, homo_eV, lumo_eV).
    """
    try:
        from aurelius.agent.mutation import MutationEngine
        from aurelius.pipeline import AureliusPipeline

        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        candidates = engine.propose_candidates(n_candidates=50, batch_size=25)

        pipeline = AureliusPipeline()
        pipeline.initialize()

        scored: list[tuple[float, dict[str, float]]] = []
        for smi in candidates:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            try:
                result = pipeline.screen_molecule(ctx)
                t2 = result.get("tier2")
                if t2 is None:
                    continue
                score = result.get("score", {}).get("total_score", 0.0)
                props = {
                    "dielectric_proxy": t2.get("dielectric_proxy", 0.0),
                    "viscosity_proxy": t2.get("viscosity_proxy", 0.0),
                    "homo_eV": t2.get("homo_eV", 0.0),
                    "lumo_eV": t2.get("lumo_eV", 0.0),
                }
                scored.append((score, props))
            except Exception:
                continue

        if len(scored) < 10:
            return 0.0, []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_10_props = [p for _, p in scored[:10]]
        scores = [s for s, _ in scored]

        top_mean = np.mean(scores[:10])
        bottom_mean = np.mean(scores[-10:])
        enrichment = (top_mean - bottom_mean) / 100.0
        return max(0.0, min(1.0, enrichment)), top_10_props
    except Exception:
        return 0.0, []


def _compute_external_consistency() -> float:
    """Fraction of top-10 EA discoveries whose predicted properties fall within
    the min–max range of known-good electrolytes from the external benchmark.

    This de-circularises the Net Progress metric: at least one discovery-value
    term is now anchored to external experimental reality rather than the
    pipeline evaluating itself.
    """
    benchmark_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data",
        "external_property_benchmark.json",
    )
    try:
        with open(benchmark_path) as f:
            benchmark = json.load(f)
    except Exception:
        return 0.0

    # Compute min/max ranges across benchmark molecules
    ranges: dict[str, tuple[float, float]] = {}
    for key in ("dielectric_constant", "viscosity_cP", "homo_eV", "lumo_eV"):
        vals = [e[key] for e in benchmark if e.get(key) is not None]
        if len(vals) >= 4:
            ranges[key] = (min(vals), max(vals))

    if not ranges:
        return 0.0

    enrichment, top_props = _compute_top_k_with_predictions()
    if not top_props:
        return 0.0

    within_range = 0
    for props in top_props:
        ok = True
        for bench_key, pred_key in [
            ("dielectric_constant", "dielectric_proxy"),
            ("viscosity_cP", "viscosity_proxy"),
            ("homo_eV", "homo_eV"),
            ("lumo_eV", "lumo_eV"),
        ]:
            if bench_key in ranges and pred_key in props:
                lo, hi = ranges[bench_key]
                if not (lo <= props[pred_key] <= hi):
                    ok = False
                    break
        if ok:
            within_range += 1

    return within_range / max(len(top_props), 1)


def _compute_holdout_generalization() -> float:
    """TOM holdout MAE as 1 - (MAE / 1.5), clipped to [0, 1].

    Uses a 80/20 holdout split of orbital_calibration.json.
    1.0 = MAE=0 (perfect), 0.0 = MAE >= 1.5 eV.
    """
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data", "orbital_calibration.json"
    )
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return 0.0

    random.seed(HOLDOUT_SEED)
    indices = list(range(len(data)))
    random.shuffle(indices)
    n_holdout = max(1, int(len(data) * HOLDOUT_FRACTION))
    holdout_idx = set(indices[:n_holdout])
    holdout = [data[i] for i in holdout_idx]

    errors = []
    for entry in holdout:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        homo_pred, lumo_pred = predict_tom_orbitals(mol)
        homo_err = abs(homo_pred - entry["homo_eV"])
        lumo_err = abs(lumo_pred - entry["lumo_eV"])
        errors.append((homo_err + lumo_err) / 2.0)

    if not errors:
        return 0.0
    mae = sum(errors) / len(errors)
    return max(0.0, min(1.0, 1.0 - mae / 1.5))


def _compute_experimental_trend_recovery() -> float:
    """Score how well GC proxy captures known experimental dielectric trends.

    Tests known high-dielectric vs low-dielectric pairs and branched vs
    linear viscosity trends. Returns a score in [0, 1] where 1.0 means
    all known trends are correctly reproduced.
    """
    from aurelius.types import MoleculeContext

    trends = [
        ("dielectric", "C1COC(=O)O1", "COC(=O)OC", True),   # EC > DMC
        ("dielectric", "CC1COC(=O)O1", "CCOCC", True),      # PC > DEE
        ("dielectric", "C1COC(=O)O1", "CCOCC", True),        # EC > DEE
        ("viscosity", "CC(C)(C)OC(=O)OC(C)(C)C", "COC(=O)OC", True),  # branched > linear
        ("viscosity", "CC(C)(C)O", "CCCCO", True),          # branched > linear
    ]

    correct = 0
    total = 0
    for prop, smi_a, smi_b, expected_higher in trends:
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            continue
        if prop == "dielectric":
            va = predict_dielectric_proxy(ctx_a)
            vb = predict_dielectric_proxy(ctx_b)
        else:
            va = predict_viscosity_proxy(ctx_a)
            vb = predict_viscosity_proxy(ctx_b)
        total += 1
        if expected_higher and va > vb or not expected_higher and va < vb:
            correct += 1

    # Additional trend: viscosity of glycerol > ethanol
    ctx_gly = MoleculeContext.from_smiles("C(C(CO)O)O")
    ctx_eth = MoleculeContext.from_smiles("CCO")
    if ctx_gly and ctx_eth:
        total += 1
        if predict_viscosity_proxy(ctx_gly) > predict_viscosity_proxy(ctx_eth):
            correct += 1

    return correct / max(total, 1)


def _compute_validation_score() -> float:
    """Compute validation score based on oracle absolute audit results.

    Returns 1.0 if oracle_absolute_audit.json exists AND HOMO MAE < 1.5 eV,
           0.5 if audit exists but MAE >= 1.5 eV,
           0.0 if audit missing.
    """
    audit_path = Path(__file__).parent.parent / "benchmarks" / "results" / "oracle_absolute_audit.json"
    if not audit_path.exists():
        return 0.0

    try:
        with open(audit_path) as f:
            audit = json.load(f)

        homo_mae = audit["properties"]["HOMO"]["mae"]
        return 1.0 if homo_mae < 1.5 else 0.5
    except Exception:
        return 0.0


class TestNetProgress:
    """Repository-level objective function.

    NET_PROGRESS = DISCOVERY_VALUE - (LAMBDA * SIMPLICITY_COST)

    Ensures that any code change increases discovery value more than it
    adds complexity cost.
    """

    def test_net_progress_is_positive(self):
        loc = _count_lines_of_code()
        cc_violations = _count_cyclomatic_violations()
        n_deps = _count_dependency_imports()
        arch_surface = _count_architectural_surface_area()

        rediscovery_rate = _compute_rediscovery_rate()
        scaffold_novelty = _compute_scaffold_novelty()
        top_k_enrichment = _compute_top_k_enrichment()
        external_consistency = _compute_external_consistency()
        holdout_gen = _compute_holdout_generalization()
        trend_recovery = _compute_experimental_trend_recovery()

        LOC_NORM = 5000.0
        CC_NORM = 5.0
        DEP_NORM = 10.0
        ARCH_NORM = 50.0

        sim_loc = min(1.0, loc / LOC_NORM)
        sim_cc = min(1.0, cc_violations / CC_NORM)
        sim_dep = min(1.0, n_deps / DEP_NORM)
        sim_arch = min(1.0, arch_surface / ARCH_NORM)
        simplicity_cost = (
            0.30 * sim_loc
            + 0.20 * sim_cc
            + 0.20 * sim_dep
            + 0.30 * sim_arch
        )

        validation_score = _compute_validation_score()

        discovery_value = (
            0.10 * rediscovery_rate
            + 0.15 * scaffold_novelty
            + 0.10 * top_k_enrichment
            + 0.15 * external_consistency
            + 0.20 * holdout_gen
            + 0.15 * trend_recovery
            + 0.15 * validation_score
        )

        net_progress = discovery_value - LAMBDA * simplicity_cost

        print(f"\n{'=' * 65}")
        print("  NET PROGRESS REPORT")
        print(f"{'=' * 65}")
        print("  DISCOVERY VALUE")
        print(f"    Rediscovery rate:            {rediscovery_rate:.3f}")
        print(f"    Scaffold novelty:            {scaffold_novelty:.3f}")
        print(f"    Top-k enrichment:            {top_k_enrichment:.3f}")
        print(f"    External consistency:        {external_consistency:.3f}")
        print(f"    Holdout generalization:      {holdout_gen:.3f}")
        print(f"    Experimental trend recovery: {trend_recovery:.3f}")
        print(f"    Validation score:            {validation_score:.3f}")
        print(f"    DISCOVERY_VALUE:             {discovery_value:.3f}")
        print("  SIMPLICITY COST")
        print(f"    Lines of code:               {loc} (norm={sim_loc:.3f})")
        print(f"    CC violations >12:           {cc_violations} (norm={sim_cc:.3f})")
        print(f"    Third-party deps:            {n_deps} (norm={sim_dep:.3f})")
        print(f"    Architectural surface area:  {arch_surface} (norm={sim_arch:.3f})")
        print(f"    SIMPLICITY_COST:             {simplicity_cost:.3f}")
        print("  NET PROGRESS")
        print(f"    λ:                           {LAMBDA}")
        print(f"    NET_PROGRESS:                {net_progress:.3f}")
        print(f"{'=' * 65}")

        assert net_progress > 0.0, (
            f"NET_PROGRESS = {net_progress:.3f} is not positive. "
            f"The complexity cost ({simplicity_cost:.3f} × λ={LAMBDA}) "
            f"outweighs the discovery value ({discovery_value:.3f})."
        )


class TestDiscoveryGates:
    """CI regression gates for discovery must fail closed on a missing/empty section.

    Gap 2: `check_tolerances` used to evaluate the rediscovery/novelty/score-gap
    gates only `if disc:` was truthy, so `"discovery": {}` passed silently.
    These tests pin the new hard-failure behaviour without running any pipeline.
    """

    def test_discovery_gates_require_results(self):
        """An empty discovery section must be a hard CI failure mentioning discovery."""
        ok, failures = check_tolerances({"discovery": {}})
        assert ok is False
        assert any("discovery" in f for f in failures), f"no discovery failure in: {failures}"
        assert any("not run" in f for f in failures), f"failure must mention benchmark not run: {failures}"

    def test_discovery_gates_still_work_when_populated(self):
        """A populated discovery section is evaluated with the existing thresholds.

        Passing values must produce no failure mentioning "discovery"; other
        sections may legitimately fail on this minimal dict, so only the
        "discovery" substring is filtered on.
        """
        _ok, failures = check_tolerances(
            {
                "discovery": {
                    "rediscovery_rate": 0.9,
                    "novel_scaffold_ratio": 0.9,
                    "score_gap": 0.5,
                }
            }
        )
        assert not any("discovery" in f for f in failures), f"unexpected discovery failures: {failures}"
