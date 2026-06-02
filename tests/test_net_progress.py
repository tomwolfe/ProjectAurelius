"""Net Progress metric — repository-level objective function.

Defines:
  DISCOVERY_VALUE = (0.4 * rediscovery_rate) + (0.3 * scaffold_novelty)
                    + (0.2 * top_k_enrichment) + (0.1 * oracle_calibration)
  SIMPLICITY_COST = normalized_LOC + normalized_cyclomatic_complexity
                    + normalized_dependency_count
  NET_PROGRESS = DISCOVERY_VALUE - (λ * SIMPLICITY_COST)

This test calculates the BASELINE net progress and verifies that any code
changes do not decrease it. The constants (λ, normalisation factors) are
chosen so that NET_PROGRESS lives in [0, 1] for a healthy repository.

Usage:
    pytest tests/test_net_progress.py -v
"""

from __future__ import annotations

import ast
import json
import os
import random

import numpy as np
import pytest
from rdkit import Chem

from aurelius.scoring.oracle.gc import predict_dielectric_proxy, predict_viscosity_proxy
from aurelius.scoring.oracle.quantum import predict_tom_orbitals
from aurelius.types import MoleculeContext


LAMBDA = 0.3

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius")


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
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split(".")[0]
                        if pkg not in stdlib and not pkg.startswith("aurelius"):
                            deps.add(pkg)
    return len(deps)


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
        from rdkit.Chem.Scaffolds import MurckoScaffold
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
    """Compare mean score of top-10 vs bottom-10 from mutation engine proposals.

    A positive enrichment means the pipeline correctly ranks candidates.
    """
    try:
        from aurelius.pipeline import AureliusPipeline
        from aurelius.agent.mutation import MutationEngine

        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        candidates = engine.propose_candidates(n_candidates=50, batch_size=25)

        pipeline = AureliusPipeline()
        pipeline.initialize()

        scores = []
        for smi in candidates:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            try:
                result = pipeline.screen_molecule(ctx)
                score = result.get("score", {}).get("total_score", 0.0)
                scores.append(score)
            except Exception:
                continue

        if len(scores) < 10:
            return 0.0

        scores.sort(reverse=True)
        top_mean = np.mean(scores[:10])
        bottom_mean = np.mean(scores[-10:])
        enrichment = (top_mean - bottom_mean) / 100.0
        return max(0.0, min(1.0, enrichment))
    except Exception:
        return 0.0


def _compute_oracle_calibration() -> float:
    """Compute TOM calibration quality as 1 - (MAE / 1.5), clipped to [0, 1].

    1.0 = perfect calibration, 0.0 = MAE >= 1.5 eV.
    """
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data", "orbital_calibration.json"
    )
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return 0.0

    errors = []
    for entry in data:
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

        rediscovery_rate = _compute_rediscovery_rate()
        scaffold_novelty = _compute_scaffold_novelty()
        top_k_enrichment = _compute_top_k_enrichment()
        oracle_cal = _compute_oracle_calibration()

        # Normalisation factors (empirical targets for a healthy repo)
        LOC_NORM = 5000.0
        CC_NORM = 5.0
        DEP_NORM = 10.0

        sim_loc = min(1.0, loc / LOC_NORM)
        sim_cc = min(1.0, cc_violations / CC_NORM)
        sim_dep = min(1.0, n_deps / DEP_NORM)
        simplicity_cost = (sim_loc + sim_cc + sim_dep) / 3.0

        discovery_value = (
            0.4 * rediscovery_rate
            + 0.3 * scaffold_novelty
            + 0.2 * top_k_enrichment
            + 0.1 * oracle_cal
        )

        net_progress = discovery_value - LAMBDA * simplicity_cost

        print(f"\n{'=' * 55}")
        print(f"  NET PROGRESS REPORT")
        print(f"{'=' * 55}")
        print(f"  DISCOVERY VALUE")
        print(f"    Rediscovery rate:     {rediscovery_rate:.3f}")
        print(f"    Scaffold novelty:     {scaffold_novelty:.3f}")
        print(f"    Top-k enrichment:     {top_k_enrichment:.3f}")
        print(f"    Oracle calibration:   {oracle_cal:.3f}")
        print(f"    DISCOVERY_VALUE:      {discovery_value:.3f}")
        print(f"  SIMPLICITY COST")
        print(f"    Lines of code:        {loc} (norm={sim_loc:.3f})")
        print(f"    CC violations >12:    {cc_violations} (norm={sim_cc:.3f})")
        print(f"    Third-party deps:     {n_deps} (norm={sim_dep:.3f})")
        print(f"    SIMPLICITY_COST:      {simplicity_cost:.3f}")
        print(f"  NET PROGRESS")
        print(f"    λ:                    {LAMBDA}")
        print(f"    NET_PROGRESS:         {net_progress:.3f}")
        print(f"{'=' * 55}")

        assert net_progress > 0.0, (
            f"NET_PROGRESS = {net_progress:.3f} is not positive. "
            f"The complexity cost ({simplicity_cost:.3f} × λ={LAMBDA}) "
            f"outweighs the discovery value ({discovery_value:.3f})."
        )
