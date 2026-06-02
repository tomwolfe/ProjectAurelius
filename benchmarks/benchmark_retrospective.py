#!/usr/bin/env python3
"""Benchmark: Retrospective Rediscovery — can the EA recover known electrolytes?

Randomly withholds 20% of known_electrolytes.json from the seed pool and
commercial fingerprint database, then runs the mutation engine seeded with
the remaining 80% to see what fraction of withheld molecules appear in the
top-100 scored discoveries.

Target rediscovery rate: >=30% combined (exact + scaffold + Tanimoto >= 0.70).

Usage:
    python -m benchmarks.benchmark_retrospective
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
import time
import unittest.mock
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger("aurelius").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.agent.mutation import MutationEngine
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext

HOLDOUT_FRACTION = 0.20
TOP_N = 100
RANDOM_SEED = 42
MAX_CANDIDATES = 2000


def _load_known_electrolytes() -> list[str]:
    data_path = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "known_electrolytes.json"
    with open(data_path) as f:
        return json.load(f)


def _canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def _scaffold_from_smiles(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if s:
            return s
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def _safe_canonical(smiles: str) -> str:
    if "." in smiles:
        components = smiles.split(".")
        canon_parts = []
        for comp in components:
            mol = Chem.MolFromSmiles(comp.strip())
            canon_parts.append(Chem.MolToSmiles(mol) if mol else comp)
        return ".".join(canon_parts)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def _load_and_split_known() -> tuple[list[str], list[str]]:
    """Load known electrolytes, canononalize, and split into holdout/retained."""
    all_known = _load_known_electrolytes()
    all_canon = [_canonicalize(s) for s in all_known]
    random.seed(RANDOM_SEED)
    indices = list(range(len(all_canon)))
    random.shuffle(indices)
    n_holdout = max(1, int(len(all_canon) * HOLDOUT_FRACTION))
    holdout_idx = set(indices[:n_holdout])
    holdout = [all_canon[i] for i in holdout_idx]
    retained = [all_canon[i] for i in indices if i not in holdout_idx]
    print(f"         Total known electrolytes: {len(all_canon)}")
    print(f"         Withheld for rediscovery: {len(holdout)} ({HOLDOUT_FRACTION:.0%})")
    print(f"         Retained (seeds + fingerprint DB): {len(retained)}")
    return holdout, retained


def _make_patched_load_func(override_path: str):
    """Return a patched _load_known_electrolytes that reads from override_path."""
    def patched_load(self):
        import json as _json
        try:
            with open(override_path) as f:
                smiles_list = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return
        existing_smis = set()
        for ctx in self.seed_contexts:
            try:
                existing_smis.add(Chem.MolToSmiles(ctx.mol))
            except Exception:
                continue
        for smi in smiles_list:
            ctx = self._get_ctx(smi)
            if ctx is not None:
                canon = Chem.MolToSmiles(ctx.mol)
                if canon not in existing_smis:
                    self._commercial_fps.append(ctx.get_ecfp4())
                    self._known_smiles.add(canon)
    return patched_load


def run_trial(seed_smiles: list[str]) -> list[tuple[float, str]]:
    """Generate candidates + score seeds through the pipeline, return top-N."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    engine = MutationEngine(seed_smiles=seed_smiles)
    pipeline = AureliusPipeline()
    pipeline.initialize()
    n_candidates = min(MAX_CANDIDATES, max(500, len(seed_smiles) * 20))
    generated = engine.propose_candidates(n_candidates=n_candidates, batch_size=50)
    pool = set(generated) | set(seed_smiles)
    scored: list[tuple[float, str]] = []
    for smi in pool:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        try:
            result = pipeline.screen_molecule(ctx)
            score = result.get("score", {}).get("total_score", 0.0)
            if score >= 10.0:
                scored.append((score, smi))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[0])
    return scored[:TOP_N]


def _compute_tanimoto_matches(
    holdout_smiles: list[str],
    holdout_fps: list,
    valid_top_fps: list,
) -> tuple[set, list]:
    """Print per-holdout Tanimoto similarities and return matched set + fps list."""
    from rdkit.DataStructs import BulkTanimotoSimilarity
    matched = set()
    for i, hfp in enumerate(holdout_fps):
        if hfp is None or not valid_top_fps:
            print(f"         [{i}] {holdout_smiles[i]:40s} <skip>")
            continue
        sims = BulkTanimotoSimilarity(hfp, valid_top_fps)
        max_sim = max(sims)
        marker = " <<<" if max_sim >= 0.70 else ""
        print(f"         [{i}] {holdout_smiles[i]:40s} max Tanimoto to top-{TOP_N}: {max_sim:.3f}{marker}")
        if max_sim >= 0.70:
            matched.add(holdout_smiles[i])
    return matched, holdout_fps


def _compute_scaffolds_dict(smiles_list: list[str]) -> dict[str, str]:
    """Build {smiles: scaffold} mapping, filtering out failures."""
    result = {}
    for s in smiles_list:
        scaf = _scaffold_from_smiles(s)
        if scaf:
            result[s] = scaf
    return result


def _match_scaffolds(
    holdout_scaffolds: dict[str, str],
    top_scaffolds: dict[str, str],
) -> set[str]:
    """Return holdout SMILES whose scaffold appears in top_scaffolds."""
    matched = set()
    for smi_h, scaf_h in holdout_scaffolds.items():
        if scaf_h and any(scaf_t == scaf_h for scaf_t in top_scaffolds.values()):
            matched.add(smi_h)
    return matched


def _build_fingerprint_list(smiles_list: list[str]) -> list:
    """Build list of RDKFingerprints from SMILES, None for invalid."""
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        fps.append(Chem.RDKFingerprint(mol) if mol else None)
    return fps


def _analyze_rediscovery(
    holdout_smiles: list[str],
    top_results: list[tuple[float, str]],
) -> dict:
    """Compute rediscovery metrics (exact, scaffold, Tanimoto >= 0.70)."""
    holdout_canon = set(_canonicalize(s) for s in holdout_smiles)
    top_smiles = [_safe_canonical(s) for _, s in top_results]

    holdout_scaffolds = _compute_scaffolds_dict(holdout_smiles)
    top_scaffolds = _compute_scaffolds_dict([s for _, s in top_results])

    rediscovered_exact = holdout_canon & set(top_smiles)
    rediscovered_scaffold = _match_scaffolds(holdout_scaffolds, top_scaffolds)

    holdout_fps = _build_fingerprint_list(holdout_smiles)
    top_fps = _build_fingerprint_list([s for _, s in top_results])
    valid_top_fps = [fp for fp in top_fps if fp is not None]

    rediscovered_tanimoto, _ = _compute_tanimoto_matches(
        holdout_smiles, holdout_fps, valid_top_fps,
    )

    rediscovered_combined = rediscovered_exact | rediscovered_scaffold | rediscovered_tanimoto
    n = max(len(holdout_canon), 1)
    return {
        "holdout_canon": holdout_canon,
        "holdout_scaffolds": holdout_scaffolds,
        "top_scaffolds": top_scaffolds,
        "rediscovered_exact": rediscovered_exact,
        "rediscovered_scaffold": rediscovered_scaffold,
        "rediscovered_tanimoto": rediscovered_tanimoto,
        "rediscovered_combined": rediscovered_combined,
        "exact_rate": len(rediscovered_exact) / n * 100,
        "scaffold_rate": len(rediscovered_scaffold) / n * 100,
        "tanimoto_rate": len(rediscovered_tanimoto) / n * 100,
        "combined_rate": len(rediscovered_combined) / n * 100,
        "holdout_fps": holdout_fps,
        "valid_top_fps": valid_top_fps,
    }


def _match_label(smi: str, analysis: dict) -> str:
    """Return match label (EXACT / SCAFFOLD / TANIMOTO) for a discovery."""
    from rdkit.DataStructs import BulkTanimotoSimilarity
    s = _safe_canonical(smi)
    if s in analysis["holdout_canon"]:
        return " <<< EXACT"
    if smi in analysis["top_scaffolds"]:
        scaf = analysis["top_scaffolds"][smi]
        if any(analysis["holdout_scaffolds"].get(h) == scaf for h in analysis["holdout_scaffolds"]):
            return " <<< SCAFFOLD"
    ctx = MoleculeContext.from_smiles(smi)
    if ctx is None:
        return ""
    tfp = Chem.RDKFingerprint(ctx.mol)
    if tfp is None:
        return ""
    for hfp in analysis["holdout_fps"]:
        if hfp is not None and BulkTanimotoSimilarity(tfp, [hfp])[0] >= 0.70:
            return " <<< TANIMOTO"
    return ""


def _print_top10_markers(
    top_results: list[tuple[float, str]],
    analysis: dict,
) -> None:
    for i, (score, smi) in enumerate(top_results[:10], 1):
        label = _match_label(smi, analysis)
        print(f"           {i:2d}. {smi[:55]:<55s} score={score:.1f}{label}")


def main() -> None:
    print("=" * 65)
    print("  RETROSPECTIVE REDISCOVERY BENCHMARK")
    print("=" * 65)
    print()

    print("  [1/5] Loading known electrolytes and splitting...")
    holdout_smiles, retained_smiles = _load_and_split_known()
    print()

    print("  [2/5] Preparing trimmed fingerprint database...")
    tmp_dir = tempfile.mkdtemp(prefix="aurelius_retro_")
    trimmed_path = os.path.join(tmp_dir, "known_electrolytes.json")
    with open(trimmed_path, "w") as f:
        json.dump(retained_smiles, f)
    print()

    print("  [3/5] Running discovery trial")
    print(f"         Seeds: {len(retained_smiles)} retained molecules")
    t0 = time.time()
    with unittest.mock.patch.object(
        MutationEngine, "_load_known_electrolytes",
        _make_patched_load_func(trimmed_path),
    ):
        top_results = run_trial(seed_smiles=retained_smiles)
    trial_time = time.time() - t0
    print(f"         Done in {trial_time:.1f}s")
    print(f"         Top {len(top_results)} results generated")
    print()

    print("  [4/5] Analyzing rediscovery rate...")
    print()
    if not top_results:
        print("         No results — exiting.")
        return
    analysis = _analyze_rediscovery(holdout_smiles, top_results)

    print()
    print(f"         Top {TOP_N} EA discoveries:")
    _print_top10_markers(top_results, analysis)
    if len(top_results) > 10:
        print(f"           ... ({len(top_results) - 10} more)")
    print()

    print(f"         Withheld molecules: {len(set(_canonicalize(s) for s in holdout_smiles))}")
    print(f"         Exact SMILES rediscovered: {len(analysis['rediscovered_exact'])} ({analysis['exact_rate']:.1f}%)")
    print(f"         Scaffold rediscovered:     {len(analysis['rediscovered_scaffold'])} ({analysis['scaffold_rate']:.1f}%)")
    print(f"         Tanimoto>=0.70 rediscovered: {len(analysis['rediscovered_tanimoto'])} ({analysis['tanimoto_rate']:.1f}%)")
    print(f"         Combined rediscovery:       {len(analysis['rediscovered_combined'])} ({analysis['combined_rate']:.1f}%)")
    print()
    for key, label in [("rediscovered_exact", "Exact"),
                       ("rediscovered_scaffold", "Scaffold"),
                       ("rediscovered_tanimoto", "Tanimoto")]:
        val = analysis[key]
        if val:
            print(f"         {label} rediscovered: {sorted(val)}")
    print()

    print("  [5/5] Verifying assertion...")
    target_rate = 30.0
    combined_rate = analysis["combined_rate"]
    print(f"         Combined rediscovery rate: {combined_rate:.1f}% (target > {target_rate:.0f}%)")
    assert combined_rate >= target_rate, (
        f"FAILED: EA rediscovered only {combined_rate:.1f}% of withheld known "
        f"electrolytes via combined metric ({len(analysis['rediscovered_combined'])}/"
        f"{len(set(_canonicalize(s) for s in holdout_smiles))}). "
        f"Target > {target_rate:.0f}%."
    )
    print("         PASSED: EA rediscovered >=30% of withheld known electrolytes.")
    print()

    print("=" * 65)
    print("  RETROSPECTIVE REDISCOVERY: ALL ASSERTIONS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
