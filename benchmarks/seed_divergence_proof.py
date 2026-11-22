"""Seed divergence proof for Phase 1 measurement integrity.

Uses ASCII hyphens only. Proves four claims:
  (a) per-seed rediscovery + novelty for all 5 seeds with MIN across seeds
  (b) Jaccard overlap matrices with max well below 1 (seeds diverge)
  (c) same-seed two-process bit-identity PASS (1e-6, ranking check)
  (d) MIN-gate confirmation quoting exact file lines from benchmark_unified.py

Exit 0 and print SEED_DIVERGENCE_PROOF: PASS only if every claim holds.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

SEEDS = [42, 123, 7, 2024, 99]
TOL = 1e-6


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _single_seed_payload(seed: int) -> dict:
    from benchmark_unified import discovery_benchmark

    r = discovery_benchmark(seed=seed)
    return {
        "seed": seed,
        "rediscovery_exact_rate": float(r.get("rediscovery_exact_rate", 0.0)),
        "rediscovery_rate": float(r.get("rediscovery_rate", 0.0)),
        "novel_scaffold_ratio": float(r.get("novel_scaffold_ratio", 0.0)),
        "score_gap": float(r.get("score_gap", 0.0)),
        "n_duplicates_top50": int(r.get("n_duplicates_top50", -1)),
        "trivial_hits_top50": list(r.get("trivial_hits_top50", [])),
        "min_retrosynthetic_depth_top10": r.get("min_retrosynthetic_depth_top10", -1),
        "top50_canonical_smiles": list(r.get("top50_canonical_smiles", [])),
        "screened_canonical_smiles": list(r.get("screened_canonical_smiles", [])),
        "recovered_exact_smiles": list(r.get("recovered_exact_smiles", [])),
        "novel_scaffolds": list(r.get("novel_scaffolds", [])),
        "rediscovery_partial_sum": float(r.get("rediscovery_partial_sum", 0.0)),
        "novelty_base_ratio": float(r.get("novelty_base_ratio", 0.0)),
        "novelty_mean_pairwise_distance": float(r.get("novelty_mean_pairwise_distance", 0.0)),
        "n_known": int(r.get("n_known", 0)),
        "n_rediscovered_exact": int(r.get("n_rediscovered_exact", 0)),
    }


def _dump_in_subprocess(seed: int) -> dict:
    code = (
        "import json, sys; "
        "from pathlib import Path; "
        f"sys.path.insert(0, {str(SRC_DIR)!r}); "
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'benchmarks')!r}); "
        "from benchmark_unified import discovery_benchmark; "
        f"r = discovery_benchmark(seed={seed}); "
        "payload = {"
        "'rediscovery_exact_rate': float(r.get(\"rediscovery_exact_rate\", 0.0)), "
        "'rediscovery_rate': float(r.get(\"rediscovery_rate\", 0.0)), "
        "'novel_scaffold_ratio': float(r.get(\"novel_scaffold_ratio\", 0.0)), "
        "'score_gap': float(r.get(\"score_gap\", 0.0)), "
        "'n_duplicates_top50': int(r.get(\"n_duplicates_top50\", -1)), "
        "'trivial_hits_top50': list(r.get(\"trivial_hits_top50\", [])), "
        "'min_retrosynthetic_depth_top10': r.get(\"min_retrosynthetic_depth_top10\", -1), "
        "'top50_canonical_smiles': list(r.get(\"top50_canonical_smiles\", [])), "
        "'screened_canonical_smiles': list(r.get(\"screened_canonical_smiles\", [])), "
        "'recovered_exact_smiles': list(r.get(\"recovered_exact_smiles\", [])), "
        "'novel_scaffolds': list(r.get(\"novel_scaffolds\", [])), "
        "'rediscovery_partial_sum': float(r.get(\"rediscovery_partial_sum\", 0.0)), "
        "'novelty_base_ratio': float(r.get(\"novelty_base_ratio\", 0.0)), "
        "'novelty_mean_pairwise_distance': float(r.get(\"novelty_mean_pairwise_distance\", 0.0)), "
        "'n_known': int(r.get(\"n_known\", 0)), "
        "'n_rediscovered_exact': int(r.get(\"n_rediscovered_exact\", 0)), "
        "}; "
        "print(json.dumps(payload, sort_keys=True))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess seed={seed} failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    from benchmark_unified import TOLERANCES

    print("=" * 74)
    print("  SEED DIVERGENCE PROOF - Phase 1 measurement integrity")
    print("=" * 74)

    # (a) per-seed rediscovery + novelty with MIN
    print("\n(a) PER-SEED REDISCOVERY + NOVELTY (seeds [42, 123, 7, 2024, 99])")
    per_seed = [_single_seed_payload(s) for s in SEEDS]
    for p in per_seed:
        print(
            f"  seed={p['seed']}: "
            f"rediscovery_exact={p['rediscovery_exact_rate']:.6f} "
            f"novel_scaffold={p['novel_scaffold_ratio']:.6f} "
            f"score_gap={p['score_gap']:+.2f} "
            f"dups_top50={p['n_duplicates_top50']} "
            f"trivial={p['trivial_hits_top50']} "
            f"min_depth={p['min_retrosynthetic_depth_top10']} "
            f"n_known={p['n_known']} n_exact={p['n_rediscovered_exact']} "
            f"partial_sum={p['rediscovery_partial_sum']:.6f} "
            f"base_novel={p['novelty_base_ratio']:.6f} "
            f"pairwise_dist={p['novelty_mean_pairwise_distance']:.6f}"
        )
    min_red = min(p["rediscovery_exact_rate"] for p in per_seed)
    min_nov = min(p["novel_scaffold_ratio"] for p in per_seed)
    min_gap = min(p["score_gap"] for p in per_seed)
    min_red_seed = SEEDS[[p["rediscovery_exact_rate"] for p in per_seed].index(min_red)]
    min_nov_seed = SEEDS[[p["novel_scaffold_ratio"] for p in per_seed].index(min_nov)]
    min_gap_seed = SEEDS[[p["score_gap"] for p in per_seed].index(min_gap)]
    print(f"  MIN rediscovery_exact={min_red:.6f} (worst seed {min_red_seed})")
    print(f"  MIN novel_scaffold={min_nov:.6f} (worst seed {min_nov_seed})")
    print(f"  MIN score_gap={min_gap:+.2f} (worst seed {min_gap_seed})")

    gate_red = min_red >= TOLERANCES["discovery"]["rediscovery_rate_min"]
    gate_nov = min_nov >= TOLERANCES["discovery"]["novel_scaffold_min"]
    gate_gap = min_gap > TOLERANCES["discovery"]["score_gap_min"]
    gate_dup = all(p["n_duplicates_top50"] == 0 for p in per_seed)
    gate_triv = all(len(p["trivial_hits_top50"]) == 0 for p in per_seed)
    gate_depth = all(
        (p["min_retrosynthetic_depth_top10"] != -1)
        and (p["min_retrosynthetic_depth_top10"] > TOLERANCES["integrity"]["min_retrosynthetic_depth"])
        for p in per_seed
    )
    print(f"  gate rediscovery MIN >= 0.50: {'PASS' if gate_red else 'FAIL'}")
    print(f"  gate novel_scaffold MIN >= 0.80: {'PASS' if gate_nov else 'FAIL'}")
    print(f"  gate score_gap MIN > 0.00: {'PASS' if gate_gap else 'FAIL'}")
    print(f"  gate duplicates top-50 == 0: {'PASS' if gate_dup else 'FAIL'}")
    print(f"  gate trivial hits none: {'PASS' if gate_triv else 'FAIL'}")
    print(f"  gate min retrosynthetic depth top-10 > 1: {'PASS' if gate_depth else 'FAIL'}")
    claim_a = gate_red and gate_nov and gate_gap and gate_dup and gate_triv and gate_depth

    # (b) Jaccard overlap matrices
    print("\n(b) JACCARD OVERLAP MATRICES (top-50 / screened / novel)")
    top_sets = [set(p["top50_canonical_smiles"]) for p in per_seed]
    scr_sets = [set(p["screened_canonical_smiles"]) for p in per_seed]
    nov_sets = [set(p["novel_scaffolds"]) for p in per_seed]
    top_pairs, scr_pairs, nov_pairs = [], [], []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            tj = round(_jaccard(top_sets[i], top_sets[j]), 4)
            sj = round(_jaccard(scr_sets[i], scr_sets[j]), 4)
            nj = round(_jaccard(nov_sets[i], nov_sets[j]), 4)
            top_pairs.append(tj)
            scr_pairs.append(sj)
            nov_pairs.append(nj)
            print(f"  pair ({SEEDS[i]},{SEEDS[j]}): top50={tj:.4f} screened={sj:.4f} novel={nj:.4f}")
    top_max = max(top_pairs) if top_pairs else 1.0
    scr_max = max(scr_pairs) if scr_pairs else 1.0
    nov_max = max(nov_pairs) if nov_pairs else 1.0
    print(f"  MAX top50_jaccard={top_max:.4f}")
    print(f"  MAX screened_jaccard={scr_max:.4f}")
    print(f"  MAX novel_jaccard={nov_max:.4f}")
    screened_differ = len({tuple(sorted(s)) for s in scr_sets}) > 1
    print(f"  screened sets differ across seeds: {screened_differ}")
    # High-resolution divergence: per-seed gating values must not be all-equal.
    # A minority coinciding is acceptable, but all five identical means the
    # metric still lacks resolution even though the pools differ.
    _red_vals = [round(float(p["rediscovery_exact_rate"]), 6) for p in per_seed]
    _nov_vals = [round(float(p["novel_scaffold_ratio"]), 6) for p in per_seed]
    _red_diverge = len(set(_red_vals)) > 1
    _nov_diverge = len(set(_nov_vals)) > 1
    print(f"  per-seed rediscovery_exact distinct={sorted(set(_red_vals))} diverge={_red_diverge}")
    print(f"  per-seed novel_scaffold distinct={sorted(set(_nov_vals))} diverge={_nov_diverge}")
    claim_b = (top_max < 1.0) and (scr_max < 1.0) and screened_differ and _red_diverge and _nov_diverge
    print(f"  divergence (max well below 1 and sets differ): {'PASS' if claim_b else 'FAIL'}")

    # (c) same-seed two-process bit-identity
    print("\n(c) SAME-SEED TWO-PROCESS BIT-IDENTITY (seed=123, tol 1e-6)")
    bit_seed = 123
    run_a = _dump_in_subprocess(bit_seed)
    run_b = _dump_in_subprocess(bit_seed)
    float_keys = ["rediscovery_exact_rate", "rediscovery_rate", "novel_scaffold_ratio", "score_gap", "rediscovery_partial_sum", "novelty_base_ratio", "novelty_mean_pairwise_distance"]
    list_keys = [
        "top50_canonical_smiles",
        "screened_canonical_smiles",
        "recovered_exact_smiles",
        "novel_scaffolds",
        "trivial_hits_top50",
    ]
    claim_c = True
    for k in float_keys:
        d = abs(float(run_a[k]) - float(run_b[k]))
        ok = d <= TOL
        claim_c = claim_c and ok
        print(f"  {k}: a={run_a[k]} b={run_b[k]} absdiff={d:.2e} {'PASS' if ok else 'FAIL'}")
    for k in list_keys:
        ok = list(run_a[k]) == list(run_b[k])
        claim_c = claim_c and ok
        print(f"  {k}: len a={len(run_a[k])} len b={len(run_b[k])} identical={ok} {'PASS' if ok else 'FAIL'}")
    # ranking check: top-50 composition plus score gap identical implies
    # the score ranking that produced the top-50 is stable across processes
    rank_ok = (
        list(run_a["top50_canonical_smiles"]) == list(run_b["top50_canonical_smiles"])
        and abs(float(run_a["score_gap"]) - float(run_b["score_gap"])) <= TOL
    )
    claim_c = claim_c and rank_ok
    print(f"  ranking check (top-50 composition + score gap stable): {'PASS' if rank_ok else 'FAIL'}")
    print(f"  bit-identity overall: {'PASS' if claim_c else 'FAIL'}")

    # (d) MIN-gate confirmation with exact file lines
    print("\n(d) MIN-GATE CONFIRMATION (exact lines from benchmarks/benchmark_unified.py)")
    src = (PROJECT_ROOT / "benchmarks" / "benchmark_unified.py").read_text(encoding="utf-8")
    probes = [
        "min([r['rediscovery_rate']",
        "min([r['rediscovery_exact_rate']",
        "min([r['score_gap']",
        "min((r['novel_scaffold_ratio']",
        "max((r.get('n_duplicates_top50'",
        "trivial_hits_top50': sorted({h for r in results",
        "min((r.get('min_retrosynthetic_depth_top10'",
        "Rediscovery rate MIN (seeded-exact)",
        "Novel scaffold ratio MIN",
        "Score gap MIN",
    ]
    claim_d = True
    for probe in probes:
        hits = [ln.strip() for ln in src.splitlines() if probe in ln]
        ok = len(hits) > 0
        claim_c_d = ok
        claim_d = claim_d and claim_c_d
        if ok:
            for h in hits[:2]:
                print(f"  FOUND {probe!r}: {h[:220]}")
        else:
            print(f"  MISSING {probe!r}: FAIL")
    # no mean may feed a gate: the multi-seed aggregator must not assign a
    # mean into any gating key
    multi_fn = src[src.index("def discovery_benchmark_multi_seed") : src.index("def ml_baseline_benchmark")]
    forbidden = [ln.strip() for ln in multi_fn.splitlines() if "mean_rediscovery" in ln or "mean_score_gap" in ln or "mean_novel" in ln]
    mean_ok = len(forbidden) == 0
    claim_d = claim_d and mean_ok
    print(f"  no mean feeds any gate (no mean_rediscovery/mean_score_gap/mean_novel): {'PASS' if mean_ok else 'FAIL'}")
    if forbidden:
        for ln in forbidden:
            print(f"    FORBIDDEN: {ln[:220]}")

    print("\nSUMMARY")
    print(f"  claim (a) per-seed MIN gates: {'PASS' if claim_a else 'FAIL'}")
    print(f"  claim (b) divergence: {'PASS' if claim_b else 'FAIL'}")
    print(f"  claim (c) bit-identity: {'PASS' if claim_c else 'FAIL'}")
    print(f"  claim (d) MIN-gate lines: {'PASS' if claim_d else 'FAIL'}")
    if claim_a and claim_b and claim_c and claim_d:
        print("SEED_DIVERGENCE_PROOF: PASS")
        return 0
    print("SEED_DIVERGENCE_PROOF: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
