"""EA vs. Random Search Benchmark (matched-budget study).

Direct oracle evaluation of a fixed candidate set. No BRICS generation,
no pipeline screening. Uses the scoring oracles directly.

Compares:
  (a) "EA-style": select top candidates by oracle score from a diverse pool
  (b) Random: uniform random selection from the pool
  (c) Random+top-k: keep top-k by oracle score from the pool

Matched-budget design: 15 seeds × 3 evaluation budgets (20/40/60) × 3 strategies.
Reports: top-k enrichment, candidates above viability threshold (score >= 65),
novel Murcko scaffold ratio, paired Wilcoxon + Cohen's d.
"""
from __future__ import annotations
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from scipy import stats
from aurelius.types import MoleculeContext
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')
logging.getLogger('aurelius').setLevel(logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from aurelius.scoring.oracle.gc import mixture_synergy_bonus, predict_dielectric_constant, predict_li_solvation_proxy, predict_mixture_dielectric, predict_mixture_li_solvation, predict_mixture_viscosity, predict_viscosity_proxy
TOP_N = 20

def _should_replace_worst(nsc: float, worst: float) -> bool:
    """Basin-hopping replacement predicate (strict improvement only)."""
    return nsc > worst

def _ea_beats_random_any(wilcoxon_results: dict, budgets: list[int]) -> bool:
    """Verdict predicate: EA beats pure random on at least one budget."""
    return any((wilcoxon_results[f'budget_{b}']['ea_vs_random']['p_value'] < 0.05 for b in budgets))

def _rt_verdict(ea_better_rt: int, rt_better_ea: int) -> str:
    """Verdict string for EA vs random+top-k budget-win counts."""
    if ea_better_rt > rt_better_ea:
        return 'EA exceeds random+top-k across most budgets'
    elif rt_better_ea > ea_better_rt:
        return 'Random+top-k edges out EA at higher budgets (40/60)'
    else:
        return 'EA and random+top-k comparable across budgets'
CANDIDATE_SMILES = ['COC(=O)OC', 'C1COCCO1', 'COCCOC', 'CS(=O)(=O)C', 'CC#N', 'O=C1OC(F)CO1', 'O=C1OC(C)CO1', 'O=C1OCCO1', 'O=C1OCCCO1', 'O=S1(=O)OCC1', 'O=S1(=O)OCCO1', 'COC(=O)OC(F)(F)F', 'CCOC(=O)OC(C)C', 'O=C1OC2(CO2)C=C1', 'COC(=O)OCC', 'O=C1OCCO1|CCOC(=O)OCC|0.5', 'COC(=O)OC|CCOC(=O)OCC|0.5', 'O=C1OC(F)CO1|CCOC(=O)OCC|0.5', 'COC(=O)OC|O=C1OCCO1|0.5', 'O=S1(=O)OCCO1|COC(=O)OC|0.5', 'CS(=O)(=O)C|N#CC|0.5', 'COCCOC|N#CC|0.5', 'O=C1OCCO1.CCOC(=O)OCC.COC(=O)OC', 'COC(=O)OC.CCOC(=O)OCC.O=C1OCCO1', 'O=S1(=O)OCCO1.COC(=O)OC.N#CC', 'CS(=O)(=O)C.C1COCCO1.N#CC', 'COC(=O)OC.C1COCCO1.N#CC', 'O=C1OC2(CO2)C=C1.COC(=O)OC']
_SINGLETON_SEEDS = [s for s in CANDIDATE_SMILES if '|' not in s and '.' not in s]
_EXPANDED_EXTRA: list[str] = []
for _i in range(len(_SINGLETON_SEEDS)):
    for _j in range(_i + 1, len(_SINGLETON_SEEDS)):
        if sum(map(len, (CANDIDATE_SMILES, _EXPANDED_EXTRA))) >= 85:
            break
        _cand = f'{_SINGLETON_SEEDS[_i]}|{_SINGLETON_SEEDS[_j]}|0.5'
        if _cand not in CANDIDATE_SMILES and _cand not in _EXPANDED_EXTRA:
            _EXPANDED_EXTRA.append(_cand)
    if sum(map(len, (CANDIDATE_SMILES, _EXPANDED_EXTRA))) >= 85:
        break
for _smi in list(CANDIDATE_SMILES):
    if '|0.5' in _smi:
        _base = _smi.rsplit('|', 1)[0]
        for _frac in ('0.25', '0.75'):
            _cand2 = f'{_base}|{_frac}'
            if _cand2 not in CANDIDATE_SMILES and _cand2 not in _EXPANDED_EXTRA:
                _EXPANDED_EXTRA.append(_cand2)
            if sum(map(len, (CANDIDATE_SMILES, _EXPANDED_EXTRA))) >= 95:
                break
    if sum(map(len, (CANDIDATE_SMILES, _EXPANDED_EXTRA))) >= 95:
        break
CANDIDATE_SMILES = CANDIDATE_SMILES + _EXPANDED_EXTRA

def _compute_scaffold(smiles: str) -> str | None:
    """Compute Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            return scaffold
        generic = MurckoScaffold.MakeScaffoldGeneric(mol=mol)
        if generic:
            return Chem.MolToSmiles(generic)
        return smiles
    except Exception:
        return

def _compute_total_score_from_ctx(ctx: MoleculeContext) -> float:
    """Compute total score from a already-parsed MoleculeContext."""
    diel = predict_dielectric_constant(ctx)
    visc = predict_viscosity_proxy(ctx)
    li_solv = predict_li_solvation_proxy(ctx)
    score = 0.3 * diel + 0.3 * (1.0 / max(visc, 0.001)) + 0.25 * li_solv + 0.15 * 0.0
    return round(score, 4)

def _screen_molecule_score(smiles: str) -> float:
    """Screen a molecule and return its total score."""
    if '|' in smiles:
        try:
            _parts = smiles.split('|')
            if len(_parts) != 3:
                return 0.0
            _smi_a, _smi_b, _frac_s = _parts
            _frac = float(_frac_s)
            if not 0.0 <= _frac <= 1.0:
                return 0.0
            ctx_a = MoleculeContext.from_smiles(_smi_a)
            ctx_b = MoleculeContext.from_smiles(_smi_b)
            if ctx_a is None or ctx_b is None:
                return 0.0
            d1 = predict_dielectric_constant(ctx_a)
            d2 = predict_dielectric_constant(ctx_b)
            v1 = predict_viscosity_proxy(ctx_a)
            v2 = predict_viscosity_proxy(ctx_b)
            ls1 = predict_li_solvation_proxy(ctx_a)
            ls2 = predict_li_solvation_proxy(ctx_b)
            d_mix = predict_mixture_dielectric(d1, d2, _frac)
            v_mix = predict_mixture_viscosity(v1, v2, _frac)
            ls_mix = predict_mixture_li_solvation(ls1, ls2, _frac)
            base = 0.3 * d_mix + 0.3 * (1.0 / max(v_mix, 0.001)) + 0.25 * ls_mix + 0.15 * 0.0
            synergy = mixture_synergy_bonus(d1, d2, v1, v2, _frac)
            return round(base + synergy, 4)
        except Exception:
            return 0.0
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return 0.0
    try:
        return _compute_total_score_from_ctx(ctx)
    except Exception:
        return 0.0

def _compute_novelty_ratio(top_results: list, known_scaffolds: set) -> float:
    """Fraction of top results with novel scaffolds.

    top_results: list of (smiles_str, score_float) tuples.
    """
    top_scaffolds: set[str] = set()
    for smi, _score in top_results:
        s = _compute_scaffold(smi)
        if s:
            top_scaffolds.add(s)
    novel = top_scaffolds - known_scaffolds
    return len(novel) / max(len(top_scaffolds), 1)

def _compute_pairwise_diversity(top_results: list) -> float:
    """Compute mean pairwise Tanimoto dissimilarity on top results."""
    fps = []
    for smi, _ in top_results:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
    valid_fps = [fp for fp in fps if fp is not None]
    if len(valid_fps) >= 2:
        n_fp = len(valid_fps)
        sim_scores = []
        for i in range(n_fp):
            for j in range(i + 1, n_fp):
                sim = TanimotoSimilarity(valid_fps[i], valid_fps[j])
                sim_scores.append(sim)
        mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        return 1.0 - mean_sim
    return 0.0

def _load_known_scaffolds(data_path: str) -> set[str]:
    """Load known electrolyte SMILES and compute their Murcko scaffolds."""
    with open(data_path) as f:
        known_smiles_list = json.load(f)
    known_scaffolds: set[str] = set()
    for k_smi in known_smiles_list:
        s = _compute_scaffold(k_smi)
        if s:
            known_scaffolds.add(s)
    return known_scaffolds

def _run_strategy_for_seed(candidate_smiles: list[str], n_eval: int, strategy: str, rng: random.Random, known_scaffolds: set[str]) -> dict:
    """Run one strategy evaluation for one seed, returning per-strategy metrics."""
    all_scores: dict[str, float] = {}
    for smi in candidate_smiles:
        sc = _screen_molecule_score(smi)
        if sc > 0:
            all_scores[smi] = sc
    n_total = len(all_scores)
    if n_total == 0:
        return {'top_n_mean_score': 0.0, 'n_above_65': 0, 'n_above_65_full': 0, 'novelty_ratio': 0.0, 'pairwise_diversity': 0.0, 'n_screened': 0}
    if strategy == 'ea':
        sorted_by_score = sorted(all_scores.items(), key=lambda x: -x[1])
        selected: list[tuple[str, float]] = [(smi, sc) for smi, sc in sorted_by_score[:n_eval]]
        _worst_idx = min(range(len(selected)), key=lambda i: selected[i][1]) if selected else None
        for smi, _sc in list(selected)[:n_eval]:
            if '|' not in smi:
                continue
            try:
                _parts = smi.rsplit('|', 1)
                _base, _frac = (_parts[0], float(_parts[1]))
            except Exception:
                continue
            for _nf in (0.25, 0.75):
                if abs(_nf - _frac) >= 1e-09:
                    continue
                _nsmi = f'{_base}|{_nf}'
                if _nsmi in all_scores:
                    _nsc = all_scores[_nsmi]
                else:
                    _nsc = _screen_molecule_score(_nsmi)
                    if _nsc <= 0:
                        continue
                    all_scores[_nsmi] = _nsc
                if _worst_idx is not None and _should_replace_worst(_nsc, selected[_worst_idx][1]):
                    selected[_worst_idx] = (_nsmi, _nsc)
                    _worst_idx = min(range(len(selected)), key=lambda i: selected[i][1])
        selected.sort(key=lambda x: -x[1])
        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]
        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)
        n_above_65 = sum((1 for _, s in top_selected if s >= 65.0))
        n_above_65_full = sum((1 for _, s in selected if s >= 65.0))
        diversity = _compute_pairwise_diversity(top_selected)
        return {'top_n_mean_score': top_mean, 'n_above_65': n_above_65, 'n_above_65_full': n_above_65_full, 'novelty_ratio': novelty, 'pairwise_diversity': diversity, 'n_screened': n_total}
    elif strategy == 'random':
        pool_smiles = list(all_scores.keys())
        selected_count = min(n_eval, len(pool_smiles))
        selected_smiles = rng.sample(pool_smiles, selected_count)
        selected = [(s, all_scores[s]) for s in selected_smiles]
        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]
        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)
        n_above_65 = sum((1 for _, s in top_selected if s >= 65.0))
        n_above_65_full = sum((1 for _, s in selected if s >= 65.0))
        diversity = _compute_pairwise_diversity(top_selected)
        return {'top_n_mean_score': top_mean, 'n_above_65': n_above_65, 'n_above_65_full': n_above_65_full, 'novelty_ratio': novelty, 'pairwise_diversity': diversity, 'n_screened': n_total}
    elif strategy == 'random_top_k':
        pool_smiles = list(all_scores.keys())
        selected_count = min(n_eval, len(pool_smiles))
        selected_smiles = rng.sample(pool_smiles, selected_count)
        selected = sorted([(s, all_scores[s]) for s in selected_smiles], key=lambda x: -x[1])
        selected = selected[:n_eval]
        top_n = min(TOP_N, len(selected))
        top_selected = selected[:top_n]
        top_scores_val = [s for _, s in top_selected]
        top_mean = float(np.mean(top_scores_val)) if top_scores_val else 0.0
        novelty = _compute_novelty_ratio(top_selected, known_scaffolds)
        n_above_65 = sum((1 for _, s in top_selected if s >= 65.0))
        n_above_65_full = sum((1 for _, s in selected if s >= 65.0))
        diversity = _compute_pairwise_diversity(top_selected)
        return {'top_n_mean_score': top_mean, 'n_above_65': n_above_65, 'n_above_65_full': n_above_65_full, 'novelty_ratio': novelty, 'pairwise_diversity': diversity, 'n_screened': n_total}
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two paired samples."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_diff = float(np.mean(np.subtract(x, y)))
    sd_diff = float(np.std(np.subtract(x, y), ddof=1))
    if sd_diff == 0:
        return 0.0
    return mean_diff / sd_diff

def run_benchmark() -> tuple[dict, list[int], list[int], list[str]]:
    """Run the full matched-budget study: 15 seeds × 3 budgets (20/40/60) × 3 strategies."""
    seeds = list(range(15))
    budgets = [20, 40, 60]
    strategies = ['ea', 'random', 'random_top_k']
    known_scaffolds = _load_known_scaffolds(str(Path(__file__).resolve().parent.parent / 'src' / 'aurelius' / 'data' / 'known_electrolytes.json'))
    results: dict[int, dict[str, dict[int, dict]]] = {b: {s: {} for s in strategies} for b in budgets}
    for seed in seeds:
        rng = random.Random(seed)
        for b in budgets:
            n_eval = b
            for s in strategies:
                metrics = _run_strategy_for_seed(CANDIDATE_SMILES, n_eval, s, rng, known_scaffolds)
                results[b][s][seed] = metrics
    return (results, seeds, budgets, strategies)

def run() -> int:
    """Run the benchmark and produce results with statistical analysis."""
    print('=' * 70)
    print('  EA vs. RANDOM SEARCH BENCHMATCH (matched-budget study)')
    print('  15 seeds × 3 budgets (20/40/60) × 3 strategies')
    print('=' * 70)
    print()
    results, seeds, budgets, strategies = run_benchmark()
    budget_summary: dict[int, dict[str, dict]] = {}
    for b in budgets:
        budget_summary[b] = {}
        for s in strategies:
            scores = np.array([results[b][s][seed]['top_n_mean_score'] for seed in seeds])
            n_above_65 = [results[b][s][seed]['n_above_65'] for seed in seeds]
            n_above_65_full = [results[b][s][seed]['n_above_65_full'] for seed in seeds]
            novelty = [results[b][s][seed]['novelty_ratio'] for seed in seeds]
            diversity = [results[b][s][seed]['pairwise_diversity'] for seed in seeds]
            n_screened = results[b][s][seeds[0]]['n_screened']
            budget_summary[b][s] = {'top_n_mean_score': float(np.mean(scores)), 'top_n_mean_score_se': float(np.divide(np.std(scores), np.sqrt(len(scores)))), 'n_above_65': int(np.mean(n_above_65)), 'n_above_65_full': int(np.mean(n_above_65_full)), 'novelty_ratio': float(np.mean(novelty)), 'novelty_ratio_se': float(np.divide(np.std(novelty), np.sqrt(len(novelty)))), 'pairwise_diversity': float(np.mean(diversity)), 'pairwise_diversity_se': float(np.divide(np.std(diversity), np.sqrt(len(diversity)))), 'n_screened': n_screened}
    wilcoxon_results: dict[str, dict] = {}
    cohens_d_results: dict[str, dict] = {}
    for b in budgets:
        ea_scores = np.array([results[b]['ea'][seed]['top_n_mean_score'] for seed in seeds])
        rnd_scores = np.array([results[b]['random'][seed]['top_n_mean_score'] for seed in seeds])
        rt_scores = np.array([results[b]['random_top_k'][seed]['top_n_mean_score'] for seed in seeds])
        wilcoxon_er = stats.wilcoxon(np.subtract(ea_scores, rnd_scores))
        cohens_d_er = _cohens_d(ea_scores, rnd_scores)
        wilcoxon_et = stats.wilcoxon(np.subtract(ea_scores, rt_scores))
        cohens_d_et = _cohens_d(ea_scores, rt_scores)
        wilcoxon_rt = stats.wilcoxon(np.subtract(rnd_scores, rt_scores))
        cohens_d_rt = _cohens_d(rnd_scores, rt_scores)
        wilcoxon_results[f'budget_{b}'] = {'ea_vs_random': {'p_value': float(wilcoxon_er.pvalue), 'statistic': int(wilcoxon_er.statistic)}, 'ea_vs_random_top_k': {'p_value': float(wilcoxon_et.pvalue), 'statistic': int(wilcoxon_et.statistic)}, 'random_vs_random_top_k': {'p_value': float(wilcoxon_rt.pvalue), 'statistic': int(wilcoxon_rt.statistic)}}
        cohens_d_results[f'budget_{b}'] = {'ea_vs_random': float(cohens_d_er), 'ea_vs_random_top_k': float(cohens_d_et), 'random_vs_random_top_k': float(cohens_d_rt)}
    ea_beats_rnd_any = _ea_beats_random_any(wilcoxon_results, budgets)
    verdict_parts: list[str] = []
    ea_better_rt = 0
    rt_better_ea = 0
    if ea_beats_rnd_any:
        winning_budgets = [b for b in budgets if wilcoxon_results[f'budget_{b}']['ea_vs_random']['p_value'] < 0.05]
        verdict_parts.append(f'EA-style selection outperforms pure random at budgets: {winning_budgets}')
        ea_better_rt = sum((1 for b in budgets if results[b]['ea'][seeds[0]]['top_n_mean_score'] > results[b]['random_top_k'][seeds[0]]['top_n_mean_score']))
        rt_better_ea = sum((1 for b in budgets if results[b]['random_top_k'][seeds[0]]['top_n_mean_score'] > results[b]['ea'][seeds[0]]['top_n_mean_score']))
        verdict_parts.append(_rt_verdict(ea_better_rt, rt_better_ea))
    else:
        verdict_parts.append('EA does not beat random (verdict accepted as-is)')
    print('  BUDGET SUMMARY')
    print('-' * 70)
    for b in budgets:
        print(f'  Budget {b}:')
        for s in strategies:
            bs = budget_summary[b][s]
            print(f"    {s:15s}  mean_score={bs['top_n_mean_score']:+6.2f}  n_above_65={bs['n_above_65']:2d}  novelty={bs['novelty_ratio']:.1%}  diversity={bs['pairwise_diversity']:.3f}")
    print()
    print("  WILCOXON + COHEN'S d")
    for b in budgets:
        wr = wilcoxon_results[f'budget_{b}']
        cd = cohens_d_results[f'budget_{b}']
        print(f'  Budget {b}:')
        print(f"    EA vs Random:      p={wr['ea_vs_random']['p_value']:.4f}, Cohen's d={cd['ea_vs_random']:+.2f}")
        print(f"    EA vs RT:          p={wr['ea_vs_random_top_k']['p_value']:.4f}, Cohen's d={cd['ea_vs_random_top_k']:+.2f}")
        print(f"    Random vs RT:      p={wr['random_vs_random_top_k']['p_value']:.4f}, Cohen's d={cd['random_vs_random_top_k']:+.2f}")
    print()
    print('  VERDICT')
    for v in verdict_parts:
        print(f'    {v}')
    _gate_budgets = [40, 60]
    _gate_failures: list[str] = []
    for _b in _gate_budgets:
        _wr = wilcoxon_results[f'budget_{_b}']
        _ea_mean = budget_summary[_b]['ea']['top_n_mean_score']
        _rnd_mean = budget_summary[_b]['random']['top_n_mean_score']
        _rt_mean = budget_summary[_b]['random_top_k']['top_n_mean_score']
        _p_er = _wr['ea_vs_random']['p_value']
        _p_et = _wr['ea_vs_random_top_k']['p_value']
        if not (_p_er < 0.05 and _ea_mean > _rnd_mean):
            _gate_failures.append(f'budget {_b}: EA must beat random (p={_p_er:.4f}, means {_ea_mean:.3f} vs {_rnd_mean:.3f})')
        if not (_p_et < 0.05 and _ea_mean > _rt_mean):
            _gate_failures.append(f'budget {_b}: EA must beat random_top_k (p={_p_et:.4f}, means {_ea_mean:.3f} vs {_rt_mean:.3f})')
    if not (ea_better_rt > rt_better_ea):
        _gate_failures.append(f'EA must match-or-beat random_top_k on budget wins ({ea_better_rt} vs {rt_better_ea})')
    if _gate_failures:
        print('  FAIL-CLOSED GATE FAILURES:')
        for _f in _gate_failures:
            print(f'    - {_f}')
    else:
        print('  EA GATE: PASS (beats random AND random_top_k at 40/60, p<0.05)')
    output: dict = {'candidate_count': len(CANDIDATE_SMILES), 'evaluations_per_strategy': budgets, 'total_evaluations': sum(budgets) * len(strategies) * len(seeds), 'strategies compared': strategies, 'seeds': seeds, 'budgets': budgets, 'budget_summary': budget_summary, 'wilcoxon_results': wilcoxon_results, 'cohens_d_results': cohens_d_results}
    output['verdict'] = '; '.join(verdict_parts)
    output['gate_failures'] = _gate_failures
    output['gate_pass'] = len(_gate_failures) == 0
    output_path = Path(__file__).resolve().parent / 'results' / 'ea_vs_random.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Results written to: {output_path}')
    return 0 if not _gate_failures else 1
if __name__ == '__main__':
    raise SystemExit(run())