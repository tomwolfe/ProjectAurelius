"""Ablation: tournament vs NSGA-II vs random_top_k on the SAME pool and budget.

Phase 3 (aurelius-global-closure): the discovery loop must justify its
selector. All three arms select from one shared, pre-screened candidate pool
(scores computed once, shared identically), with the same batch budget, over
the same seeds. The verdict reports mean AND minimum across seeds (gate on
min, not mean). Report-only: this benchmark carries no CI gate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

RESULTS_PATH = Path(__file__).parent / 'results' / 'selection_ablation.json'
SEEDS = [42, 123, 7, 2024, 99]
BUDGET = 20
POOL_TARGET = 150


def _murcko(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _build_pool() -> list[str]:
    from aurelius.agent.mutation import MutationEngine

    panel = ['COC(=O)OC', 'C1COCCO1', 'CS(=O)(=O)C', 'CC#N', 'C1CCOC1']
    engine = MutationEngine(seed_smiles=None, seed=0)
    pool: list[str] = []
    for smi in engine.seed_pool:
        if len(pool) >= 400:
            break
        try:
            pool.extend(engine.mutate(smi, 30, force_exploration=False))
        except Exception:
            continue
        try:
            pool.extend(engine.mutate(smi, 30, force_exploration=True))
        except Exception:
            continue
    try:
        known = json.load(open(Path(__file__).parent / 'data' / 'known_electrolytes.json'))
        pool.extend([k for k in known if isinstance(k, str)])
    except Exception:
        pass
    seen: list[str] = []
    for smi in pool:
        if smi not in seen:
            seen.append(smi)
    return seen[:POOL_TARGET]


def run_ablation() -> dict:
    from aurelius.agent.loop import DiscoveryLoop, _collect_obj_scores
    from aurelius.agent.mutation import MutationEngine
    from aurelius.agent.state import LoopState
    from aurelius.pipeline import AureliusPipeline
    from aurelius.types import MoleculeContext

    pool = _build_pool()
    pipeline = AureliusPipeline()
    pipeline.initialize()

    contexts: list[MoleculeContext] = []
    scores: list[float] = []
    obj_scores: dict[str, list[float]] = {
        'total_score': [], 'dielectric_proxy': [], 'viscosity_proxy': [],
        'li_solvation_proxy': [], 'homo_eV': [], 'lumo_eV': [], 'sa_score': [],
        'synthesizability_complexity': [], 'synthesis_depth': [], 'confidence': [],
        'combined_grounding_score': [], 'novelty_to_seed': [],
    }
    for smi in pool:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        try:
            res = pipeline.screen_molecule(ctx)
        except Exception:
            continue
        t2 = res.get('tier2') or {}
        score_data = res.get('score') or {}
        total = float(score_data.get('total_score', 0.0))
        if total <= 0:
            continue
        contexts.append(ctx)
        scores.append(total)
        _collect_obj_scores(
            obj_scores, total, t2, score_data,
            float(res.get('conformal_confidence', 1.0)),
            float(score_data.get('sub_scores', {}).get('novelty', 0.0) or 0.0),
        )
    try:
        known_scaffolds = {_murcko(k) for k in json.load(
            open(Path(__file__).parent / 'data' / 'known_electrolytes.json')) if isinstance(k, str)}
        known_scaffolds.discard(None)
    except Exception:
        known_scaffolds = set()

    engine = MutationEngine(seed_smiles=['CCO'], seed=0)
    state = LoopState(path='/tmp/ablation_state.json')
    state.clear()
    loop = DiscoveryLoop(pipeline=pipeline, engine=engine, state=state,
                         max_generations=1, batch_size=BUDGET, seed=0)

    arms: dict[str, dict] = {a: {'top_mean': [], 'novelty': []} for a in ('tournament', 'nsga2', 'random_top_k')}
    if len(contexts) < 3 * BUDGET:
        raise RuntimeError(
            f'ablation pool too small ({len(contexts)} scored positives): '
            f'selectors would return the pool unchanged and the comparison '
            f'would be vacuous (fail-closed, no silent pass)'
        )
    for seed in SEEDS:
        rng = random.Random(seed)
        # Tournament arm (production path, NSGA-II off).
        loop.use_nsga2 = False
        loop.seed = seed
        loop._rng = random.Random(seed)
        sel_t, _ = loop._select_batch(list(contexts), list(scores), [1.0] * len(contexts),
                                      {k: list(v) for k, v in obj_scores.items()}, None)
        # NSGA-II arm (production path, NSGA-II on).
        loop.use_nsga2 = True
        sel_n, _ = loop._select_batch(list(contexts), list(scores), [1.0] * len(contexts),
                                      {k: list(v) for k, v in obj_scores.items()}, None)
        # random_top_k arm: uniform subset, ranked by shared score.
        idx = rng.sample(range(len(contexts)), min(BUDGET, len(contexts)))
        sel_r = [contexts[i] for i in sorted(idx, key=lambda i: -scores[i])]
        for name, sel in (('tournament', sel_t), ('nsga2', sel_n), ('random_top_k', sel_r)):
            got = [scores[contexts.index(c)] for c in sel]
            top = sorted(got, reverse=True)[:10]
            scafs = {_murcko(c.smiles) for c in sel} - {None}
            arms[name]['top_mean'].append(float(np.mean(top)) if top else 0.0)
            arms[name]['novelty'].append(len(scafs - known_scaffolds) / max(len(scafs), 1))
    verdict: dict = {'pool_size': len(contexts), 'budget': BUDGET, 'seeds': SEEDS, 'arms': {}}
    for name, m in arms.items():
        verdict['arms'][name] = {
            'top_mean': round(float(np.mean(m['top_mean'])), 3),
            'top_mean_min': round(float(min(m['top_mean'])), 3),
            'novelty_mean': round(float(np.mean(m['novelty'])), 3),
            'novelty_min': round(float(min(m['novelty'])), 3),
        }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(verdict, indent=1))
    print('=== SELECTION ABLATION (shared pool, shared scores) ===')
    print(f"pool={verdict['pool_size']} budget={BUDGET} seeds={SEEDS}")
    for name, m in verdict['arms'].items():
        print(f"  {name:12s} top10 mean={m['top_mean']:.2f} min={m['top_mean_min']:.2f} "
              f"novelty mean={m['novelty_mean']:.3f} min={m['novelty_min']:.3f}")
    print(f'wrote {RESULTS_PATH}')
    return verdict


if __name__ == '__main__':
    run_ablation()
