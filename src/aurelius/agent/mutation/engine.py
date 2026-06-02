"""MutationEngine — multi-strategy molecule mutation engine.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. SMARTS functional-group replacement (high priority)
2. BRICS fragmentation + reassembly (medium priority)

Seeds are stored as MoleculeContext objects to enforce single-point parsing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.constants import (
    ELECTROCHEMICALLY_UNSTABLE_PATTERNS as _EC_UNSTABLE_PATTERNS,
    ELECTROLYTE_MIN_HETEROATOM_RATIO,
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_UNSTABLE_PATTERNS,
)
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import _deserialize_fp

from aurelius.agent.mutation.brics import (
    _MAX_HARVESTED_FRAGMENTS,
    find_complementary_pairs,
    get_brics_types,
    has_excessive_aliphatic_chain as _has_excessive_aliphatic_chain_fn,
    inject_linkers,
)
from aurelius.agent.mutation.smarts import (
    ELECTROLYTE_FRAGMENT_POOL,
    ELECTROLYTE_SMARTS,
    is_electrolyte_like,
)

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None

logger = logging.getLogger(__name__)


class MutationEngine:
    """Multi-strategy molecule mutation engine for battery electrolytes.

    Generates candidate molecules from seed SMILES using two strategies
    in priority order:

    1. SMARTS functional-group replacement (high priority)
    2. BRICS fragmentation + reassembly (medium priority)

    Seeds are stored as MoleculeContext objects to enforce single-point parsing.
    """

    def __init__(self, seed_smiles: list[str] | None = None, known_fps_hex: list[str] | None = None, adaptive_bias: bool = True) -> None:
        self.seed_pool, self.seed_contexts, self.seed_fingerprints = self._init_seeds(seed_smiles)

        self._commercial_fps = []
        for h in known_fps_hex or []:
            try:
                self._commercial_fps.append(_deserialize_fp(h))
            except Exception:
                continue

        self._seed_smiles, self._seed_scaffolds = self._init_smiles_and_scaffolds()

        self._ctx_cache: dict[str, MoleculeContext] = {}
        self._known_smiles = set()
        self._load_known_electrolytes()

        self._generated_smiles = set()
        self._harvested_fragments = []
        self._harvested_fragment_set = set()
        self._harvested_fragment_scores: dict[str, float] = {}
        self._rng = np.random.default_rng(42)
        self._smarts_rxns = self._init_smarts_rxns()
        self._adaptive_bias = adaptive_bias
        self._reaction_scores: dict[str, list[float]] = {}
        self._product_to_reaction: dict[str, str] = {}

    @staticmethod
    def _init_seeds(seed_smiles: list[str] | None) -> tuple[list[str], list[MoleculeContext], list]:
        from pathlib import Path
        if seed_smiles is None:
            import json
            json_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "tier0_seed_smiles.json")
            with open(json_path) as f:
                seed_smiles = json.load(f)
        pool = list(set(seed_smiles))
        contexts = []
        fps = []
        for smi in pool:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                contexts.append(ctx)
                fps.append(ctx.get_ecfp4())
        return pool, contexts, fps

    def _init_smiles_and_scaffolds(self) -> tuple[set[str], set[str]]:
        smiles_set = set()
        scaffold_set = set()
        for ctx in self.seed_contexts:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                smiles_set.add(canon)
                if MurckoScaffold is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                    if scaffold:
                        scaffold_set.add(scaffold)
            except Exception:
                continue
        return smiles_set, scaffold_set

    @staticmethod
    def _init_smarts_rxns() -> list[tuple[Any, str]]:
        rxns = []
        for smarts, name in ELECTROLYTE_SMARTS:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                rxns.append((rxn, name))
            except Exception:
                logger.debug("Failed to parse SMARTS '%s' (%s)", smarts, name)
        return rxns

    def _get_ctx(self, smiles: str) -> MoleculeContext | None:
        if smiles not in self._ctx_cache:
            ctx = MoleculeContext.from_smiles(smiles)
            if ctx is not None:
                self._ctx_cache[smiles] = ctx
            else:
                self._ctx_cache[smiles] = None  # type: ignore[assignment]
        return self._ctx_cache.get(smiles)

    def _load_known_electrolytes(self) -> None:
        import json
        import os

        json_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "known_electrolytes.json"
        )
        try:
            with open(json_path) as f:
                smiles_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        existing_smis = set()
        for ctx in self.seed_contexts:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                existing_smis.add(canon)
            except Exception:
                continue

        for smi in smiles_list:
            ctx = self._get_ctx(smi)
            if ctx is not None:
                canon = Chem.MolToSmiles(ctx.mol)
                if canon not in existing_smis:
                    self._commercial_fps.append(ctx.get_ecfp4())
                    self._known_smiles.add(canon)

        logger.info(
            "Loaded %d known electrolyte fingerprints for global novelty checking.",
            len(self._commercial_fps),
        )

    def commercial_db_size(self) -> int:
        return len(self._commercial_fps)

    def add_to_db(self, smiles: str) -> None:
        ctx = self._get_ctx(smiles)
        if ctx is not None:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                self._generated_smiles.add(canon)
            except Exception:
                pass

    def _eligible_for_harvest(self, frag_smi: str) -> MoleculeContext | None:
        if frag_smi in self._harvested_fragment_set:
            return None
        if frag_smi in ELECTROLYTE_FRAGMENT_POOL:
            return None
        f_ctx = self._get_ctx(frag_smi)
        if f_ctx is None or f_ctx.mw > 250 or f_ctx.hbd > 0:
            return None
        if self._harvested_fragments and self._fragment_too_similar(frag_smi, self._harvested_fragments, threshold=0.85):
            return None
        return f_ctx

    def harvest_fragments(self, smiles: str, score: float = 65.0) -> None:
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return
        try:
            for frag_smi in BRICS.BRICSDecompose(ctx.mol):
                if frag_smi in self._harvested_fragment_set:
                    if score > self._harvested_fragment_scores.get(frag_smi, 0):
                        self._harvested_fragment_scores[frag_smi] = score
                    continue
                if self._eligible_for_harvest(frag_smi) is None:
                    continue
                self._harvested_fragments.append(frag_smi)
                self._harvested_fragment_set.add(frag_smi)
                self._harvested_fragment_scores[frag_smi] = score
                if len(self._harvested_fragments) > _MAX_HARVESTED_FRAGMENTS:
                    worst = min(self._harvested_fragments, key=lambda f: self._harvested_fragment_scores.get(f, 0))
                    self._harvested_fragments.remove(worst)
                    self._harvested_fragment_set.discard(worst)
                    self._harvested_fragment_scores.pop(worst, None)
        except Exception:
            logger.debug("Failed to harvest fragments from %s", smiles, exc_info=True)

    def _fragment_too_similar(self, new_smi: str, existing_smis: list[str], threshold: float = 0.85) -> bool:
        new_ctx = self._get_ctx(new_smi)
        if new_ctx is None:
            return False
        new_fp = new_ctx.get_ecfp4()
        existing_fps: list = []
        for old_smi in existing_smis:
            old_ctx = self._get_ctx(old_smi)
            if old_ctx is None:
                continue
            existing_fps.append(old_ctx.get_ecfp4())
        if not existing_fps:
            return False
        from rdkit.DataStructs import BulkTanimotoSimilarity
        sims = BulkTanimotoSimilarity(new_fp, existing_fps)
        return any(s >= threshold for s in sims)

    def fragment_pool_size(self) -> int:
        return len(ELECTROLYTE_FRAGMENT_POOL) + len(self._harvested_fragments)

    def _is_trivial_alkyl_extension(
        self,
        ctx: MoleculeContext,
        seed_smiles: list[str],
        seed_fingerprints: list,
        min_extra_carbons: int = 2,
    ) -> bool:
        def _heteroatom_profile(mol: Chem.Mol) -> dict[int, int]:
            profile: dict[int, int] = {}
            for a in mol.GetAtoms():
                z = a.GetAtomicNum()
                if z in {7, 8, 9, 15, 16, 17, 35}:
                    profile[z] = profile.get(z, 0) + 1
            return profile

        def _count_carbons(mol: Chem.Mol) -> int:
            return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

        ctx_profile = _heteroatom_profile(ctx.mol)
        ctx_c = _count_carbons(ctx.mol)

        if not hasattr(self, '_seed_trivial_cache'):
            self._seed_trivial_cache: list[tuple[dict[int, int], int]] = []
            for seed_smi in seed_smiles:
                seed_ctx = self._get_ctx(seed_smi)
                if seed_ctx is None:
                    continue
                self._seed_trivial_cache.append(
                    (_heteroatom_profile(seed_ctx.mol), _count_carbons(seed_ctx.mol))
                )

        for seed_profile, seed_c in self._seed_trivial_cache:
            if ctx_profile == seed_profile and ctx_c >= seed_c + min_extra_carbons:
                return True

        return False

    def _is_known_smiles(self, canon: str) -> bool:
        return canon in self._seed_smiles or canon in self._known_smiles or canon in self._generated_smiles

    def _is_novel_scaffold(self, ctx: MoleculeContext) -> bool:
        if MurckoScaffold is None or not self._seed_scaffolds:
            return True
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
            if scaffold and scaffold in self._seed_scaffolds:
                return False
        except Exception:
            pass
        return True

    def _is_novel_vs_commercial(self, fp: Any, threshold: float = 0.85) -> bool:
        if not self._commercial_fps:
            return True
        sims = BulkTanimotoSimilarity(fp, self._commercial_fps)
        return not any(s >= threshold for s in sims)

    def _novelty_check(self, ctx: MoleculeContext, check_scaffold: bool = True, force_exploration: bool = False) -> bool:
        try:
            canon = Chem.MolToSmiles(ctx.mol)
        except Exception:
            return False
        if self._is_known_smiles(canon):
            return False
        if check_scaffold and not self._is_novel_scaffold(ctx):
            return False
        if check_scaffold and self._is_trivial_alkyl_extension(ctx, self.seed_pool, self.seed_fingerprints):
            return False
        fp = ctx.get_ecfp4()
        threshold = 0.90 if force_exploration else 0.85
        if not self._is_novel_vs_commercial(fp, threshold=threshold):
            return False
        return True

    # ------------------------------------------------------------------
    # Strategy 1: SMARTS functional-group replacement
    # ------------------------------------------------------------------

    def _process_smarts_product(self, product: Chem.Mol, seed_smi: str, reaction_name: str | None = None, force_exploration: bool = False) -> str | None:
        if product is None:
            return None
        try:
            Chem.SanitizeMol(product)
        except Exception:
            return None
        product_smi = Chem.MolToSmiles(product)
        if not product_smi or product_smi == seed_smi:
            return None
        if product_smi not in self._ctx_cache:
            product_ctx = MoleculeContext(smiles=product_smi, mol=product)
            self._ctx_cache[product_smi] = product_ctx
        else:
            product_ctx = self._ctx_cache[product_smi]
        if not product_ctx.is_valid_electrolyte_mol():
            return None
        if not self._novelty_check(product_ctx, force_exploration=force_exploration):
            return None
        if reaction_name is not None and self._adaptive_bias:
            self._product_to_reaction[product_smi] = reaction_name
        return product_smi

    def _apply_smarts_reactions(self, ctx: MoleculeContext, force_exploration: bool = False) -> list[str]:
        results: list[str] = []
        rxns = self._smarts_rxns
        if self._adaptive_bias and self._reaction_scores:
            def _mean_score(name: str) -> float:
                scores = self._reaction_scores.get(name, [])
                return float(np.mean(scores)) if scores else 0.0
            rxns = sorted(rxns, key=lambda r: _mean_score(r[1]), reverse=True)
        for rxn, name in rxns:
            try:
                for product_tuple in rxn.RunReactants((ctx.mol,)):
                    for product in product_tuple:
                        p_smi = self._process_smarts_product(product, ctx.smiles, reaction_name=name, force_exploration=force_exploration)
                        if p_smi:
                            results.append(p_smi)
            except Exception:
                logger.debug("SMARTS reaction '%s' failed for %s", name, ctx.smiles)
        return list(set(results))

    def record_reaction_success(self, smiles: str, score: float) -> None:
        """Record the score achieved by a molecule produced via a SMARTS reaction.
        
        Updates the reaction's success history so that high-performing reactions
        are prioritised in future generations.
        """
        if not self._adaptive_bias:
            return
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return
        rxn_name = self._product_to_reaction.get(ctx.smiles)
        if rxn_name is not None:
            if rxn_name not in self._reaction_scores:
                self._reaction_scores[rxn_name] = []
            self._reaction_scores[rxn_name].append(score)

    # ------------------------------------------------------------------
    # Strategy 2: BRICS fragmentation + reassembly
    # ------------------------------------------------------------------

    def _collect_fragments_from_smiles(self, smiles_list: list[str]) -> list[Chem.Mol]:
        frags: list[Chem.Mol] = []
        for smi in smiles_list:
            ctx = self._get_ctx(smi)
            if ctx is None or ctx.mw > 250 or ctx.hbd > 0:
                continue
            try:
                for fs in BRICS.BRICSDecompose(ctx.mol):
                    frag_ctx = MoleculeContext.from_brics_fragment(fs)
                    if frag_ctx is not None:
                        frags.append(frag_ctx.mol)
            except Exception:
                continue
        return frags

    def _load_brics_fragments(self, exploration_bias: bool = False) -> list[Chem.Mol]:
        source_smiles: list[str] = list(ELECTROLYTE_FRAGMENT_POOL)

        if exploration_bias and self._harvested_fragments:
            repeat = max(1, len(ELECTROLYTE_FRAGMENT_POOL) // max(len(self._harvested_fragments), 1))
            source_smiles.extend(self._harvested_fragments * repeat)

        source_smiles.extend(self._harvested_fragments)
        return self._collect_fragments_from_smiles(source_smiles)

    def _is_electrolyte_like(self, ctx: MoleculeContext) -> bool:
        return is_electrolyte_like(ctx)

    def _collect_brics_fragments(self, ctx: MoleculeContext, force_exploration: bool) -> list[Chem.Mol]:
        try:
            seed_frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
        except Exception:
            return []
        seed_frags: list[Chem.Mol] = []
        for fs in seed_frag_smiles:
            frag_ctx = MoleculeContext.from_brics_fragment(fs)
            if frag_ctx is not None:
                seed_frags.append(frag_ctx.mol)

        pool_frags = self._load_brics_fragments(exploration_bias=force_exploration)
        return seed_frags + pool_frags

    @staticmethod
    def _has_excessive_aliphatic_chain(mol: Chem.Mol, max_chain: int = 12) -> bool:
        return _has_excessive_aliphatic_chain_fn(mol, max_chain)

    def _validate_brics_product(self, r_mol: Chem.Mol, force_exploration: bool = False) -> str | None:
        if r_mol is None:
            return None
        try:
            Chem.SanitizeMol(r_mol)
        except Exception:
            return None
        s = Chem.MolToSmiles(r_mol)
        if not s:
            return None
        product_ctx = MoleculeContext(smiles=s, mol=r_mol)
        if not product_ctx.is_valid_electrolyte_mol():
            return None
        if not self._novelty_check(product_ctx, force_exploration=force_exploration):
            return None
        if not is_electrolyte_like(product_ctx):
            return None
        if self._has_excessive_aliphatic_chain(r_mol):
            return None
        return s

    def _build_from_pairs(self, all_frags: list[Chem.Mol], valid_pairs: list[tuple[int, int]], force_exploration: bool = False) -> list[str]:
        generated: list[str] = []
        n_pairs = len(valid_pairs)
        indices = self._rng.integers(0, n_pairs, size=min(150, n_pairs * 5))
        for idx in indices:
            try:
                i, j = valid_pairs[idx]
                for r_mol in BRICS.BRICSBuild([all_frags[i], all_frags[j]]):
                    s = self._validate_brics_product(r_mol, force_exploration=force_exploration)
                    if s:
                        generated.append(s)
            except Exception:
                continue
        return generated

    def _brics_from_pool(self, ctx: MoleculeContext, force_exploration: bool = False) -> list[str]:
        generated: list[str] = []

        all_frags = self._collect_brics_fragments(ctx, force_exploration)
        if len(all_frags) < 2:
            return generated

        valid_pairs = find_complementary_pairs(all_frags)
        if not valid_pairs:
            return generated

        n_participating = len({i for p in valid_pairs for i in p})
        if n_participating < len(all_frags) * 0.2 and force_exploration:
            inject_linkers(all_frags)
            valid_pairs = find_complementary_pairs(all_frags)
            if not valid_pairs:
                return generated

        generated = self._build_from_pairs(all_frags, valid_pairs, force_exploration=force_exploration)
        return list(set(generated))

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def mutate(self, smiles: str, batch_size: int = 50, force_exploration: bool = False) -> list[str]:
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return []

        candidates: set[str] = set()

        if not force_exploration:
            smarts_results = self._apply_smarts_reactions(ctx, force_exploration=force_exploration)
            candidates.update(smarts_results)

        if not candidates or len(candidates) < batch_size:
            brics_results = self._brics_from_pool(ctx, force_exploration=force_exploration)
            candidates.update(brics_results)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        brics_count = sum(1 for s in result_list if s not in (smarts_results if not force_exploration else set()))
        logger.info(
            "Mutation of %s: %d candidates (%d SMARTS, %d BRICS) [force_exploration=%s]",
            smiles, len(result_list), len(result_list) - brics_count,
            brics_count, force_exploration,
        )
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50, force_exploration: bool = False) -> list[str]:
        all_variants: list[str] = []
        for smi in batch_smiles:
            variants = self.mutate(smi, batch_size, force_exploration=force_exploration)
            all_variants.extend(variants)
        return list(set(all_variants))

    def propose_candidates(
        self,
        n_candidates: int = 1000,
        batch_size: int = 50,
    ) -> list[str]:
        all_variants: list[str] = []
        for smi in self.seed_pool:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)

        unique = list(dict.fromkeys(all_variants))
        if len(unique) > n_candidates:
            indices = self._rng.choice(len(unique), size=n_candidates, replace=False)
            unique = [unique[i] for i in indices]
        return unique
