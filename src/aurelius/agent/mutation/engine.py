"""MutationEngine — multi-strategy molecule mutation engine.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. SMARTS functional-group replacement (high priority)
2. BRICS fragmentation + reassembly (medium priority)

Seeds are stored as MoleculeContext objects to enforce single-point parsing.

Delegates novelty checking to ``NoveltyValidator`` and fragment harvesting
to ``FragmentHarvester`` for single-responsibility separation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

from aurelius.agent.mutation.brics import (
    MIN_GROUNDING_SCORE,
    combined_grounding_score,
    find_complementary_pairs,
    inject_linkers,
)
from aurelius.agent.mutation.brics import (
    has_excessive_aliphatic_chain as _has_excessive_aliphatic_chain_fn,
)
from aurelius.agent.mutation.harvester import FragmentHarvester
from aurelius.agent.mutation.novelty import NoveltyValidator
from aurelius.agent.mutation.smarts import (
    ELECTROLYTE_FRAGMENT_POOL,
    ELECTROLYTE_SMARTS,
    is_electrolyte_like,
)
from aurelius.types import MoleculeContext, format_mixture_smiles
from aurelius.utils.chem_utils import _deserialize_fp

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class MutationEngine:
    """Multi-strategy molecule mutation engine for battery electrolytes.

    Generates candidate molecules from seed SMILES using two strategies:

    1. SMARTS functional-group replacement (high priority)
    2. BRICS fragmentation + reassembly (medium priority)

    Novelty checking and fragment harvesting are delegated to dedicated
    components (NoveltyValidator, FragmentHarvester).
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
        self._known_smiles: set[str] = set()
        self._load_known_electrolytes()

        self._generated_smiles: set[str] = set()
        self._rng = np.random.default_rng(42)
        self._smarts_rxns = self._init_smarts_rxns()
        self._adaptive_bias = adaptive_bias
        self._reaction_scores: dict[str, list[float]] = {}
        self._product_to_reaction: dict[str, str] = {}

        # Delegated components
        self._novelty_validator = NoveltyValidator(
            seed_smiles=self._seed_smiles,
            seed_scaffolds=self._seed_scaffolds,
            seed_fingerprints=self.seed_fingerprints,
            seed_pool=self.seed_pool,
            commercial_fps=self._commercial_fps,
            known_smiles=self._known_smiles,
            generated_smiles=self._generated_smiles,
            get_ctx=self._get_ctx,
        )
        self._fragment_harvester = FragmentHarvester(get_ctx=self._get_ctx)

    @staticmethod
    def _init_seeds(seed_smiles: list[str] | None) -> tuple[list[str], list[MoleculeContext], list[Any]]:
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
            cached_ctx = self._get_ctx(smi)
            if cached_ctx is not None:
                canon = Chem.MolToSmiles(cached_ctx.mol)
                if canon not in existing_smis:
                    self._commercial_fps.append(cached_ctx.get_ecfp4())
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

    # ------------------------------------------------------------------
    # Novelty — delegated to NoveltyValidator
    # ------------------------------------------------------------------

    def _is_known_smiles(self, canon: str) -> bool:
        return self._novelty_validator.is_known_smiles(canon)

    def _is_novel_scaffold(self, ctx: MoleculeContext) -> bool:
        return self._novelty_validator.is_novel_scaffold(ctx)

    def _is_novel_vs_commercial(self, fp: object, threshold: float = 0.85) -> bool:
        return self._novelty_validator.is_novel_vs_commercial(fp, threshold=threshold)

    def _is_trivial_alkyl_extension(self, ctx: MoleculeContext, min_extra_carbons: int = 2) -> bool:
        return self._novelty_validator.is_trivial_alkyl_extension(ctx, min_extra_carbons=min_extra_carbons)

    def _novelty_check(self, ctx: MoleculeContext, check_scaffold: bool = True, force_exploration: bool = False) -> bool:
        return self._novelty_validator.novelty_check(ctx, check_scaffold=check_scaffold, force_exploration=force_exploration)

    # ------------------------------------------------------------------
    # Fragment Harvesting — delegated to FragmentHarvester
    # ------------------------------------------------------------------

    def fragment_pool_size(self) -> int:
        return self._fragment_harvester.fragment_pool_size()

    def harvest_fragments(self, smiles: str, score: float = 65.0) -> None:
        self._fragment_harvester.harvest_fragments(smiles, score=score)

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
        harvested = self._fragment_harvester.get_harvested_fragments()

        if exploration_bias and harvested:
            repeat = max(1, len(ELECTROLYTE_FRAGMENT_POOL) // max(len(harvested), 1))
            source_smiles.extend(harvested * repeat)

        source_smiles.extend(harvested)
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
        # Grounding gate: reject BRICS products where < 40% of fragments
        # or functional groups map to commercial building blocks. Ensures
        # novel scaffolds remain synthesizable from catalog precursors.
        if combined_grounding_score(r_mol) < MIN_GROUNDING_SCORE:
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

    # ------------------------------------------------------------------
    # Mixture-level mutation
    # ------------------------------------------------------------------

    @staticmethod
    def _component_tanimoto(smi_a: str, smi_b: str) -> float:
        """Compute Tanimoto similarity between two component molecules."""
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return 1.0
        from rdkit.DataStructs import TanimotoSimilarity
        return TanimotoSimilarity(ctx_a.get_ecfp4(), ctx_b.get_ecfp4())

    def _mutate_mixture(
        self, smi_a: str, smi_b: str, frac: float, batch_size: int = 10
    ) -> list[str]:
        """Generate mixture variants by mutating components or perturbing fraction.

        Mutation types (randomly chosen):
          - 40%: mutate component A via SMARTS/BRICS
          - 40%: mutate component B via SMARTS/BRICS
          - 20%: perturb volume fraction frac_A by ±0.05-0.15
        """
        variants: list[str] = []
        r = self._rng.random()

        if r < 0.40:
            mutated = self.mutate(smi_a, batch_size=batch_size)
            for new_a in mutated[:3]:
                tan = self._component_tanimoto(new_a, smi_b)
                if tan <= 0.80:
                    variants.append(format_mixture_smiles(new_a, smi_b, frac))
        elif r < 0.80:
            mutated = self.mutate(smi_b, batch_size=batch_size)
            for new_b in mutated[:3]:
                tan = self._component_tanimoto(smi_a, new_b)
                if tan <= 0.80:
                    variants.append(format_mixture_smiles(smi_a, new_b, frac))
        else:
            delta = (self._rng.random() - 0.5) * 0.20
            new_frac = max(0.1, min(0.9, frac + delta))
            variants.append(format_mixture_smiles(smi_a, smi_b, new_frac))

        return list(set(variants))

    def propose_mixture_candidates(
        self,
        seed_smiles: list[str],
        n_mixtures: int = 10,
        batch_size: int = 10,
    ) -> list[str]:
        """Generate binary mixture candidates from a list of seed SMILES.

        Pairs molecules from the seed pool and creates mixture variants.
        Pairs with component Tanimoto > 0.80 are excluded as trivial.
        """
        if len(seed_smiles) < 2:
            return []
        unique_seeds = list(dict.fromkeys(seed_smiles))
        candidates: list[str] = []
        n_pairs = min(len(unique_seeds), 5)
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                smi_a = unique_seeds[i]
                smi_b = unique_seeds[j]
                tan = self._component_tanimoto(smi_a, smi_b)
                if tan > 0.80:
                    continue
                frac = round(self._rng.uniform(0.3, 0.7), 2)
                candidates.append(format_mixture_smiles(smi_a, smi_b, frac))
                mix_variants = self._mutate_mixture(smi_a, smi_b, frac, batch_size=batch_size)
                candidates.extend(mix_variants)
        unique = list(dict.fromkeys(candidates))
        if len(unique) > n_mixtures:
            idx = self._rng.choice(len(unique), size=n_mixtures, replace=False)
            unique = [unique[i] for i in idx]
        return unique
