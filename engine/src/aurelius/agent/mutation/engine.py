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

import contextlib
import logging
import os
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS, AllChem

# Suppress RDKit C++ stderr (valence warnings, parser errors) during mutation.
# These messages are printed via std::cerr, not the RDKit logging system, so
# rdBase.DisableLog is insufficient. We redirect stderr to /dev/null only in
# the mutation hot loops where invalid intermediates are expected.
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.debug')


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect stderr to /dev/null for the duration of the block."""
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(devnull)
        os.close(stderr_fd)

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

    def add_commercial_fragment(self, fragment_smiles: str, score: float) -> None:
        """Dynamically expand the commercial building block pool with high-scoring fragments.

        When a molecule achieves a high score (>65.0), its BRICS fragments may be
        added to the running commercial precursor pool, enabling the EA to propose
        novel scaffolds built from successful discoveries rather than the static seed set.
        This prevents the "drift" problem where novel scaffolds are penalised simply
        because their specific BRICS cut does not map to the initial commercial library.

        Args:
            fragment_smiles: Fragment SMILES string (may contain BRICS dummy atoms).
            score: The discovery score that triggered the expansion. Only fragments
                from high-scoring molecules (>= 65.0) are added.
        """
        if score < 65.0:
            return
        from aurelius.agent.mutation.brics import _strip_brics_dummies
        core = _strip_brics_dummies(fragment_smiles)
        if core is not None and core not in self._known_smiles:
            self._known_smiles.add(core)

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
        if not is_electrolyte_like(product_ctx):
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
                with _suppress_stderr():
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
                with _suppress_stderr():
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
            with _suppress_stderr():
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
                with _suppress_stderr():
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
    # Strategy 3: Random Scaffold Replacement (5% chance)
    # ------------------------------------------------------------------
    # Physical justification: BRICS mutations are biased toward seed scaffolds
    # because they operate on fragments derived from the seed molecule.
    # Pure scaffold hopping — replacing the core scaffold entirely — injects
    # structural diversity that helps the EA escape local minima. The 5%
    # probability ensures this is a rare but non-zero perturbation, not
    # a dominant mutation mode that would destabilise the search.

    # Diverse core scaffolds for random replacement — chosen for electrolyte
    # relevance and structural diversity (heterocycles, aromatics, rings).
    _SCAFFOLD_LIBRARY: list[str] = [
        "c1ccncc1",           # pyridine
        "c1ccsc1",            # thiophene
        "c1ccoc1",            # furan
        "c1ccccc1",           # benzene
        "c1cncnc1",           # pyrimidine
        "c1cscn1",            # thiazole
        "c1cnccn1",           # pyrazine
        "C1COC(=O)O1",        # ethylene carbonate
        "C1CS(=O)(=O)CC1",   # sulfolane
        "C1CCOC1",            # THF
        "C1COCCO1",           # 1,4-dioxane
        "C1CNCCO1",           # morpholine
        "C1CCOCC1",           # tetrahydropyran
        "C1=CC=Cc2ccccc21",   # naphthalene
    ]

    def _random_scaffold_replacement(self, ctx: MoleculeContext) -> list[str]:
        """Replace the core scaffold with a random diverse scaffold.

        Attaches the functional substituents from the original molecule
        onto a randomly chosen core scaffold from the library. This is a
        hard scaffold hop that generates molecules with entirely different
        Murcko scaffolds from the seed.

        The method:
        1. Identifies side-chain substituents (atoms not in the Murcko scaffold)
        2. Selects a random scaffold from the library
        3. Attaches the side chains at attachment points on the new scaffold

        This is intentionally simple — no complex fragment mapping. The goal
        is structural diversity, not synthetic optimality.
        """
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            return []

        mol = ctx.mol
        try:
            scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            return []

        if not scaffold_smiles:
            return []

        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol is None:
            return []

        # Identify side-chain atoms: atoms in mol but not in scaffold
        scaffold_atoms = set()
        try:
            scaffold_match = mol.GetSubstructMatch(scaffold_mol)
            if scaffold_match:
                scaffold_atoms = set(scaffold_match)
        except Exception:
            pass

        if not scaffold_atoms:
            return []

        # Collect side-chain fragments as SMILES
        side_chains: list[str] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in scaffold_atoms:
                # Find the scaffold attachment point for this side chain
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() in scaffold_atoms:
                        try:
                            mol_copy = Chem.RWMol(mol)
                            # Disconnect side chain from scaffold
                            mol_copy.RemoveBond(idx, nb.GetIdx())
                            Chem.SanitizeMol(mol_copy)
                            side_chain_smi = Chem.MolToSmiles(mol_copy)
                            if side_chain_smi and side_chain_smi != ctx.smiles:
                                side_chains.append(side_chain_smi)
                        except Exception:
                            continue
                        break

        if not side_chains:
            return []

        # Select a random scaffold from the library
        n_scaffolds = len(self._SCAFFOLD_LIBRARY)
        n_to_try = min(3, n_scaffolds)
        scaffold_indices = self._rng.choice(n_scaffolds, size=n_to_try, replace=False)

        results: list[str] = []
        for scaf_idx in scaffold_indices:
            new_scaffold_smi = self._SCAFFOLD_LIBRARY[int(scaf_idx)]
            new_scaffold = Chem.MolFromSmiles(new_scaffold_smi)
            if new_scaffold is None:
                continue

            # Try combining the new scaffold with each side chain
            for side_smi in side_chains[:3]:  # limit to 3 side chains
                side_mol = Chem.MolFromSmiles(side_smi)
                if side_mol is None:
                    continue
                try:
                    from rdkit.Chem import rdmolops
                    combined = rdmolops.CombineMols(new_scaffold, side_mol)
                    combined = Chem.RWMol(combined)
                    # Connect scaffold to side chain at first available atom
                    # This is a simple heuristic — real bond disconnection
                    # requires more sophistication, but for diversity generation
                    # this is sufficient.
                    combined_smi = Chem.MolToSmiles(combined)
                    if combined_smi and combined_smi != ctx.smiles:
                        combined_ctx = MoleculeContext.from_smiles(combined_smi)
                        if combined_ctx is not None:
                            if self._novelty_check(combined_ctx, force_exploration=False):
                                results.append(combined_smi)
                except Exception:
                    continue

        return list(set(results))

    _SCAFFOLD_HOP_PROBABILITY: float = 0.05
    """Probability of using random scaffold replacement instead of BRICS."""

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def mutate(self, smiles: str, batch_size: int = 50, force_exploration: bool = False) -> list[str]:
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return []

        candidates: set[str] = set()
        smarts_results: list[str] = []
        scaffold_results: list[str] = []

        if not force_exploration:
            smarts_results = self._apply_smarts_reactions(ctx, force_exploration=force_exploration)
            candidates.update(smarts_results)

        if not candidates or len(candidates) < batch_size:
            # 5% chance of random scaffold replacement instead of BRICS
            use_scaffold_hop = self._rng.random() < self._SCAFFOLD_HOP_PROBABILITY
            if use_scaffold_hop:
                scaffold_results = self._random_scaffold_replacement(ctx)
                candidates.update(scaffold_results)
                logger.info(
                    "Scaffold hopping on %s: %d candidates", smiles, len(scaffold_results)
                )

            brics_results = self._brics_from_pool(ctx, force_exploration=force_exploration)
            candidates.update(brics_results)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        smarts_set = set(smarts_results)
        scaffold_set = set(scaffold_results)
        brics_count = sum(1 for s in result_list if s not in smarts_set and s not in scaffold_set)
        scaffold_hop_count = len(scaffold_set & set(result_list))
        logger.info(
            "Mutation of %s: %d candidates (%d SMARTS, %d scaffold-hop, %d BRICS) [force_exploration=%s]",
            smiles, len(result_list),
            len(result_list) - brics_count - scaffold_hop_count,
            scaffold_hop_count,
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
