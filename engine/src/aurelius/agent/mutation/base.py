"""Abstract base for pluggable mutation strategies.

Defines the ``MutationStrategy`` interface and shared ``StrategyContext``
dataclass that decouples strategy implementations from the engine
orchestrator. Concrete strategies live in sibling modules.
"""

from __future__ import annotations

import contextlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from aurelius.types import MoleculeContext

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None

logger = logging.getLogger(__name__)


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


@dataclass
class StrategyContext:
    """Shared runtime context injected into every mutation strategy.

    Centralises state that would otherwise require strategies to hold
    a back-reference to the engine.
    """

    ctx_cache: dict[str, MoleculeContext | None] = field(default_factory=dict)
    novelty_validator: NoveltyValidator | None = None
    fragment_harvester: FragmentHarvester | None = None
    rng: Any = field(default_factory=lambda: np.random.default_rng(42))
    adaptive_bias: bool = True
    reaction_scores: dict[str, list[float]] = field(default_factory=dict)
    product_to_reaction: dict[str, str] = field(default_factory=dict)
    strict: bool = False
    seed_fingerprints: list[Any] = field(default_factory=list)


class MutationStrategy(ABC):
    """Pluggable mutation strategy.

    Each strategy implements ``mutate`` which receives a seed
    ``MoleculeContext`` and a shared ``StrategyContext`` and returns
    a list of candidate SMILES strings.
    """

    @abstractmethod
    def mutate(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
        batch_size: int = 50,
        force_exploration: bool = False,
        diagnostics: list[str] | None = None,
    ) -> list[str]:
        ...


class SmartsStrategy(MutationStrategy):
    """SMARTS functional-group replacement strategy.

    Applies electrolyte-relevant SMARTS transformations (fluorination,
    methylation, carbonate/ether edits, scaffold-hopping ring edits).
    """

    def __init__(self) -> None:
        self._smarts_rxns = self._init_smarts_rxns()

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

    def _process_smarts_product(
        self,
        product: Chem.Mol,
        seed_smi: str,
        context: StrategyContext,
        reaction_name: str | None = None,
        force_exploration: bool = False,
        _diagnostics: list[str] | None = None,
    ) -> str | None:
        if product is None:
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: product is None")
            return None
        try:
            Chem.SanitizeMol(product)
        except Exception:
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: invalid valence")
            return None
        product_smi = Chem.MolToSmiles(product)
        if not product_smi or product_smi == seed_smi:
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: failed — identical to seed")
            return None
        if product_smi not in context.ctx_cache:
            product_ctx = MoleculeContext(smiles=product_smi, mol=product)
            context.ctx_cache[product_smi] = product_ctx
        else:
            product_ctx = context.ctx_cache[product_smi]
        if not product_ctx.is_valid_electrolyte_mol():
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: failed: invalid valence")
            return None
        if not is_electrolyte_like(product_ctx):
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: failed: not electrolyte-like")
            return None
        if context.novelty_validator is not None and not context.novelty_validator.novelty_check(
            product_ctx, force_exploration=force_exploration,
        ):
            if _diagnostics is not None:
                _diagnostics.append("SMARTS: failed: novelty check")
            return None
        if reaction_name is not None and context.adaptive_bias:
            context.product_to_reaction[product_smi] = reaction_name
        return product_smi

    def mutate(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
        batch_size: int = 50,
        force_exploration: bool = False,
        diagnostics: list[str] | None = None,
    ) -> list[str]:
        if force_exploration:
            return []

        results: list[str] = []
        rxns = self._smarts_rxns
        if context.adaptive_bias and context.reaction_scores:
            def _mean_score(name: str) -> float:
                scores = context.reaction_scores.get(name, [])
                return float(np.mean(scores)) if scores else 0.0
            rxns = sorted(rxns, key=lambda r: _mean_score(r[1]), reverse=True)

        for rxn, name in rxns:
            try:
                with _suppress_stderr():
                    for product_tuple in rxn.RunReactants((ctx.mol,)):
                        for product in product_tuple:
                            p_smi = self._process_smarts_product(
                                product, ctx.smiles, context,
                                reaction_name=name,
                                force_exploration=force_exploration,
                                _diagnostics=diagnostics,
                            )
                            if p_smi:
                                results.append(p_smi)
            except Exception as exc:
                logger.warning("SMARTS reaction failed for %s: %s", ctx.smiles, exc)
                if context.strict:
                    raise
        return list(set(results))


class BricsStrategy(MutationStrategy):
    """BRICS fragmentation + reassembly strategy.

    Includes the 5 % random scaffold-hopping perturbation as an
    alternative to BRICS recombination for structural diversity.
    """

    _SCAFFOLD_LIBRARY: list[str] = [
        "c1ccncc1",
        "c1ccsc1",
        "c1ccoc1",
        "c1ccccc1",
        "c1cncnc1",
        "c1cscn1",
        "c1cnccn1",
        "C1COC(=O)O1",
        "C1CS(=O)(=O)CC1",
        "C1CCOC1",
        "C1COCCO1",
        "C1CNCCO1",
        "C1CCOCC1",
        "C1=CC=Cc2ccccc21",
    ]
    _SCAFFOLD_HOP_PROBABILITY: float = 0.05

    def _collect_fragments_from_smiles(
        self, smiles_list: list[str], context: StrategyContext
    ) -> list[Chem.Mol]:
        frags: list[Chem.Mol] = []
        for smi in smiles_list:
            cached = context.ctx_cache.get(smi)
            if cached is None:
                cached = MoleculeContext.from_smiles(smi)
                context.ctx_cache[smi] = cached
            if cached is None or cached.mw > 250 or cached.hbd > 0:
                continue
            try:
                with _suppress_stderr():
                    for fs in BRICS.BRICSDecompose(cached.mol):
                        frag_ctx = MoleculeContext.from_brics_fragment(fs)
                        if frag_ctx is not None:
                            frags.append(frag_ctx.mol)
            except Exception as exc:
                logger.warning("BRICS decomposition failed for %s: %s", smi, exc)
                if context.strict:
                    raise
                continue
        return frags

    def _load_brics_fragments(
        self, context: StrategyContext, exploration_bias: bool = False
    ) -> list[Chem.Mol]:
        source_smiles: list[str] = list(ELECTROLYTE_FRAGMENT_POOL)
        if context.fragment_harvester is not None:
            harvested = context.fragment_harvester.get_harvested_fragments()
            if exploration_bias and harvested:
                repeat = max(
                    1,
                    len(ELECTROLYTE_FRAGMENT_POOL) // max(len(harvested), 1),
                )
                source_smiles.extend(harvested * repeat)
            source_smiles.extend(harvested)
        return self._collect_fragments_from_smiles(source_smiles, context)

    def _collect_brics_fragments(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
        force_exploration: bool,
    ) -> list[Chem.Mol]:
        try:
            with _suppress_stderr():
                seed_frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
        except Exception as exc:
            logger.warning("BRICS decomposition failed for %s: %s", ctx.smiles, exc)
            if context.strict:
                raise
            return []
        seed_frags: list[Chem.Mol] = []
        for fs in seed_frag_smiles:
            frag_ctx = MoleculeContext.from_brics_fragment(fs)
            if frag_ctx is not None:
                seed_frags.append(frag_ctx.mol)

        pool_frags = self._load_brics_fragments(
            context, exploration_bias=force_exploration,
        )
        return seed_frags + pool_frags

    def _validate_brics_product(
        self,
        r_mol: Chem.Mol,
        context: StrategyContext,
        force_exploration: bool = False,
        _diagnostics: list[str] | None = None,
    ) -> str | None:
        if r_mol is None:
            if _diagnostics is not None:
                _diagnostics.append("BRICS: product is None")
            return None
        try:
            Chem.SanitizeMol(r_mol)
        except Exception:
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: invalid valence")
            return None
        s = Chem.MolToSmiles(r_mol)
        if not s:
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: empty SMILES")
            return None
        product_ctx = MoleculeContext(smiles=s, mol=r_mol)
        if not product_ctx.is_valid_electrolyte_mol():
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: invalid valence")
            return None
        if context.novelty_validator is not None and not context.novelty_validator.novelty_check(
            product_ctx, force_exploration=force_exploration,
        ):
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: novelty check")
            return None
        if not is_electrolyte_like(product_ctx):
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: not electrolyte-like")
            return None
        if _has_excessive_aliphatic_chain_fn(r_mol):
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: excessive aliphatic chain")
            return None
        if combined_grounding_score(r_mol) < MIN_GROUNDING_SCORE:
            if _diagnostics is not None:
                _diagnostics.append("BRICS: failed: grounding score too low")
            return None
        return s

    def _build_from_pairs(
        self,
        all_frags: list[Chem.Mol],
        valid_pairs: list[tuple[int, int]],
        context: StrategyContext,
        force_exploration: bool = False,
        _diagnostics: list[str] | None = None,
    ) -> list[str]:
        generated: list[str] = []
        n_pairs = len(valid_pairs)
        indices = context.rng.integers(0, n_pairs, size=min(150, n_pairs * 5))
        for idx in indices:
            try:
                i, j = valid_pairs[idx]
                with _suppress_stderr():
                    for r_mol in BRICS.BRICSBuild([all_frags[i], all_frags[j]]):
                        s = self._validate_brics_product(
                            r_mol, context, force_exploration=force_exploration,
                            _diagnostics=_diagnostics,
                        )
                        if s:
                            generated.append(s)
            except Exception as exc:
                logger.warning("BRICS build failed: %s", exc)
                if context.strict:
                    raise
        return generated

    def _brics_from_pool(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
        force_exploration: bool = False,
        _diagnostics: list[str] | None = None,
    ) -> list[str]:
        generated: list[str] = []

        all_frags = self._collect_brics_fragments(ctx, context, force_exploration)
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

        generated = self._build_from_pairs(
            all_frags, valid_pairs, context, force_exploration=force_exploration,
            _diagnostics=_diagnostics,
        )
        return list(set(generated))

    def _random_scaffold_replacement(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
    ) -> list[str]:
        if MurckoScaffold is None:
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

        scaffold_atoms = set()
        try:
            scaffold_match = mol.GetSubstructMatch(scaffold_mol)
            if scaffold_match:
                scaffold_atoms = set(scaffold_match)
        except Exception:
            pass

        if not scaffold_atoms:
            return []

        side_chains: list[str] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in scaffold_atoms:
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() in scaffold_atoms:
                        try:
                            mol_copy = Chem.RWMol(mol)
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

        n_scaffolds = len(self._SCAFFOLD_LIBRARY)
        n_to_try = min(3, n_scaffolds)
        scaffold_indices = context.rng.choice(
            n_scaffolds, size=n_to_try, replace=False,
        )

        results: list[str] = []
        for scaf_idx in scaffold_indices:
            new_scaffold_smi = self._SCAFFOLD_LIBRARY[int(scaf_idx)]
            new_scaffold = Chem.MolFromSmiles(new_scaffold_smi)
            if new_scaffold is None:
                continue

            for side_smi in side_chains[:3]:
                side_mol = Chem.MolFromSmiles(side_smi)
                if side_mol is None:
                    continue
                try:
                    from rdkit.Chem import rdmolops
                    combined = rdmolops.CombineMols(new_scaffold, side_mol)
                    combined = Chem.RWMol(combined)
                    combined_smi = Chem.MolToSmiles(combined)
                    if (
                        combined_smi
                        and combined_smi != ctx.smiles
                        and (combined_ctx := MoleculeContext.from_smiles(combined_smi)) is not None
                        and context.novelty_validator is not None
                        and context.novelty_validator.novelty_check(combined_ctx, force_exploration=False)
                    ):
                        results.append(combined_smi)
                except Exception:
                    continue

        return list(set(results))

    def mutate(
        self,
        ctx: MoleculeContext,
        context: StrategyContext,
        batch_size: int = 50,
        force_exploration: bool = False,
        diagnostics: list[str] | None = None,
    ) -> list[str]:
        candidates: set[str] = set()

        use_scaffold_hop = context.rng.random() < self._SCAFFOLD_HOP_PROBABILITY
        if use_scaffold_hop:
            scaffold_results = self._random_scaffold_replacement(ctx, context)
            candidates.update(scaffold_results)
            logger.info(
                "Scaffold hopping on %s: %d candidates",
                ctx.smiles, len(scaffold_results),
            )

        brics_results = self._brics_from_pool(
            ctx, context, force_exploration=force_exploration,
            _diagnostics=diagnostics,
        )
        candidates.update(brics_results)

        return list(candidates)


__all__ = [
    "MutationStrategy",
    "SmartsStrategy",
    "BricsStrategy",
    "StrategyContext",
]
