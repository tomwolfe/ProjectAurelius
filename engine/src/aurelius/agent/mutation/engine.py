"""MutationEngine — multi-strategy molecule mutation engine.

Generates candidate molecules from seed SMILES using pluggable strategies
(defaulting to SMARTS + BRICS for backward compatibility).

Seeds are stored as MoleculeContext objects to enforce single-point parsing.

Delegates novelty checking to ``NoveltyValidator``, fragment harvesting
to ``FragmentHarvester``, and mutation logic to ``MutationStrategy``
implementations.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Generator

import numpy as np
from rdkit import Chem, rdBase

from aurelius.agent.mutation.base import (
    BricsStrategy,
    MutationStrategy,
    SmartsStrategy,
    StrategyContext,
)
from aurelius.agent.mutation.harvester import FragmentHarvester
from aurelius.agent.mutation.novelty import NoveltyValidator
from aurelius.cache.lru import LRUCache
from aurelius.types import MoleculeContext, format_mixture_smiles
from aurelius.utils.chem_utils import _deserialize_fp

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None  # type: ignore[assignment]

# Suppress RDKit C++ stderr warnings during mutation hot loops.
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.debug")

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_stderr() -> Generator[None, None, None]:
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


class MutationEngine:
    """Multi-strategy molecule mutation engine for battery electrolytes.

    Accepts an optional list of ``MutationStrategy`` instances. If none
    are provided, defaults to ``SmartsStrategy`` and ``BricsStrategy``
    for backward compatibility.

    Args:
        seed_smiles: Initial seed molecule SMILES. If None, loads tier0 seeds.
        known_fps_hex: Hex-encoded fingerprints of known electrolyte molecules.
        adaptive_bias: Re-order SMARTS reactions by historical success rate.
        strict: Re-raise exceptions instead of catching them.
        strategies: Pluggable strategy instances. If None, defaults to the
            original two (SMARTS then BRICS).
    """

    def __init__(
        self,
        seed_smiles: list[str] | None = None,
        known_fps_hex: list[str] | None = None,
        adaptive_bias: bool = True,
        strict: bool = False,
        strategies: list[MutationStrategy] | None = None,
    ) -> None:
        self.strict = strict
        self.seed_pool, self.seed_contexts, self.seed_fingerprints = self._init_seeds(
            seed_smiles
        )

        self._commercial_fps: list[Any] = []
        for h in known_fps_hex or []:
            try:
                self._commercial_fps.append(_deserialize_fp(h))
            except Exception:
                continue

        self._seed_smiles, self._seed_scaffolds = self._init_smiles_and_scaffolds()

        self._ctx_cache: LRUCache[MoleculeContext | None] = LRUCache(maxsize=4096)
        self._known_smiles: set[str] = set()
        self._load_known_electrolytes()

        self._generated_smiles: set[str] = set()
        self._rng = np.random.default_rng(42)
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

        # Build shared context for strategies
        self._strategy_context = StrategyContext(
            ctx_cache=self._ctx_cache,
            novelty_validator=self._novelty_validator,
            fragment_harvester=self._fragment_harvester,
            rng=self._rng,
            adaptive_bias=self._adaptive_bias,
            reaction_scores=self._reaction_scores,
            product_to_reaction=self._product_to_reaction,
            strict=self.strict,
            seed_fingerprints=self.seed_fingerprints,
        )

        if strategies is not None:
            self._strategies = list(strategies)
        else:
            self._strategies = [SmartsStrategy(), BricsStrategy()]

    @staticmethod
    def _init_seeds(
        seed_smiles: list[str] | None,
    ) -> tuple[list[str], list[MoleculeContext], list[Any]]:
        from pathlib import Path

        if seed_smiles is None:
            import json

            json_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "tier0_seed_smiles.json"
            )
            with open(json_path) as f:
                seed_smiles = json.load(f)
        pool = list(set(seed_smiles))
        contexts: list[MoleculeContext] = []
        fps: list[Any] = []
        for smi in pool:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                contexts.append(ctx)
                fps.append(ctx.get_ecfp4())
        return pool, contexts, fps

    def _init_smiles_and_scaffolds(self) -> tuple[set[str], set[str]]:
        smiles_set: set[str] = set()
        scaffold_set: set[str] = set()
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

    def _get_ctx(self, smiles: str) -> MoleculeContext | None:
        if smiles not in self._ctx_cache:
            ctx = MoleculeContext.from_smiles(smiles)
            self._ctx_cache[smiles] = ctx
        return self._ctx_cache.get(smiles)

    def _load_known_electrolytes(self) -> None:
        import json as _json
        import os as _os

        json_path = _os.path.join(
            _os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "known_electrolytes.json",
        )
        try:
            with open(json_path) as f:
                smiles_list = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return

        existing_smis: set[str] = set()
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

    def _is_trivial_alkyl_extension(
        self, ctx: MoleculeContext, min_extra_carbons: int = 2
    ) -> bool:
        return self._novelty_validator.is_trivial_alkyl_extension(
            ctx, min_extra_carbons=min_extra_carbons
        )

    def _novelty_check(
        self,
        ctx: MoleculeContext,
        check_scaffold: bool = True,
        force_exploration: bool = False,
    ) -> bool:
        return self._novelty_validator.novelty_check(
            ctx, check_scaffold=check_scaffold, force_exploration=force_exploration
        )

    # ------------------------------------------------------------------
    # Fragment Harvesting — delegated to FragmentHarvester
    # ------------------------------------------------------------------

    def fragment_pool_size(self) -> int:
        return self._fragment_harvester.fragment_pool_size()

    def harvest_fragments(self, smiles: str, score: float = 65.0) -> None:
        self._fragment_harvester.harvest_fragments(smiles, score=score)

    # ------------------------------------------------------------------
    # Adaptive bias: record reaction success for re-ordering
    # ------------------------------------------------------------------

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
    # Public mutation API
    # ------------------------------------------------------------------

    def mutate(
        self,
        smiles: str,
        batch_size: int = 50,
        force_exploration: bool = False,
    ) -> list[str]:
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return []

        candidates: set[str] = set()
        for strategy in self._strategies:
            results = strategy.mutate(
                ctx, self._strategy_context, batch_size, force_exploration
            )
            candidates.update(results)
            if len(candidates) >= batch_size:
                break

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        logger.info(
            "Mutation of %s: %d candidates [force_exploration=%s]",
            smiles,
            len(result_list),
            force_exploration,
        )
        return result_list

    def mutate_batch(
        self,
        batch_smiles: list[str],
        batch_size: int = 50,
        force_exploration: bool = False,
    ) -> list[str]:
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
        ctx_a = MoleculeContext.from_smiles(smi_a)
        ctx_b = MoleculeContext.from_smiles(smi_b)
        if ctx_a is None or ctx_b is None:
            return 1.0
        from rdkit.DataStructs import TanimotoSimilarity

        return TanimotoSimilarity(ctx_a.get_ecfp4(), ctx_b.get_ecfp4())

    def _mutate_mixture(
        self, smi_a: str, smi_b: str, frac: float, batch_size: int = 10
    ) -> list[str]:
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
                mix_variants = self._mutate_mixture(
                    smi_a, smi_b, frac, batch_size=batch_size
                )
                candidates.extend(mix_variants)
        unique = list(dict.fromkeys(candidates))
        if len(unique) > n_mixtures:
            idx = self._rng.choice(len(unique), size=n_mixtures, replace=False)
            unique = [unique[i] for i in idx]
        return unique

    # ------------------------------------------------------------------
    # Concept-Grounded Mutation
    # ------------------------------------------------------------------

    def mutate_by_concept(
        self,
        smiles: str,
        concept_names: list[str] | None = None,
        batch_size: int = 50,
    ) -> list[str]:
        """Mutate a seed SMILES while biasing toward specified electrolyte concepts.

        The mutation engine applies SMARTS and BRICS strategies, but this method
        post-filters and re-weights candidates to prefer molecules that preserve
        or introduce the named concepts (e.g. ``"cyclic_carbonate"`` or
        ``"fluorinated_ether"``).

        Args:
            smiles: Seed SMILES string to mutate.
            concept_names: List of concept names from the concept library.
                If None, all concepts in the library are used.
            batch_size: Maximum number of candidate SMILES to return.

        Returns:
            Up to *batch_size* candidate SMILES strings, biased toward
            molecules that match the specified concepts.
        """
        ctx = self._get_ctx(smiles)
        if ctx is None:
            return []

        from importlib import resources

        package_dir = resources.files("aurelius.data")
        concept_path = package_dir / "concept_library.json"

        try:
            with concept_path.open("r") as fh:
                import json
                library = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Concept library not found (%s); returning empty.", exc)
            return []

        concepts = library.get("concepts", [])
        if not concept_names:
            targets = [c["name"] for c in concepts]
        else:
            concept_map = {c["name"]: c for c in concepts}
            targets = [n for n in concept_names if n in concept_map]

        if not targets:
            return []

        candidates: dict[str, float] = {}
        for strategy in self._strategies:
            results = strategy.mutate(
                ctx, self._strategy_context, batch_size, force_exploration=False,
            )
            for smi in results:
                if smi not in candidates:
                    candidates[smi] = 0.0

        scored: list[tuple[str, float]] = []
        for smi, _base in candidates.items():
            match_ctx = MoleculeContext.from_smiles(smi)
            if match_ctx is None:
                continue
            score = 0.0
            for target in targets:
                for concept in concepts:
                    if concept["name"] != target:
                        continue
                    pattern = Chem.MolFromSmarts(concept["smarts"])
                    if pattern is not None and match_ctx.mol.HasSubstructMatch(pattern):
                        score += 1.0
                        break
            if score > 0:
                scored.append((smi, score))

        if not scored:
            for smi in candidates:
                scored.append((smi, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        n = min(batch_size, len(scored))
        if n == 0:
            return []
        indices = self._rng.choice(len(scored), size=n, replace=False)
        return [scored[i][0] for i in indices]
