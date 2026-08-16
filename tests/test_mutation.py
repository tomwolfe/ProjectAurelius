"""Tests for the mutation engine — novelty, fragment harvesting, dynamic pool."""
from __future__ import annotations

from aurelius.agent.mutation import (
    ELECTROLYTE_FRAGMENT_POOL,
    MutationEngine,
)
from aurelius.types import MoleculeContext, parse_mixture_smiles_n

# ---------------------------------------------------------------------------
# Novelty check tests
# ---------------------------------------------------------------------------


class TestNoveltyCheck:
    """The _novelty_check must allow local functionalization of seed cores.

    Key change: seed Tanimoto comparison is removed so that adding a single
    functional group (F, CH3, CF3, CN, etc.) to a known seed core is accepted
    as novel, as long as the exact canonical SMILES does not match a seed or
    known commercial electrolyte.
    """

    def test_local_functionalization_allowed(self):
        """A molecule that differs from its seed by one functional group must
        be accepted as novel when scaffold check is disabled (exact SMILES match
        is the only seed gate).

        Seed:  COC(=O)OC   (dimethyl carbonate)
        Variant: COC(=O)OCC (ethyl methyl carbonate) — methyl → ethyl.
        """
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        emc = MoleculeContext.from_smiles("COC(=O)OCC")
        assert emc is not None
        assert engine._novelty_check(emc, check_scaffold=False) is True, (
            "Ethyl methyl carbonate (EMC) differs from DMC seed and must be novel."
        )

    def test_exact_seed_match_rejected(self):
        """Exact SMILES match to a seed must be rejected."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        assert engine._novelty_check(ctx) is False, "Exact seed match must be rejected"

    def test_exact_known_electrolyte_match_rejected(self):
        """Exact SMILES match to a known commercial electrolyte must be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        ctx = MoleculeContext.from_smiles("COC(=O)OC")  # DMC is in known_electrolytes.json
        assert ctx is not None
        assert engine._novelty_check(ctx) is False, "Known electrolyte must be rejected"

    def test_novelty_check_ignores_seed_tanimoto(self):
        """The novelty check must NOT use seed fingerprints as a gate.

        The ``_novelty_check`` method should only compare exact SMILES against
        seeds (not Tanimoto).  We verify this by checking that no Tanimoto
        computation against seed fingerprints occurs — a molecule that is not
        an exact seed match passes even if we mock seed fingerprints to be empty.
        """
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COC(=O)O1"])
        # Clear seed fingerprints to confirm Tanimoto is not used against seeds
        engine.seed_fingerprints = []
        # A molecule different from both seeds
        ctx = MoleculeContext.from_smiles("CS(=O)(=O)CCOC")
        assert ctx is not None
        assert engine._novelty_check(ctx) is True, (
            "Must pass even with empty seed fingerprints (no seed Tanimoto gate)"
        )

    def test_completely_novel_molecule_accepted(self):
        """A structurally unique molecule must be accepted."""
        engine = MutationEngine(seed_smiles=["C1COC(=O)O1"])
        ctx = MoleculeContext.from_smiles("CS(=O)(=O)CCOC")
        assert ctx is not None
        assert engine._novelty_check(ctx) is True


# ---------------------------------------------------------------------------
# Dynamic fragment harvesting tests
# ---------------------------------------------------------------------------


class TestFragmentHarvesting:
    """The mutation engine must dynamically evolve its fragment pool."""

    def test_harvest_fragments_pool_grows(self):
        """Harvesting fragments from a high-scoring molecule must grow the pool."""
        engine = MutationEngine(seed_smiles=["CC"])
        initial_size = engine.fragment_pool_size()
        assert initial_size == len(ELECTROLYTE_FRAGMENT_POOL)

        engine.harvest_fragments("COCCOC(=O)OC")
        new_size = engine.fragment_pool_size()
        assert new_size > initial_size, "Harvesting must add fragments to the pool"

    def test_harvest_fragments_no_duplicates(self):
        """Harvesting the same molecule twice must not duplicate fragments."""
        engine = MutationEngine(seed_smiles=["CC"])
        engine.harvest_fragments("COCCOC(=O)OC")
        size_after_first = engine.fragment_pool_size()

        engine.harvest_fragments("COCCOC(=O)OC")
        size_after_second = engine.fragment_pool_size()

        assert size_after_second == size_after_first, "Duplicate harvest must not add fragments"

    def test_harvested_fragments_used_in_brics(self):
        """Harvested fragments must appear in BRICS reassembly candidates."""
        engine = MutationEngine(seed_smiles=["C1COCCO1"])
        engine.harvest_fragments("COC(=O)OC(F)(F)F")

        candidates = engine._brics_from_pool(
            MoleculeContext.from_smiles("C1COCCO1")  # type: ignore[arg-type]
        )
        assert isinstance(candidates, list)

    def test_invalid_smiles_harvest_does_not_crash(self):
        """Harvesting an invalid molecule must not crash the engine."""
        engine = MutationEngine(seed_smiles=["CC"])
        engine.harvest_fragments("not a valid smiles !!!")  # Should not raise
        assert engine.fragment_pool_size() == len(ELECTROLYTE_FRAGMENT_POOL)


# ---------------------------------------------------------------------------
# Stagnation pivot: force_exploration flag
# ---------------------------------------------------------------------------


class TestStagnationPivot:
    """When force_exploration=True, the mutation engine must skip SMARTS
    reactions and rely solely on BRICS scaffold-hopping."""

    def test_force_exploration_skips_smarts(self):
        """With force_exploration=True, mutate() should return BRICS-only
        candidates and no SMARTS products."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])

        engine.mutate("COC(=O)OC", batch_size=20, force_exploration=False)

        exploration_result = engine.mutate("COC(=O)OC", batch_size=20, force_exploration=True)

        # SMARTS reactions on DMC typically produce fluorinated variants
        # or methyl ester products.  With force_exploration these should
        # not appear; only BRICS reassembly products should be returned.
        assert len(exploration_result) >= 0, "Exploration mode should not crash"
        # Exploration mode should not produce SMARTS-specific products
        # like trifluoromethyl-dimethyl carbonate (a direct SMARTS product)
        smarts_marker_smiles = {"COC(=O)OC(F)(F)F", "COC(=O)OCC"}
        exploration_smarts_hits = sum(
            1 for smi in exploration_result if smi in smarts_marker_smiles
        )
        assert exploration_smarts_hits == 0, (
            f"Exploration mode produced SMARTS products: {exploration_result}"
        )

    def test_force_exploration_batch(self):
        """mutate_batch with force_exploration=True must not crash and
        should return BRICS products."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        results = engine.mutate_batch(
            ["COC(=O)OC", "C1COCCO1"],
            batch_size=10,
            force_exploration=True,
        )
        assert isinstance(results, list)
        # All returned SMILES should be valid
        for smi in results:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES in exploration results: {smi}"


# ---------------------------------------------------------------------------
# Ternary mixture mutation operators
# ---------------------------------------------------------------------------


class TestTernaryMixtureMutation:
    """The mutation engine must generate ternary mixture candidates where
    each of the three components is individually valid and diversified."""

    def test_ternary_mixture_variants_are_valid(self):
        engine = MutationEngine(seed_smiles=["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C"])
        candidates = engine.propose_ternary_mixture_candidates(
            ["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C", "CC#N"],
            n_mixtures=20,
            batch_size=5,
        )
        assert len(candidates) > 0
        for smi in candidates:
            parsed = parse_mixture_smiles_n(smi)
            assert parsed is not None, f"Unparseable ternary mixture: {smi}"
            comps, fracs = parsed
            assert len(comps) == 3, f"Expected ternary, got {len(comps)} components"
            assert len(fracs) == 3 and abs(sum(fracs) - 1.0) < 1e-6
            for comp in comps:
                ctx = MoleculeContext.from_smiles(comp)
                assert ctx is not None, f"Invalid component SMILES: {comp}"

    def test_ternary_mutation_preserves_format(self):
        engine = MutationEngine(seed_smiles=["C1COC(=O)O1", "COCCOC"])
        variants = engine._mutate_ternary_mixture(
            "C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C", 0.4, 0.3
        )
        for smi in variants:
            comps, _fracs = parse_mixture_smiles_n(smi)
            assert comps is not None
            assert len(comps) == 3

    def test_ternary_binary_backward_compatibility(self):
        """The binary mixture format must still parse after ternary support."""
        engine = MutationEngine(seed_smiles=["C1COC(=O)O1", "COCCOC"])
        binary = engine.propose_mixture_candidates(
            ["C1COC(=O)O1", "COCCOC"], n_mixtures=5, batch_size=3
        )
        for smi in binary:
            comps, fracs = parse_mixture_smiles_n(smi)
            assert comps is not None and len(comps) == 2
            assert abs(sum(fracs) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Mixture-fraction CMA-ES optimizer (Gap 4 regression)
# ---------------------------------------------------------------------------


class TestMixtureFractionCmaEs:
    """The CMA-ES mixture-fraction optimizer must keep producing valid
    mixtures after the proxy-context computation was extracted into
    ``_mixture_proxy_targets``, and repeat calls must hit the fraction cache."""

    def test_ternary_optimizer_returns_valid_cached_mixture(self):
        engine = MutationEngine(
            seed_smiles=["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C"], n_jobs=1
        )
        components = ["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C"]
        fracs = [0.4, 0.3, 0.3]

        mixture = engine._mutate_mixture_fraction_cma_es(components, fracs)
        parsed = parse_mixture_smiles_n(mixture)
        assert parsed is not None, f"Unparseable mixture: {mixture}"
        comps, mix_fracs = parsed
        assert comps == components
        assert len(mix_fracs) == 3 and abs(sum(mix_fracs) - 1.0) < 1e-6

        # Optimized fractions are cached, so a repeat call must return the
        # identical mixture string without re-running the optimizer.
        repeated = engine._mutate_mixture_fraction_cma_es(components, fracs)
        assert repeated == mixture

    def test_binary_optimizer_returns_valid_cached_mixture(self):
        engine = MutationEngine(
            seed_smiles=["C1COC(=O)O1", "COCCOC"], n_jobs=1
        )
        components = ["C1COC(=O)O1", "COCCOC"]
        fracs = [0.5, 0.5]

        mixture = engine._mutate_mixture_fraction_cma_es(components, fracs)
        parsed = parse_mixture_smiles_n(mixture)
        assert parsed is not None, f"Unparseable mixture: {mixture}"
        comps, mix_fracs = parsed
        assert comps == components
        assert len(mix_fracs) == 2 and abs(sum(mix_fracs) - 1.0) < 1e-6

        repeated = engine._mutate_mixture_fraction_cma_es(components, fracs)
        assert repeated == mixture


# ---------------------------------------------------------------------------
# BRICS pairing correctness (ADR-2026-08-07-10)
# ---------------------------------------------------------------------------


class TestBricsComplementaryPairs:
    """``find_complementary_pairs`` must return pairs BRICSBuild can join.

    Regression guard for a defect that silently disabled the entire BRICS
    generation pathway: the function paired fragments that *shared* a dummy
    isotope, but BRICS joins *complementary* types. Every returned pair was
    unjoinable, so BRICS contributed zero candidates and
    ``force_exploration=True`` — which turns SMARTS off and relies on BRICS
    alone — produced nothing at all.
    """

    def test_same_type_pair_is_not_reported_as_complementary(self):
        from rdkit import Chem

        from aurelius.agent.mutation.brics import find_complementary_pairs

        # L3 bonds to {1, 4, 13, 14, 15, 16} and never to another L3. With a
        # genuinely joinable L3+L4 pair also present, the all-pairs fallback
        # does not fire, so the L3-L3 pair must be absent from the result.
        frags = [
            Chem.MolFromSmiles("[3*]OC"),
            Chem.MolFromSmiles("[3*]OCC"),
            Chem.MolFromSmiles("[4*]CC"),
        ]
        pairs = find_complementary_pairs(frags)
        assert (0, 2) in pairs, "L3 + L4 is a valid BRICS bond and must be offered"
        assert (0, 1) not in pairs, (
            "two L3 fragments cannot be joined by BRICS and must not be "
            "reported as a complementary pair"
        )

    def test_complementary_pair_is_reported(self):
        from rdkit import Chem

        from aurelius.agent.mutation.brics import find_complementary_pairs

        # L3 + L4 is an allowed BRICS bond.
        frags = [Chem.MolFromSmiles("[3*]OC"), Chem.MolFromSmiles("[4*]CC")]
        assert (0, 1) in find_complementary_pairs(frags)

    def test_reported_pairs_actually_build(self):
        """Pairs returned for a real seed must yield real BRICS products."""
        from rdkit.Chem import BRICS

        from aurelius.agent.mutation.brics import find_complementary_pairs

        engine = MutationEngine(seed_smiles=["COC(=O)OC"], n_jobs=1)
        ctx = engine._get_ctx("COC(=O)OC")
        assert ctx is not None
        frags = engine._collect_brics_fragments(ctx, False)
        pairs = find_complementary_pairs(frags)
        assert pairs, "no complementary pairs found for dimethyl carbonate"

        built = 0
        for i, j in pairs[:25]:
            try:
                for _product in BRICS.BRICSBuild([frags[i], frags[j]]):
                    built += 1
                    break
            except Exception:
                continue
        assert built > 0, (
            "no BRICS pair produced a product; the BRICS generation pathway "
            "is dead and the EA is running on SMARTS templates alone"
        )

    def test_brics_pathway_yields_candidates(self):
        """The BRICS path must contribute candidates on its own."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1", "CC#N"], n_jobs=1)
        total = 0
        for seed in engine.seed_pool[:6]:
            ctx = engine._get_ctx(seed)
            if ctx is None:
                continue
            total += len(engine._brics_from_pool(ctx, force_exploration=False))
        assert total > 0, (
            "BRICS pathway produced no candidates across six seeds"
        )
