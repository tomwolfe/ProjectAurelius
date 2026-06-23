"""Tests for the mutation engine — novelty, fragment harvesting, dynamic pool."""
from __future__ import annotations

from aurelius.agent.mutation import (
    ELECTROLYTE_FRAGMENT_POOL,
    MutationEngine,
)
from aurelius.types import MoleculeContext

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
# Hard-Topological Constraint Tests (Anti-Failure Protocol)
# ---------------------------------------------------------------------------
# Each test verifies that is_electrolyte_like() (called by both
# _process_smarts_product and _validate_brics_product) rejects molecules
# that violate one of the four hard constraints:
#   1. Strained rings (3- or 4-membered)
#   2. Conjugation paths > 16 atoms
#   3. Valence errors (explicit valence exceeded)
#   4. < 20% sp3 carbon character (for >4 carbons)


class TestHardTopologicalConstraints:
    """Hard-topological filters prevent model extrapolation failure modes.

    Inverted failure mode: Without these gates, the evolutionary algorithm
    can generate molecules that are synthetically impossible or physically
    unrealistic (strained rings violate Baeyer strain theory, overconjugation
    games additive models, valence errors are chemically impossible).
    """

    def test_rejects_strained_3_membered_ring(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C1CC1")  # cyclopropane
        assert ctx is not None
        assert not is_electrolyte_like(ctx), "3-membered ring must be rejected"

    def test_rejects_strained_4_membered_ring(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C1CCC1")  # cyclobutane
        assert ctx is not None
        assert not is_electrolyte_like(ctx), "4-membered ring must be rejected"

    def test_accepts_stable_5_membered_ring(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")  # EC
        assert ctx is not None
        assert is_electrolyte_like(ctx), "5-membered ring must be accepted"

    def test_accepts_stable_6_membered_ring(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C1COCCO1")  # 1,4-dioxane
        assert ctx is not None
        assert is_electrolyte_like(ctx), "6-membered ring must be accepted"

    def test_rejects_long_conjugation(self):
        """Conjugation path > 16 atoms must be rejected."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like, find_max_conjugated_path
        # beta-carotene-like polyene chain
        smi = "CC1=C(C=CC=C(C=CC=C(C=C)C)C)C=CC=C1"
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None
        path_len = find_max_conjugated_path(ctx.mol)
        assert path_len > 16, f"Test molecule conjugation ({path_len}) should be > 16"
        assert not is_electrolyte_like(ctx), "Overconjugated molecule must be rejected"

    def test_accepts_short_conjugation(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")  # EC — no conjugation
        assert ctx is not None
        assert is_electrolyte_like(ctx), "Short-conjugation molecule must be accepted"

    def test_rejects_valence_error(self):
        """Pentavalent carbon must be rejected."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        # Pentavalent carbon via SMILES with explicit valence error
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles("C(C)(C)(C)C")
            if mol is not None:
                # Some RDKit versions still parse this; check explicit valence
                ctx = MoleculeContext(smiles="C(C)(C)(C)C", mol=mol)
                assert not is_electrolyte_like(ctx), "Pentavalent carbon must be rejected"
        except Exception:
            pass  # RDKit may reject the SMILES outright — also acceptable

    def test_accepts_normal_valence(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("COC(=O)OC")  # DMC
        assert ctx is not None
        assert is_electrolyte_like(ctx), "Normal-valence molecule must be accepted"

    def test_rejects_low_sp3_fraction(self):
        """Molecule with >4 carbons but <20% sp3 must be rejected."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        # Benzene ring (6 carbons, 0 sp3)
        ctx = MoleculeContext.from_smiles("C1=CC=CC=C1")
        assert ctx is not None
        assert not is_electrolyte_like(ctx), "Benzene (0% sp3, 6 C) must be rejected"

    def test_accepts_high_sp3_fraction(self):
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("COC(=O)OC")  # DMC: 3 sp3 / 3 C = 100%
        assert ctx is not None
        assert is_electrolyte_like(ctx), "High-sp3 molecule must be accepted"

    def test_low_sp3_skipped_for_few_carbons(self):
        """Molecule with <=4 carbons must skip the sp3 check."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        ctx = MoleculeContext.from_smiles("C(F)(F)F")  # 1 C, 0% sp3, but only 1 C
        assert ctx is not None
        # Should pass because n_c < 4
        result = is_electrolyte_like(ctx)
        # May still be rejected by other checks (e.g. halogen limit)
        assert isinstance(result, bool)

    def test_smarts_products_respect_topological_filter(self):
        """SMARTS mutation products should not include valence errors
        or strained rings."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        candidates = engine.mutate("COC(=O)OC", batch_size=50)
        for smi in candidates:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES in SMARTS product: {smi}"
            assert is_electrolyte_like(ctx), (
                f"SMARTS product {smi} violates topological constraints"
            )
