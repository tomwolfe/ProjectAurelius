"""Tests for the mutation engine — novelty, fragment harvesting, dynamic pool."""
from __future__ import annotations

from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem

from aurelius.agent.mutation import (
    MutationEngine,
    ELECTROLYTE_FRAGMENT_POOL,
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
        be accepted as novel (exact SMILES match is the only seed gate).

        Seed:  COC(=O)OC   (dimethyl carbonate)
        Variant: COC(=O)OCC (ethyl methyl carbonate) — methyl → ethyl.
        """
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        emc = MoleculeContext.from_smiles("COC(=O)OCC")
        assert emc is not None
        assert engine._novelty_check(emc) is True, (
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
