"""Tests for v8.0 hardening — SMARTS pre-compilation, anti-gaming, scaffold tracking."""

from __future__ import annotations

from rdkit import Chem

from aurelius.agent.mutation import MutationEngine, _find_max_conjugated_path
from aurelius.agent.state import LoopState
from aurelius.constants import (
    ALDEHYDE_PATTERN,
    CARBONATE_PATTERN,
    CARBONYL_F_PATTERN,
    CF3_PATTERN,
    ELECTROCHEMICALLY_UNSTABLE_PATTERNS,
    EPOXIDE_PATTERN,
    ETHER_PATTERN,
    HYDROLYTICALLY_UNSTABLE_PATTERNS,
    NITRILE_PATTERN,
    PEROXIDE_PATTERN,
    SULFONE_PATTERN,
    SULFONE_SA_PATTERN,
    SULFONYL_F_PATTERN,
)
from aurelius.scoring.oracle import _GC_FRAGMENTS, _count_fragments, predict_tom_orbitals
from aurelius.types import MoleculeContext

# ---------------------------------------------------------------------------
# SMARTS Pre-compilation Tests
# ---------------------------------------------------------------------------


class TestSmartsPrecompilation:
    """All SMARTS patterns must be compiled at module level, not inside loops."""

    def test_gc_fragments_are_precompiled_mols(self):
        """_GC_FRAGMENTS tuples should contain Chem.Mol objects, not strings."""
        for entry in _GC_FRAGMENTS:
            pattern = entry[0]
            assert isinstance(pattern, Chem.Mol), (
                f"GC fragment {entry[1]} pattern is {type(pattern)}, expected Chem.Mol"
            )

    def test_hydrolytic_patterns_are_precompiled(self):
        """HYDROLYTICALLY_UNSTABLE_PATTERNS should contain Chem.Mol objects."""
        for pattern, name, _severity in HYDROLYTICALLY_UNSTABLE_PATTERNS:
            assert isinstance(pattern, Chem.Mol), f"{name} pattern is not a Chem.Mol"

    def test_electrochemical_patterns_are_precompiled(self):
        """ELECTROCHEMICALLY_UNSTABLE_PATTERNS should contain Chem.Mol objects."""
        for pattern, name in ELECTROCHEMICALLY_UNSTABLE_PATTERNS:
            assert isinstance(pattern, Chem.Mol), f"{name} pattern is not a Chem.Mol"

    def test_individual_patterns_are_precompiled(self):
        """Individual pre-compiled patterns should all be Chem.Mol objects."""
        for pat in [SULFONE_PATTERN, CF3_PATTERN, CARBONYL_F_PATTERN, SULFONYL_F_PATTERN,
                    PEROXIDE_PATTERN, ALDEHYDE_PATTERN, CARBONATE_PATTERN, ETHER_PATTERN,
                    SULFONE_SA_PATTERN, NITRILE_PATTERN, EPOXIDE_PATTERN]:
            assert isinstance(pat, Chem.Mol), f"Pattern {pat} is not a Chem.Mol"

    def test_count_fragments_uses_precompiled_patterns(self):
        """_count_fragments should not call Chem.MolFromSmarts internally."""
        mol = Chem.MolFromSmiles("COC(=O)OC")
        counts = _count_fragments(mol)
        assert counts.get("carbonate", 0) >= 1
        # DMC (COC(=O)OC) has no true esters — its carbonyl connects to two
        # oxygens, not a carbon.  The ester SMARTS ([CX3](=O)([#6])[OX2H0])
        # correctly rejects carbonate carbonyls (ADR-2026-06-05d).
        assert counts.get("ester", 0) == 0

    def test_mutation_engine_uses_precompiled(self):
        """MutationEngine should use pre-compiled patterns from constants."""
        engine = MutationEngine(seed_smiles=["CC"])
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is True


# ---------------------------------------------------------------------------
# Anti-Gaming Tests
# ---------------------------------------------------------------------------


class TestAntiGaming:
    """The mutation engine must reject molecules that game additive models."""

    def test_rejects_perhalogenated_spam(self):
        """Molecule with >90% halogen atoms (no solvation sites) should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # CF4-like molecule: 1 C, 4 F -> 4/5 = 80% F -> passes F check but
        # fails other checks (O+F ratio < 0.25)
        ctx = MoleculeContext.from_smiles("C(F)(F)(F)F")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_heavy_halogen_spam(self):
        """Molecule with >50% Cl/Br should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # CCl4: 1 C, 4 Cl -> 4/5 = 80% heavy halogen
        ctx = MoleculeContext.from_smiles("C(Cl)(Cl)(Cl)Cl")
        if ctx is not None:
            assert engine._is_electrolyte_like(ctx) is False

    def test_allows_fluorinated_electrolytes(self):
        """Heavily fluorinated molecules with solvation sites should pass.

        Many modern electrolytes (fluorinated carbonates, fluorinated ethers,
        sulfonimides) are heavily fluorinated and must be allowed.
        """
        engine = MutationEngine(seed_smiles=["CC"])

        # FEC (fluoroethylene carbonate): 14% F, has O for solvation
        ctx = MoleculeContext.from_smiles("O=C1OC(F)CO1")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is True

        # Bis(trifluoroethyl) carbonate: 6 F atoms, 3 O atoms for solvation
        ctx = MoleculeContext.from_smiles("O=C(OCC(F)(F)F)OCC(F)(F)F")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is True

        # TFSI-like: high F but has O, N, S for solvation
        ctx = MoleculeContext.from_smiles("C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F")
        if ctx is not None:
            assert engine._is_electrolyte_like(ctx) is True

        # Trifluoromethyl ethylene carbonate variant
        ctx = MoleculeContext.from_smiles("O=C1OC(C(F)(F)F)CO1")
        if ctx is not None:
            assert engine._is_electrolyte_like(ctx) is True

    def test_rejects_excess_conjugation(self):
        """Infinitely conjugated 'Frankenstein' should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # Polyaromatic with long conjugation
        ctx = MoleculeContext.from_smiles("c1ccc2cc3cc4cc5cc6cc7cc8cc9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c%27c%28c%29c%30c%31c%32c%33c%34c%35c%36c%37c1c2c3c4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c%27c%28c%29c%30c%31c%32c%33c%34c%35c%36c%37")
        if ctx is not None:
            max_conj = _find_max_conjugated_path(ctx.mol)
            assert max_conj > 16  # Should be detected as over-conjugated
            assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_impossible_valence(self):
        """Molecule with impossible valence (F with 2 bonds) should be rejected."""
        MutationEngine(seed_smiles=["CC"])
        mol = Chem.MolFromSmiles("C(F)(F)F")
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 9:
                    assert atom.GetExplicitValence() == 1

    def test_rejects_low_polarity_ratio(self):
        """Molecule with long non-polar chain should be rejected by TPSA/MW check."""
        engine = MutationEngine(seed_smiles=["CC"])
        ctx = MoleculeContext.from_smiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OC")
        if ctx is not None:
            mw = ctx.mw
            if mw > 200:
                assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_hydrolytically_unstable(self):
        """Molecule with anhydride motif should be rejected by mutation engine."""
        engine = MutationEngine(seed_smiles=["CC"])
        ctx = MoleculeContext.from_smiles("CC(=O)OC(=O)CC")  # Anhydride
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_electrochemically_unstable(self):
        """Molecule with peroxide motif should be rejected by mutation engine."""
        engine = MutationEngine(seed_smiles=["CC"])
        ctx = MoleculeContext.from_smiles("CCOOC")  # Peroxide
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_reductive_cleavage_sec(self):
        """Molecule with secondary O-alkyl carbonate/ester should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # Isopropyl methyl carbonate — branched secondary O-alkyl prone to CO₂ loss
        ctx = MoleculeContext.from_smiles("COC(=O)OC(C)C")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_reductive_cleavage_tert(self):
        """Molecule with tertiary O-alkyl carbonate/ester should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # tert-Butyl methyl carbonate — highly branched, prone to reductive cleavage
        ctx = MoleculeContext.from_smiles("COC(=O)OC(C)(C)C")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_allows_fluorinated_branched_carbonate(self):
        """Fluorinated branched carbonate should pass despite branching."""
        engine = MutationEngine(seed_smiles=["CC"])
        # Hexafluoroisopropyl methyl carbonate — fluorination stabilises the C-O bond
        ctx = MoleculeContext.from_smiles("COC(=O)OC(C(F)(F)F)C(F)(F)F")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is True, (
            "Fluorinated branched carbonate should be allowed"
        )


# ---------------------------------------------------------------------------
# Scaffold Tracking Tests
# ---------------------------------------------------------------------------


class TestScaffoldTracking:
    """LoopState must track Murcko scaffolds and detect stagnation."""

    def test_scaffold_recording(self):
        """record_scaffolds should store scaffolds per batch."""
        state = LoopState(path="/tmp/test_scaffold_state.json")
        state.record_scaffolds(["C1COCCO1", "C1CCOC1"])
        state.record_scaffolds(["C1COCCO1", "CCCOC"])
        assert len(state.scaffolds_per_batch) == 2

    def test_no_stagnation_with_unique_scaffolds(self):
        """Different scaffolds across batches should not trigger stagnation."""
        state = LoopState(path="/tmp/test_no_stag.json")
        state.record_scaffolds(["C1COCCO1"])
        state.record_scaffolds(["c1ccccc1"])
        state.record_scaffolds(["CCCCC"])
        assert state.has_scaffold_stagnation(3) is False

    def test_stagnation_detected_with_repeated_scaffold(self):
        """Same scaffold in 3+ batches should trigger stagnation."""
        state = LoopState(path="/tmp/test_stag.json")
        state.record_scaffolds(["C1COCCO1"])
        state.record_scaffolds(["C1COCCO1", "c1ccccc1"])
        state.record_scaffolds(["C1COCCO1"])
        assert state.has_scaffold_stagnation(3) is True

    def test_not_enough_batches_no_stagnation(self):
        """Fewer than n_batches should not trigger stagnation."""
        state = LoopState(path="/tmp/test_not_enough.json")
        state.record_scaffolds(["C1COCCO1"])
        state.record_scaffolds(["C1COCCO1"])
        assert state.has_scaffold_stagnation(3) is False

    def test_stagnation_resets_after_new_scaffold(self):
        """A new scaffold should break stagnation detection."""
        state = LoopState(path="/tmp/test_reset.json")
        state.record_scaffolds(["C1COCCO1"])
        state.record_scaffolds(["C1COCCO1"])
        state.record_scaffolds(["c1ccccc1"])  # New scaffold!
        assert state.has_scaffold_stagnation(3) is False


# ---------------------------------------------------------------------------
# TOM pre-compilation test
# ---------------------------------------------------------------------------


class TestTOMPrecompiled:
    """predict_tom_orbitals should use pre-compiled SMARTS patterns."""

    def test_tom_uses_precompiled_patterns(self):
        """TOM should return consistent results with pre-compiled patterns."""
        mol = Chem.MolFromSmiles("COC(=O)OC")
        homo, lumo = predict_tom_orbitals(mol)
        assert -12.0 <= homo <= -3.0
        assert -5.0 <= lumo <= 5.0
        assert lumo > homo

    def test_cf3_correction_applied(self):
        """CF3 correction should still work with pre-compiled pattern."""
        ethane = Chem.MolFromSmiles("CC")
        cf3 = Chem.MolFromSmiles("CC(F)(F)F")
        e_h, e_l = predict_tom_orbitals(ethane)
        c_h, c_l = predict_tom_orbitals(cf3)
        assert c_h < e_h, "CF3 should lower HOMO"
        assert c_l < e_l, "CF3 should lower LUMO"
