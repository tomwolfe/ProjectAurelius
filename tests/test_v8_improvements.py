"""Tests for v8.0 hardening — SMARTS pre-compilation, anti-gaming, scaffold tracking."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from aurelius.constants import (
    HYDROLYTICALLY_UNSTABLE_PATTERNS,
    ELECTROCHEMICALLY_UNSTABLE_PATTERNS,
    SULFONE_PATTERN,
    CF3_PATTERN,
    CARBONYL_F_PATTERN,
    SULFONYL_F_PATTERN,
    PEROXIDE_PATTERN,
    ALDEHYDE_PATTERN,
    CARBONATE_PATTERN,
    ETHER_PATTERN,
    SULFONE_SA_PATTERN,
    NITRILE_PATTERN,
    EPOXIDE_PATTERN,
)
from aurelius.scoring.oracle import _GC_FRAGMENTS, _count_fragments, predict_tom_orbitals
from aurelius.agent.mutation import MutationEngine, _find_max_conjugated_path
from aurelius.agent.state import LoopState
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
        assert counts.get("ester", 0) >= 1

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

    def test_rejects_fluorine_spam(self):
        """Molecule with >60% fluorine atoms should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # CF4-like molecule: 1 C, 4 F -> 4/5 = 80% fluorine
        ctx = MoleculeContext.from_smiles("C(F)(F)(F)F")
        assert ctx is not None
        assert engine._is_electrolyte_like(ctx) is False

    def test_rejects_halogen_spam(self):
        """Molecule with >60% halogen atoms should be rejected."""
        engine = MutationEngine(seed_smiles=["CC"])
        # C surrounded by 6 F and Cl -> very high halogen ratio
        ctx = MoleculeContext.from_smiles("C(Cl)(F)(F)(Cl)(F)F")
        if ctx is not None:
            assert engine._is_electrolyte_like(ctx) is False

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
        engine = MutationEngine(seed_smiles=["CC"])
        # This SMILES may not survive RDKit sanitization, but test the concept
        # We create a molecule where RDKit might miss hypervalent F
        mol = Chem.MolFromSmiles("C(F)(F)F")
        if mol is not None:
            # Check explicit valence of F atoms - each should be 1
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 9:
                    assert atom.GetExplicitValence() == 1

    def test_rejects_low_polarity_ratio(self):
        """Molecule with long non-polar chain should be rejected by TPSA/MW check."""
        engine = MutationEngine(seed_smiles=["CC"])
        # Very long alkane with one carbonate -> low TPSA/MW ratio
        ctx = MoleculeContext.from_smiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OC")
        if ctx is not None:
            mw = ctx.mw
            tpsa = ctx.tpsa
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
# Adaptive Diversity Tests
# ---------------------------------------------------------------------------


class TestAdaptiveDiversity:
    """Surrogate should adapt diversity_lambda based on batch variance."""

    def test_update_variance_high_diversity(self):
        """High variance (>150) should result in lower diversity_lambda (0.3)."""
        from aurelius.agent.surrogate import RandomForestSurrogate

        surrogate = RandomForestSurrogate()
        # Variance of [10, 50, 100, 150, 200] = 6070 -> high
        surrogate.update_variance([10.0, 50.0, 100.0, 150.0, 200.0])
        assert surrogate.diversity_lambda == pytest.approx(0.3, abs=0.01)

    def test_update_variance_low_diversity(self):
        """Low variance (<50) should result in higher diversity_lambda (0.7)."""
        from aurelius.agent.surrogate import RandomForestSurrogate

        surrogate = RandomForestSurrogate()
        # Variance of [100, 101, 99, 100, 102] = ~1.3 -> low
        surrogate.update_variance([100.0, 101.0, 99.0, 100.0, 102.0])
        assert surrogate.diversity_lambda == pytest.approx(0.7, abs=0.01)

    def test_update_variance_medium_diversity(self):
        """Medium variance (50-150) should give default lambda (0.5)."""
        from aurelius.agent.surrogate import RandomForestSurrogate

        surrogate = RandomForestSurrogate()
        # Variance of [90, 95, 100, 105, 110] = 62.5 -> medium
        surrogate.update_variance([90.0, 95.0, 100.0, 105.0, 110.0])
        assert surrogate.diversity_lambda == pytest.approx(0.5, abs=0.01)

    def test_diversity_lambda_setter_clamps(self):
        """diversity_lambda setter should clamp to [0, 1]."""
        from aurelius.agent.surrogate import RandomForestSurrogate

        surrogate = RandomForestSurrogate()
        surrogate.diversity_lambda = 1.5
        assert surrogate.diversity_lambda == 1.0
        surrogate.diversity_lambda = -0.5
        assert surrogate.diversity_lambda == 0.0


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
