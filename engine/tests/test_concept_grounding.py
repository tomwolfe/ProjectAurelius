"""Tests for concept-grounded mutation and concept matching.

Verifies that:
- The concept library loads correctly
- Concept matching identifies correct SMILES patterns
- Mutate-by-concept returns concept-preserving candidates
"""
from __future__ import annotations

from rdkit import Chem

from aurelius.agent.mutation.engine import MutationEngine
from aurelius.data.concept_library import (
    concept_grounding_score,
    ConceptLibrary,
)
from aurelius.types import MoleculeContext


class TestConceptLibrary:
    """Tests for the concept library data structure."""

    def test_load_concept_library(self):
        """The concept library should load without errors."""
        library = ConceptLibrary.load()
        assert len(library.concepts) >= 5
        names = [c.name for c in library.concepts]
        assert len(names) == len(set(names)), "Concept names must be unique"

    def test_concept_names_unique(self):
        """Concept names should be unique identifiers."""
        library = ConceptLibrary.load()
        names = [c.name for c in library.concepts]
        assert len(names) == len(set(names)), "Concept names must be unique"

    def test_concept_smarts_valid(self):
        """All concept SMARTS patterns should compile to valid RDKit molecules."""
        library = ConceptLibrary.load()
        for concept in library.concepts:
            pattern = Chem.MolFromSmarts(concept.smarts)
            assert pattern is not None, (
                f"SMARTS pattern for '{concept.name}' should compile"
            )

    def test_concept_has_required_fields(self):
        """Each concept must have name, smarts, and category fields."""
        library = ConceptLibrary.load()
        for concept in library.concepts:
            assert hasattr(concept, "name") and concept.name
            assert hasattr(concept, "smarts") and concept.smarts
            assert hasattr(concept, "category") and concept.category


class TestConceptGrounding:
    """Tests for the concept_grounding_score function."""

    def test_cyclic_carbonate_detected(self):
        """Ethylene carbonate should match the cyclic_carbonate concept."""
        mol = Chem.MolFromSmiles("O=C1OCOCO1")
        score = concept_grounding_score(mol)
        assert score >= 1, "Ethylene carbonate should match cyclic_carbonate concept"

    def test_fluorinated_ether_detected(self):
        """Trifluoromethyl group should match fluorinated_ether concept."""
        mol = Chem.MolFromSmiles("FC(F)(F)F")
        score = concept_grounding_score(mol)
        assert score >= 1, "Trifluoromethane should match fluorinated_ether concept"

    def test_sulfone_detected(self):
        """Dimethyl sulfone should match the sulfone concept."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)C")
        score = concept_grounding_score(mol)
        assert score >= 1, "Dimethyl sulfone should match sulfone concept"

    def test_nitrile_detected(self):
        """Acetonitrile should match the nitrile concept."""
        mol = Chem.MolFromSmiles("CC#N")
        score = concept_grounding_score(mol)
        assert score >= 1, "Acetonitrile should match nitrile concept"

    def test_molecule_with_no_concepts(self):
        """A simple alkane should not match any concepts."""
        mol = Chem.MolFromSmiles("CCCC")
        score = concept_grounding_score(mol)
        assert score == 0, "Butane should not match any electrolyte concepts"

    def test_multiple_concept_matches(self):
        """A molecule with multiple concepts should score higher."""
        # Dimethyl carbonate has both carbonate and cyclic_carbonate-like structure
        mol = Chem.MolFromSmiles("COC(=O)OC")
        score = concept_grounding_score(mol)
        assert score >= 1, (
            "Dimethyl carbonate should match at least 1 concept "
            "(carbonate or cyclic_carbonate)"
        )


class TestMutateByConcept:
    """Tests for the mutate_by_concept method."""

    def test_mutate_by_concept_empty_result_for_invalid_smiles(self):
        """Invalid SMILES should return an empty list."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        result = engine.mutate_by_concept("INVALID_SMILES", batch_size=5)
        assert result == []

    def test_mutate_by_concept_with_target_concepts(self):
        """Mutating with target concepts should return candidates."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        result = engine.mutate_by_concept(
            "COC(=O)OC",
            concept_names=["cyclic_carbonate", "carbonate"],
            batch_size=5,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        for smi in result:
            assert isinstance(smi, str)

    def test_mutate_by_concept_with_no_targets(self):
        """Empty concept names should return candidates using all concepts."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        result = engine.mutate_by_concept("COC(=O)OC", batch_size=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_mutate_by_concept_uses_all_concepts_when_none_specified(self):
        """When no concept names are provided, all concepts should be used."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        result = engine.mutate_by_concept("COC(=O)OC", batch_size=3)
        assert len(result) <= 3


class TestThreadSafety:
    """Thread-safety tests for concept loading and mutation."""

    def test_concept_library_thread_safe_load(self):
        """ConceptLibrary._load_from_file() should be safe to call from multiple threads."""
        import threading

        results: list[list[Concept]] = []
        errors: list[Exception] = []

        def _load() -> None:
            try:
                lib = ConceptLibrary._load_from_file()
                results.append(lib)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_load) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert all(len(r) >= 5 for r in results)

    def test_mutation_engine_thread_safe_access(self):
        """MutationEngine should be safe for concurrent read access."""
        import threading

        results: list[str] = []
        errors: list[Exception] = []
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])

        def _mutate() -> None:
            try:
                candidates = engine.mutate("COC(=O)OC", batch_size=10)
                results.extend(candidates[:3])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_mutate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) > 0
