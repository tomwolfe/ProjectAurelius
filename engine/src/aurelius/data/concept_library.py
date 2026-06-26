"""Concept library for concept-grounded mutation and scoring.

Provides:
- ConceptLibrary: A class to load and query the concept library JSON file
- concept_grounding_score: Computes how many distinct concepts a molecule matches
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rdkit import Chem


@dataclass(frozen=True)
class Concept:
    """A single electrolyte concept with its SMARTS pattern.

    Attributes:
        name: Unique concept name (e.g. "cyclic_carbonate").
        label: Human-readable label (e.g. "Cyclic Carbonate").
        smarts: SMARTS pattern for the concept.
        description: Human-readable description of the concept.
        category: High-level category grouping (e.g. "ring", "halogenated").
    """

    name: str
    label: str
    smarts: str
    description: str
    category: str


@lru_cache(maxsize=1)
def _load_concepts() -> list[Concept]:
    """Load and cache the concept library from the JSON file.

    Returns an empty list if the library file cannot be loaded.
    """
    from importlib import resources

    package_dir = resources.files("aurelius.data")
    concept_path = package_dir / "concept_library.json"

    try:
        with concept_path.open("r") as fh:
            import json
            library = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    concepts: list[Concept] = []
    for entry in library.get("concepts", []):
        concepts.append(Concept(
            name=entry["name"],
            label=entry["label"],
            smarts=entry["smarts"],
            description=entry.get("description", ""),
            category=entry.get("category", "unknown"),
        ))
    return concepts


@lru_cache(maxsize=1)
def _get_concept_library() -> list[Concept]:
    """Return the cached list of concepts from the library."""
    return _load_concepts()


class ConceptLibrary:
    """In-memory concept library loaded from the JSON file.

    This class provides a programmatic interface to the concept library,
    allowing filtering by category, name lookup, and validation.
    """

    def __init__(self, concepts: list[Concept]) -> None:
        self.concepts = concepts
        self._name_index: dict[str, Concept] = {c.name: c for c in concepts}
        self._category_index: dict[str, list[Concept]] = {}
        for c in concepts:
            self._category_index.setdefault(c.category, []).append(c)

    @classmethod
    def load(cls) -> ConceptLibrary:
        """Load the concept library from the bundled JSON file.

        Returns:
            ConceptLibrary instance with all concepts loaded.
        """
        concepts = _get_concept_library()
        return cls(concepts)

    def get_by_name(self, name: str) -> Concept | None:
        """Get a concept by its unique name.

        Args:
            name: The concept name (e.g. "cyclic_carbonate").

        Returns:
            The Concept object, or None if not found.
        """
        return self._name_index.get(name)

    def get_by_category(self, category: str) -> list[Concept]:
        """Get all concepts belonging to a category.

        Args:
            category: Category name (e.g. "ring", "halogenated").

        Returns:
            List of Concept objects in the category.
        """
        return self._category_index.get(category, [])

    def match_patterns(self, mol: Chem.Mol) -> list[Concept]:
        """Find all concepts whose SMARTS pattern matches the molecule.

        Args:
            mol: RDKit Mol object to check against concept patterns.

        Returns:
            List of matching Concept objects (may be empty).
        """
        matches: list[Concept] = []
        for concept in self.concepts:
            pattern = Chem.MolFromSmarts(concept.smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                matches.append(concept)
        return matches

    @staticmethod
    def _load_from_file() -> list[Concept]:
        """Static alias for loading the concept library from file.

        Returns:
            List of Concept objects.
        """
        return _get_concept_library()


def concept_grounding_score(mol: Chem.Mol) -> int:
    """Count how many distinct concepts from the concept library a molecule matches.

    A concept is considered matched if the molecule contains the concept's SMARTS
    pattern as a substructure. Returns the number of distinct concepts matched
    (0-based). This is a coarse-grained grounding score that supplements the
    BRICS-based grounding check by rewarding concept retention during mutation.

    Args:
        mol: RDKit Mol object to evaluate.

    Returns:
        Number of distinct concepts matched (0 = no concepts matched).

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("O=C1OCOCO1")
        >>> concept_grounding_score(mol)
        2
    """
    library = ConceptLibrary.load()
    matches = 0
    for concept in library.concepts:
        pattern = Chem.MolFromSmarts(concept.smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matches += 1
    return matches
