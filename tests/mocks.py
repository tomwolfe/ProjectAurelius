"""Mock objects for testing and demonstration purposes.

Provides deterministic pseudo-DFT energy predictions based on
molecule SMILES, with caching and batch query support.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class MockDFTOracle:
    """Mock DFT oracle for testing and demonstration purposes.

    Provides deterministic pseudo-DFT energy predictions based on
    molecule SMILES, with caching and batch query support.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the mock oracle.

        Args:
            seed: Random seed for deterministic output.
        """
        self._cache: dict[str, float] = {}
        self._dataset: list[dict[str, Any]] = []
        self._rng = np.random.RandomState(seed)

    def query(self, smiles: str) -> float:
        """Query the oracle for a molecule's predicted activation energy.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Predicted activation energy in eV.
        """
        if smiles in self._cache:
            return self._cache[smiles]

        seed = hash(smiles) % 10000
        energy = 0.5 + (seed % 500) / 1000.0
        energy = float(np.clip(energy, 0.45, 0.95))
        self._cache[smiles] = energy
        return energy

    def query_batch(self, smiles_list: list[str]) -> list[float]:
        """Query multiple molecules at once.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of predicted activation energies.
        """
        return [self.query(s) for s in smiles_list]

    def append_dataset(self, smiles_list: list[str], energies: list[float]) -> None:
        """Append data to the training dataset.

        Args:
            smiles_list: List of SMILES strings.
            energies: List of activation energies in eV.
        """
        for smi, e in zip(smiles_list, energies, strict=False):
            self._dataset.append({"smiles": smi, "energy": e})
            self._cache[smi] = e

    def append_to_dataset(self, smiles_list: list[str], energies: list[float]) -> list[dict[str, Any]]:
        """Alias for append_dataset for backward compatibility.

        Args:
            smiles_list: List of SMILES strings.
            energies: List of activation energies in eV.

        Returns:
            List of entries added to the dataset.
        """
        entries: list[dict[str, Any]] = []
        for smi, e in zip(smiles_list, energies, strict=False):
            entry = {"smiles": smi, "ec_reduction": e}
            self._dataset.append(entry)  # type: ignore[arg-type]
            self._cache[smi] = e
            entries.append(entry)
        return entries

    def clear_cache(self) -> int:
        """Clear all cached query results.

        Returns:
            Number of entries cleared from cache.
        """
        count = len(self._cache)
        self._cache.clear()
        return count
