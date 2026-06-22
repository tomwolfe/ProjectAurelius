"""Tests for memory usage and profiling utilities."""
from __future__ import annotations

import psutil
from rdkit import Chem

from aurelius.agent.mutation.novelty import _heteroatom_profile


def test_memory_basic_usage() -> None:
    """psutil can report process memory, and a minimal molecule parse
    does not leak an unreasonable amount of memory."""
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    for _ in range(100):
        Chem.MolFromSmiles("COC(=O)OC")
    mem_after = proc.memory_info().rss
    increase = mem_after - mem_before
    # 100 RDKit parses should not increase RSS by more than 50 MB
    assert increase < 50_000_000, f"Memory increased by {increase / 1_000_000:.1f} MB"


def test_profiler_heteroatom() -> None:
    """Heteroatom profile correctly identifies heteroatoms in a molecule."""
    mol = Chem.MolFromSmiles("COC(=O)OC")
    assert mol is not None
    profile = _heteroatom_profile(mol)
    assert profile[8] == 3  # three oxygens
