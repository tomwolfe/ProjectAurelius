"""Smoke test for the full DiscoveryLoop integration.

Runs a minimal 2-generation loop with a small batch to verify
the end-to-end pipeline works.
"""

from __future__ import annotations

import json
import os

import pytest

from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.state import LoopState
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext

pytestmark = pytest.mark.slow


def _load_seed_smiles() -> list[str]:
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data", "tier0_seed_smiles.json"
    )
    with open(path) as f:
        return json.load(f)


def test_discovery_loop_smoke(tmp_path) -> None:
    """Run a 2-generation discovery loop and verify basic output."""
    seed_smiles = _load_seed_smiles()

    pipeline = AureliusPipeline()
    pipeline.initialize()

    assert pipeline._oracle is not None, "Oracle must be initialised"
    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC(=O)O1")
    assert ctx is not None
    result = pipeline._oracle.evaluate(ctx)
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert "gap_eV" in result
    assert "domain_applicable" in result

    engine = MutationEngine(seed_smiles=seed_smiles)
    state = LoopState(path=str(tmp_path / "state.json"))

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        state=state,
        max_generations=2,
        batch_size=5,
        max_wall_time=120.0,
    )

    result = loop.execute()

    assert len(result["all_results"]) >= 1, "Should have at least one screening result"
    for r in result["all_results"]:
        assert 0.0 <= r.total_score <= 100.0, f"Score {r.total_score} out of [0, 100]"

    assert loop.total_screened >= 1
    assert isinstance(result["discoveries"], list)
