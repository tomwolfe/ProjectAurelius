"""Smoke test for the full DiscoveryLoop integration.

Runs a minimal 2-generation loop with a small batch to verify
the end-to-end pipeline works.
"""

from __future__ import annotations

import json
import os

import pytest

from aurelius.agent.loop import DiscoveryLoop
from aurelius.pipeline import AureliusPipeline
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.state import CheckpointManager


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

    engine = MutationEngine(seed_smiles=seed_smiles)
    checkpoint = CheckpointManager(path=str(tmp_path / "checkpoint.json"))

    loop = DiscoveryLoop(
        pipeline=pipeline,
        engine=engine,
        checkpoint=checkpoint,
        max_generations=2,
        batch_size=5,
        max_wall_time=120.0,
    )

    result = loop.execute()

    assert len(result["all_results"]) >= 1, "Should have at least one screening result"
    for r in result["all_results"]:
        assert 0.0 <= r.total_score <= 100.0, f"Score {r.total_score} out of [0, 100]"
