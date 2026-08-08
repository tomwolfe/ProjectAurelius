"""Test active learning escalation in Project Aurelius v11.0.

Verifies that when quantum_confidence == "tom_low" AND conformal_confidence <
active_learning_threshold, xTB evaluation is automatically triggered.

Physical justification: Active learning ensures that low-confidence TOM
predictions are immediately re-evaluated with higher-accuracy xTB,
maximizing information gain per compute dollar and preventing the EA
from exploiting TOM's blind spots.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aurelius.agent.loop import AgentConfig
from aurelius.agent.mutation import MutationEngine


def test_active_learning_escalation():
    """Test that xTB is called when confidence is low (tom_low)."""
    with tempfile.TemporaryDirectory():
        AgentConfig(
            max_generations=10,
            batch_size=5,
            use_nsga2=False,
            active_learning_threshold=0.7,
        )

        engine = MutationEngine(seed_smiles=["COC(=O)OC"])

        mock_ctx = MagicMock()
        mock_ctx.smiles = "CCOC(=O)OCC"
        mock_ctx.mol = MagicMock()

        from aurelius.agent.loop import DiscoveryLoop

        loop = DiscoveryLoop(
            pipeline=MagicMock(),
            engine=engine,
            state=MagicMock(),
            max_generations=10,
            batch_size=5,
            use_nsga2=False,
            active_learning_threshold=0.7,
        )

        mock_feedback_controller = MagicMock()
        loop.feedback_controller = mock_feedback_controller

        with patch.object(loop, '_evaluate_with_xtb', return_value={
            "tier2": {"quantum_confidence": "xtb", "homo_eV": -6.8, "lumo_eV": -0.9},
            "score": {
                "total_score": 85.0,
                "sub_scores": {"confidence": 0.8}
            },
            "conformal_confidence": 0.8
        }) as mock_xtb:
            result = loop._maybe_escalate(
                smi="CCOC(=O)OCC",
                ctx=mock_ctx,
                total_score=80.0,
                conformal_conf=0.5,
                score_data={},
                sub_scores={},
                t2={"quantum_confidence": "tom_low", "homo_eV": -7.0, "lumo_eV": -1.0}
            )

        mock_feedback_controller.log_active_learning_trigger.assert_called_once()
        mock_xtb.assert_called_once()

        updated_total_score, updated_conf, updated_score_data, updated_sub_scores, updated_t2 = result
        assert updated_total_score == 85.0
        assert updated_conf == 0.8


def test_no_escalation_when_confidence_high():
    """Test that xTB is NOT called when confidence is high."""
    from aurelius.agent.loop import DiscoveryLoop

    loop = DiscoveryLoop(
        pipeline=MagicMock(),
        engine=MagicMock(),
        state=MagicMock(),
        max_generations=10,
        batch_size=5,
        use_nsga2=False,
        active_learning_threshold=0.7,
    )

    mock_feedback_controller = MagicMock()
    loop.feedback_controller = mock_feedback_controller

    with patch.object(loop, '_evaluate_with_xtb') as mock_xtb:
        result = loop._maybe_escalate(
            smi="CCOC(=O)OCC",
            ctx=MagicMock(),
            total_score=80.0,
            conformal_conf=0.9,
            score_data={},
            sub_scores={},
            t2={"quantum_confidence": "xtb", "homo_eV": -7.0, "lumo_eV": -1.0}
        )

        mock_feedback_controller.log_active_learning_trigger.assert_not_called()
        mock_xtb.assert_not_called()

        updated_total_score, updated_conf, updated_score_data, updated_sub_scores, updated_t2 = result
        assert updated_total_score == 80.0
        assert updated_conf == 0.9


def test_no_escalation_when_tom_not_low():
    """Test that xTB is NOT called when TOM confidence is not tom_low."""
    from aurelius.agent.loop import DiscoveryLoop

    loop = DiscoveryLoop(
        pipeline=MagicMock(),
        engine=MagicMock(),
        state=MagicMock(),
        max_generations=10,
        batch_size=5,
        use_nsga2=False,
        active_learning_threshold=0.7,
    )

    mock_feedback_controller = MagicMock()
    loop.feedback_controller = mock_feedback_controller

    with patch.object(loop, '_evaluate_with_xtb') as mock_xtb:
        result = loop._maybe_escalate(
            smi="CCOC(=O)OCC",
            ctx=MagicMock(),
            total_score=80.0,
            conformal_conf=0.5,
            score_data={},
            sub_scores={},
            t2={"quantum_confidence": "tom_medium", "homo_eV": -7.0, "lumo_eV": -1.0}
        )

        mock_feedback_controller.log_active_learning_trigger.assert_not_called()
        mock_xtb.assert_not_called()

        updated_total_score, updated_conf, updated_score_data, updated_sub_scores, updated_t2 = result
        assert updated_total_score == 80.0
        assert updated_conf == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
