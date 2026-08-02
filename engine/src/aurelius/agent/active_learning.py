"""Active learning queue manager for the discovery loop.

Extracts the active learning logic from ``DiscoveryLoop`` into a single
dedicated class, giving the loop a single responsibility: orchestration.
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.types import MoleculeContext, ScreeningResult

log = logging.getLogger(__name__)


class ActiveLearningManager:
    """Manages the active learning queue and real-quantum evaluation pathway.

    Responsibilities:
      - Checking GC UQ uncertainty and populating the active learning queue.
      - Selecting molecules from the queue in FIFO order.
      - Evaluating queued molecules with the real ``QuantumOracle`` (bypassing
        the surrogate).
      - Computing epistemic uncertainties for UCB-based exploration.
    """

    def __init__(
        self,
        pipeline: Any,
        engine: Any,
        state: Any,
        batch_size: int = 50,
        max_wall_time: float = 43200.0,
    ) -> None:
        self.pipeline = pipeline
        self.engine = engine
        self.state = state
        self.batch_size = batch_size
        self.max_wall_time = max_wall_time
        self._wall_start: float = 0.0

    # ------------------------------------------------------------------
    # Wall-time helper
    # ------------------------------------------------------------------

    def set_wall_start(self, t: float) -> None:
        self._wall_start = t

    def _wall_time_exceeded(self) -> bool:
        import time
        return self._wall_start > 0 and time.time() - self._wall_start > self.max_wall_time

    # ------------------------------------------------------------------
    # Uncertainty checks & queue population
    # ------------------------------------------------------------------

    def check_uq_and_queue(self, ctx: MoleculeContext, gc_uq: Any) -> None:
        """Add molecule to active learning queue if GC UQ variance exceeds threshold."""
        smi = ctx.smiles
        try:
            _, _, diel_high = gc_uq.predict_dielectric(ctx)
        except Exception:
            return
        try:
            _, _, visc_high = gc_uq.predict_viscosity(ctx)
        except Exception:
            return
        if (diel_high or visc_high) and smi not in self.state.active_learning_queue:
            self.state.active_learning_queue.append(smi)
            log.info("  Added %s to active learning queue (high UQ)", smi)

    def check_and_queue(
        self, ctx: MoleculeContext, result_map: dict[str, Any]
    ) -> tuple[float, dict[str, Any]] | None:
        """Check if the candidate qualifies for real-quantum evaluation.

        Three pathways:
        1. Already in the active learning queue
        2. High GC UQ ensemble uncertainty (dielectric or viscosity)
        3. High surrogate quantum uncertainty (>0.5)

        If any pathway triggers, the molecule is added to the queue (if not
        already present) and evaluated with the real ``QuantumOracle``.

        Returns ``(total_score, tier2_dict)`` if real-quantum evaluation
        was triggered, or ``None`` to proceed with the normal screening path.
        """
        smi = ctx.smiles

        # PATH 1: Already in the active learning queue
        if smi in self.state.active_learning_queue:
            return self.evaluate_with_real_quantum(ctx, result_map)

        # PATH 2: High GC UQ ensemble uncertainty
        gc_uq = getattr(getattr(self.pipeline, '_oracle', None), '_gc_uq', None)
        if gc_uq is not None:
            try:
                _, _, diel_high = gc_uq.predict_dielectric(ctx)
                _, _, visc_high = gc_uq.predict_viscosity(ctx)
                if diel_high or visc_high:
                    if smi not in self.state.active_learning_queue:
                        self.state.active_learning_queue.append(smi)
                        log.info(
                            "  Added %s to active learning queue (high UQ from GcUqEnsemble)",
                            smi,
                        )
                    return self.evaluate_with_real_quantum(ctx, result_map)
            except Exception:
                pass

        # PATH 3: High surrogate quantum uncertainty
        try:
            from aurelius.scoring.oracle.surrogate import SurrogateQuantumOracle
            surrogate = SurrogateQuantumOracle()
            homo, lumo, uncertainty = surrogate.predict(ctx)
            penalty = surrogate.compute_penalty(homo, uncertainty)
            if penalty == 1.0 and uncertainty > 0.5:
                return self.evaluate_with_real_quantum(ctx, result_map)
        except Exception:
            pass

        return None

    def get_uncertainties(self, contexts: list[MoleculeContext]) -> list[float]:
        """Compute combined UQ uncertainties for a list of contexts."""
        gc_uq = getattr(getattr(self.pipeline, '_oracle', None), '_gc_uq', None)
        uncertainties: list[float] = []
        for ctx in contexts:
            if gc_uq is not None:
                try:
                    _, diel_std, _ = gc_uq.predict_dielectric(ctx)
                    _, visc_std, _ = gc_uq.predict_viscosity(ctx)
                    uncertainties.append((diel_std + visc_std) / 2.0)
                except Exception:
                    uncertainties.append(0.0)
            else:
                uncertainties.append(0.0)
        return uncertainties

    def populate_from_uncertainties(
        self,
        result_contexts: list[MoleculeContext],
        result_map: dict[str, Any],
    ) -> None:
        """Identify top 10% of evaluated candidates with highest uncertainty_score and add to AL queue."""
        scored: list[tuple[str, float]] = []
        for ctx in result_contexts:
            t2 = result_map.get(ctx.smiles)
            if t2 is None:
                continue
            uq = t2.get("uncertainty_score", 0.0) or 0.0
            scored.append((ctx.smiles, uq))

        if len(scored) < 2:
            return

        scored.sort(key=lambda x: -x[1])
        n_queue = max(1, len(scored) // 10)
        for smi, unc in scored[:n_queue]:
            if smi not in self.state.active_learning_queue:
                self.state.active_learning_queue.append(smi)
                log.info("  Added %s to active learning queue (uncertainty_score=%.4f)", smi, unc)

    # ------------------------------------------------------------------
    # Queue selection (FIFO)
    # ------------------------------------------------------------------

    def select_from_queue(
        self,
        result_contexts: list[MoleculeContext],
        all_scores: list[float],
    ) -> tuple[list[MoleculeContext], list[float]] | None:
        """If the active learning queue has items, select from it in FIFO order.

        Returns (selected, scores) or None if the queue is empty.
        """
        if not self.state.active_learning_queue:
            return None

        result_by_smiles: dict[str, tuple[MoleculeContext, float]] = {
            ctx.smiles: (ctx, score)
            for ctx, score in zip(result_contexts, all_scores, strict=False)
        }

        selected_contexts: list[MoleculeContext] = []
        selected_scores: list[float] = []
        remaining_queue: list[str] = []

        for smi in self.state.active_learning_queue:
            if smi in result_by_smiles:
                ctx, score = result_by_smiles[smi]
                if len(selected_contexts) < self.batch_size:
                    selected_contexts.append(ctx)
                    selected_scores.append(score)
                else:
                    remaining_queue.append(smi)
            else:
                remaining_queue.append(smi)

        self.state.active_learning_queue = remaining_queue

        if not selected_contexts:
            return None

        return selected_contexts, selected_scores

    # ------------------------------------------------------------------
    # Queue inspection helpers
    # ------------------------------------------------------------------

    @property
    def queue_size(self) -> int:
        """Number of molecules currently in the active learning queue."""
        return len(self.state.active_learning_queue)

    def get_queue_smiles(self, n: int) -> list[str]:
        """Return up to *n* SMILES from the front of the active learning queue."""
        if not self.state.active_learning_queue:
            return []
        return list(self.state.active_learning_queue[:n])

    # ------------------------------------------------------------------
    # Real quantum evaluation (xTB / TOM)
    # ------------------------------------------------------------------

    def evaluate_with_real_quantum(
        self, ctx: MoleculeContext, result_map: dict[str, Any]
    ) -> tuple[float, dict[str, Any]] | None:
        """Evaluate a candidate using the real QuantumOracle (bypassing surrogate)."""
        from aurelius.scoring.oracle.gc import (
            predict_ced_proxy,
            predict_dielectric_proxy,
            predict_li_solvation_proxy,
            predict_viscosity_proxy,
        )
        if self._wall_time_exceeded():
            return None

        from aurelius.scoring.oracle.quantum import QuantumOracle

        qo = QuantumOracle()
        qr = qo.evaluate(ctx.mol)

        homo_eV = qr.get("homo_eV", -99.0)
        lumo_eV = qr.get("lumo_eV", -99.0)

        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)
        ced = predict_ced_proxy(ctx)

        score = self.pipeline._compute_score(
            homo_eV=homo_eV, lumo_eV=lumo_eV,
            dielectric_proxy=dielectric,
            viscosity_proxy=viscosity,
            li_solvation_proxy=li_solvation,
            ced_proxy=ced,
            ctx=ctx,
            quantum_confidence="xtb",
        )

        t2 = {
            "homo_eV": homo_eV,
            "lumo_eV": lumo_eV,
            "gap_eV": qr.get("gap_eV", lumo_eV - homo_eV),
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "li_solvation_proxy": li_solvation,
            "ced_proxy": ced,
        }

        smi = ctx.smiles
        result_map[smi] = t2
        self.engine.add_to_db(smi)

        total_score = score.get("total_score", 0.0)
        self.engine.record_reaction_success(smi, total_score)

        novelty = self._compute_novelty(ctx)
        sr = _build_screening_result(
            smi, total_score, score, t2, novelty, ctx, score.get("sub_scores", {}),
        )
        if _is_discovery(total_score, score):
            self.state.add_discovery(sr)
            log.info("  ** DISCOVERY (active learning) ** %s (score=%.1f)", smi, total_score)
        self.state.add_result(sr)
        log.info("  ** ACTIVE LEARNING ** %s evaluated via real QuantumOracle", smi)
        return total_score, t2

    def _compute_novelty(self, ctx: MoleculeContext) -> float | None:
        """Compute novelty score (1 - max Tanimoto to seed pool)."""
        fp = ctx.get_ecfp4()
        seed_fps = getattr(self.engine, "seed_fingerprints", None)
        if not isinstance(seed_fps, list) or not seed_fps:
            return None
        from rdkit.DataStructs import BulkTanimotoSimilarity
        sims = BulkTanimotoSimilarity(fp, seed_fps)
        return 1.0 - max(sims) if sims else None


def _build_screening_result(
    smi: str, total_score: float, score_data: dict[str, Any],
    t2: dict[str, Any], novelty: float | None,
    ctx: MoleculeContext, sub_scores: dict[str, Any],
) -> ScreeningResult:
    from aurelius.scoring.oracle.gc import compute_estimated_cost_score
    return ScreeningResult(
        smiles=smi,
        total_score=total_score,
        is_viable=score_data.get("is_viable", False),
        rejection_reasons=score_data.get("rejection_reasons", []),
        fingerprint=ctx.get_feature_vector(),
        novelty_to_seed=novelty,
        homo_eV=t2.get("homo_eV"),
        lumo_eV=t2.get("lumo_eV"),
        dielectric_proxy=t2.get("dielectric_proxy"),
        viscosity_proxy=t2.get("viscosity_proxy"),
        li_solvation_proxy=t2.get("li_solvation_proxy"),
        sa_score=score_data.get("sa_score"),
        sub_scores=sub_scores,
        estimated_cost_score=compute_estimated_cost_score(ctx),
        uncertainty_score=t2.get("uncertainty_score"),
    )


def _is_discovery(total_score: float, score_data: dict[str, Any]) -> bool:
    from aurelius.constants import DISCOVERY_THRESHOLD
    return (total_score >= DISCOVERY_THRESHOLD
            and score_data.get("is_viable", False)
            and not score_data.get("rejection_reasons", []))
