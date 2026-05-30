"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. **Filter** — Quick structural validity and synthetic accessibility (SA) check.
  2. **Oracle** — Evaluate target property (e.g. HOMO/LUMO gap) using the
     PropertyOracle (QSPR-based).

The results are then fed back to the RF surrogate for Bayesian optimisation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from scipy.stats import norm as norm_dist

from aurelius.config import AureliusConfig, apply_global_config
from aurelius.scoring.oracle import PropertyOracle
from aurelius.screening.tier1 import Filter

logger = logging.getLogger(__name__)


class AureliusPipeline:
    """Full Aurelius screening pipeline orchestrator.

    Coordinates the Filter -> Oracle pipeline and computes
    the final Aurelius Score.
    """

    def __init__(
        self,
        config: AureliusConfig | None = None,
        use_real_models: bool = True,
    ) -> None:
        """Initialise the Aurelius pipeline.

        Args:
            config: Pipeline configuration. If None, loads default.
        """
        self.config = config or apply_global_config()
        self._filter: Filter | None = None
        self._use_real_models = use_real_models
        self._oracle: PropertyOracle | None = None

    def initialize(self) -> None:
        """Initialise all pipeline components."""
        try:
            import rdkit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for pipeline initialisation. "
                "Install with: pip install rdkit"
            ) from None

        # Phase 1: Structural filter
        if self._use_real_models:
            try:
                self._filter = Filter()
                logger.info("Tier 1 (Filter): ENABLED")
            except Exception as exc:
                logger.warning("Tier 1 (Filter): DISABLED - %s", exc)
                self._filter = None

        # Phase 2: Oracle for property evaluation
        self._oracle = PropertyOracle()
        logger.info("Oracle (PropertyOracle): ENABLED")

    def _generate_failed_run(self, smiles: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a failed run result dict for early-exit scenarios."""
        t1_result = {
            "molecule_smiles": smiles,
            "is_viable": False,
            "inference_time_ms": 0.0,
        }

        oracle_result = self._oracle.evaluate(smiles) if self._oracle else {}
        lumo_gap = oracle_result.get("lumo_gap_eV", 999.0)

        total_score = max(0.0, 100.0 - lumo_gap * 5.0)
        is_viable = lumo_gap < 6.0

        return {
            "tier1": t1_result,
            "tier2": None,
            "score": {
                "total_score": total_score,
                "is_viable": is_viable,
                "rejection_reasons": [reason],
            },
        }

    def screen_molecule(self, smiles: str, **kwargs: Any) -> dict[str, Any]:
        """Run a single molecule through the Filter -> Oracle pipeline.

        Returns a dict with tier results and the final Aurelius score.
        Includes per-tier timing metrics for performance monitoring.
        """
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")

        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        # Step 1: Filter (structural validity + SA score)
        t1_result = None
        if self._filter:
            t1_start = time.perf_counter()
            t1_result = self._filter.screen_molecule(smiles)
            tier_timings: dict[str, float] = {}
            tier_timings["tier1_ms"] = (time.perf_counter() - t1_start) * 1000
            results: dict[str, Any] = {"tier1": t1_result}
            logger.info(
                "Tier 1 Result: %s -> %s (time=%.1fms)",
                smiles,
                "VIABLE" if t1_result.get("is_viable", False) else "REJECTED",
                t1_result.get("inference_time_ms", 0.0),
            )
            if not t1_result.get("is_viable", True):
                logger.warning("Short-circuiting: %s failed Tier 1.", smiles)
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter", **kwargs)
        else:
            results = {}
            tier_timings = {}

        # Step 2: Oracle (property evaluation)
        t2_result = None
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(smiles)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            t2_result = {
                "homo_eV": oracle_result.get("homo_eV", 999.0),
                "lumo_eV": oracle_result.get("lumo_eV", 999.0),
                "lumo_gap_eV": oracle_result.get("lumo_gap_eV", 999.0),
                "dipole_debye": oracle_result.get("dipole_debye", 999.0),
            }
            results["tier2"] = t2_result
            logger.info(
                "Property Oracle Result: %s -> LUMO gap=%.3f eV",
                smiles,
                t2_result["lumo_gap_eV"],
            )

            if t2_result["lumo_gap_eV"] > 10.0:
                logger.warning(
                    "Short-circuiting: %s failed Oracle (LUMO gap=%.3f eV)",
                    smiles,
                    t2_result["lumo_gap_eV"],
                )
                return self._generate_failed_run(
                    smiles,
                    f"Failed Oracle (LUMO gap: {t2_result['lumo_gap_eV']} eV)",
                    **kwargs,
                )

        score = self._compute_score(oracle_result)
        results["score"] = score

        logger.debug("Scorecard:\n%s", self._format_score(score))

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            logger.info("Performance: total=%.1fms | %s", total_ms, " | ".join(timing_lines))

        return results

    def _compute_score(self, oracle_result: dict[str, float]) -> dict[str, Any]:
        """Compute the final Aurelius Score from oracle results.

        Rewards LUMO in [-1.5, -0.5] eV (SEI formation window) and
        HOMO < -6.0 eV (oxidative stability).  Outside these bounds,
        a Gaussian penalty is applied.

        Args:
            oracle_result: Dictionary with keys like ``lumo_gap_eV``, ``homo_eV``, etc.

        Returns:
            Dict with ``total_score``, ``is_viable``, and ``rejection_reasons``.
        """
        lumo_gap = oracle_result.get("lumo_gap_eV", 999.0)
        homo = oracle_result.get("homo_eV", 999.0)

        lumo_mean, lumo_std = -1.0, 0.35
        homo_mean, homo_std = -6.0, 1.0

        lumo_score = norm_dist.pdf(lumo_gap, loc=lumo_mean, scale=lumo_std) * 100.0
        homo_score = norm_dist.pdf(homo, loc=homo_mean, scale=homo_std) * 100.0

        if lumo_gap < -1.5 or lumo_gap > -0.5:
            lumo_score *= 0.1
        if homo > -6.0:
            homo_score *= 0.1

        total_score = min(lumo_score, homo_score)
        total_score = max(0.0, total_score)

        is_viable = total_score >= 20.0

        rejection_reasons: list[str] = []
        if not is_viable:
            rejection_reasons.append(
                f"Aurelius Score {total_score:.1f} below viability threshold "
                f"(LUMO gap: {lumo_gap:.3f} eV, HOMO: {homo:.3f} eV)"
            )

        return {
            "total_score": total_score,
            "is_viable": is_viable,
            "rejection_reasons": rejection_reasons,
        }

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        """Format a score dict for logging."""
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"

    def screen_batch(self, smiles_list: list[str], n_workers: int = 1, **kwargs: Any) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the full pipeline.

        When ``n_workers`` is greater than 1, molecules are screened
        in parallel using ``ThreadPoolExecutor``.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(smiles, **kwargs) for smiles in smiles_list]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, smiles, **kwargs): i
                for i, smiles in enumerate(smiles_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(smiles_list))]
