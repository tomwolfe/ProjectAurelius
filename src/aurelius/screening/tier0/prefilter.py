"""Tier-0 GC prefilter for Project Aurelius.

Physical justification: Before invoking the expensive quantum oracle (xTB/TOM), we
filter out molecules that will almost certainly fail on bulk properties. The
Goldilocks zone for electrolyte solvents requires:
  - Dielectric proxy > 2.0: Enough polar character to dissociate salts
  - Viscosity proxy < 8.0: Adequate ion mobility

These thresholds are calibrated against the external benchmark and reflect
empirical constraints for viable electrolyte solvents.
"""

import logging
from typing import Any

from aurelius.scoring.oracle import (
    predict_dielectric_proxy,
    predict_viscosity_proxy,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


class Tier0Prefilter:
    """Pre-filter molecules using cheap GC-only predictions.

    This filter runs before the quantum oracle to reject molecules that cannot
    achieve the required dielectric/viscosity regime. It uses only fragment-
    additive GC predictions (no QM) to keep evaluation cost low while
    dramatically improving Oracle throughput.

    The filter is a hard gate: molecules failing either check are discarded
    entirely (counted as invalid for the batch).
    """

    def __init__(self, min_dielectric: float = 2.6, max_viscosity: float = 8.0) -> None:
        """Initialize prefilter thresholds.

        Args:
            min_dielectric: Minimum dielectric constant for viability.
            max_viscosity: Maximum viscosity proxy for viability.

        ADR-2026-08-07-04: ``min_dielectric`` moved 2.0 -> 2.6 when the oracle
        switched to the true ε scale. This remains a deliberately permissive
        gate that rejects only clearly non-dissociating hydrocarbons
        (hexane 2.14, cyclohexane 2.23, toluene 2.41, CCl4 2.38) while
        retaining every practical co-solvent, including the low-ε linear
        carbonates DEC (2.81) and DMC (3.13) that are essential as
        viscosity-reducing blend components.
        """
        self.min_dielectric = min_dielectric
        self.max_viscosity = max_viscosity

    def filter(self, contexts: list[MoleculeContext]) -> tuple[list[MoleculeContext], dict[str, Any]]:
        """Filter a list of contexts based on GC properties.

        Args:
            contexts: List of MoleculeContext objects to filter.

        Returns:
            Tuple of (passed_contexts, filter_stats) where:
            - passed_contexts: Contexts that passed the prefilter
            - filter_stats: Statistics about filtering performance
        """
        passed: list[MoleculeContext] = []
        rejection_reasons = {"dielectric": 0, "viscosity": 0, "passed": 0}

        for ctx in contexts:
            # Predict dielectric proxy
            dielectric_proxy = predict_dielectric_proxy(ctx)
            # Predict viscosity proxy
            viscosity_proxy = predict_viscosity_proxy(ctx)

            # Apply thresholds
            if dielectric_proxy < self.min_dielectric:
                rejection_reasons["dielectric"] += 1
                continue
            if viscosity_proxy > self.max_viscosity:
                rejection_reasons["viscosity"] += 1
                continue

            # Passed filter
            passed.append(ctx)
            rejection_reasons["passed"] += 1

        filter_stats = {
            "total_evaluated": len(contexts),
            "passed_count": rejection_reasons["passed"],
            "dielectric_rejected": rejection_reasons["dielectric"],
            "viscosity_rejected": rejection_reasons["viscosity"],
            "pass_rate": rejection_reasons["passed"] / max(1, len(contexts)),
        }

        passed_count = len(passed)
        if passed_count > 0:
            logger.info(
                "Tier-0 GC prefilter: %d/%d molecules passed (dielectric < %.1f, "
                "viscosity > %.1f). Throughput: %.1f%%",
                passed_count, len(contexts),
                self.min_dielectric, self.max_viscosity,
                100.0 * passed_count / max(1, len(contexts)),
            )

        return passed, filter_stats

    def screen_molecule(self, ctx: MoleculeContext) -> tuple[bool, str]:
        """Screen a single molecule through Tier-0.

        Args:
            ctx: MoleculeContext to screen.

        Returns:
            Tuple of (is_viable, reason)
        """
        dielectric_proxy = predict_dielectric_proxy(ctx)
        viscosity_proxy = predict_viscosity_proxy(ctx)

        if dielectric_proxy < self.min_dielectric:
            return False, f"dielectric_proxy {dielectric_proxy:.3f} < {self.min_dielectric}"
        if viscosity_proxy > self.max_viscosity:
            return False, f"viscosity_proxy {viscosity_proxy:.3f} > {self.max_viscosity}"

        return True, ""

    def get_cache_size(self) -> int:
        """Get approximate cache size (theoretical for GC models).

        Returns:
            This implementation always returns 0 since GC predictions
            are stateless and computed on-demand.
        """
        return 0

    def clear_cache(self) -> None:
        """Clear prediction cache (no-op for GC models)."""
        pass
