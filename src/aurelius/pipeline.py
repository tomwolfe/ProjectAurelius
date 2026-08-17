"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. Filter — Quick structural validity (Tier 1) check with LogP and MW gates.
  2. Oracle — Multi-objective property evaluation via hybrid quantum/GC
     correction (HOMO, LUMO, Dielectric proxy, Viscosity proxy, SA Score).
  3. Score — Multi-objective composite with Al corrosion penalty.
     Each objective is a callable that transforms a raw value to a sub-score.

All stages accept a pre-parsed MoleculeContext to enforce single-point parsing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.agent.mutation.retrosynthetic import brics_retrosynthetic_depth
from aurelius.constants import (
    AL_CORROSION_LUMO_THRESHOLD,
    AL_CORROSION_MIN_FLUORINE,
    AL_CORROSION_PENALTY_FACTOR,
    DIELECTRIC_TARGET,
    EA_SIGMA,
    EA_TARGET,
    HOMO_THRESHOLD,
    LI_SOLVATION_TARGET,
    SA_THRESHOLD,
    SCORE_WEIGHT_DIELECTRIC,
    SCORE_WEIGHT_HOMO,
    SCORE_WEIGHT_LI_SOLVATION,
    SCORE_WEIGHT_LUMO,
    SCORE_WEIGHT_SA,
    SCORE_WEIGHT_VISCOSITY,
    VIABILITY_THRESHOLD,
    VISCOSITY_THRESHOLD,
)
from aurelius.constants import (
    CARBONYL_F_PATTERN as _CARBONYL_F_PATTERN,
)
from aurelius.constants import (
    CF3_PATTERN as _CF3_PATTERN,
)
from aurelius.constants import (
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_PATTERNS,
)
from aurelius.constants import (
    HYPOFLUORITE_PATTERN as _HYPOFLUORITE_PATTERN,
)
from aurelius.constants import (
    HYPOFLUORITE_PENALTY_FACTOR as _HYPOFLUORITE_PENALTY,
)
from aurelius.constants import (
    SULFONYL_F_PATTERN as _SULFONYL_F_PATTERN,
)
from aurelius.scoring.oracle import (
    PropertyOracle,
    mixture_synergy_bonus,
    predict_mixture_dielectric,
    predict_mixture_li_solvation,
    predict_mixture_viscosity,
)
from aurelius.screening.tier1 import Filter
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import electrolyte_synthetic_accessibility
from aurelius.utils.chem_utils import synthesizability_complexity as _synthesizability_complexity

logger = logging.getLogger(__name__)


def _gaussian(value: float, target: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((value - target) / sigma) ** 2))


def _sigmoid(value: float, target: float, steepness: float, higher_is_better: bool = True) -> float:
    if higher_is_better:
        return float(1.0 / (1.0 + np.exp(-steepness * (value - target))))
    return float(1.0 / (1.0 + np.exp(steepness * (value - target))))


@dataclass
class Objective:
    """A single scoring objective with a direct callable function.

    Each objective defines how a raw property value is converted to a
    sub-score via a callable function, then weighted in the final composite.
    """

    name: str
    property_key: str
    weight: float
    function: Callable[[float], float]
    failure_reason_template: str = "{name}={value:.3f} (below threshold)"

    def __call__(self, value: float) -> float:
        return self.function(value)


def _extract_ea(tier2_result: dict[str, Any] | None) -> float | None:
    """Pull the ΔSCF electron affinity out of an oracle result, if present.

    Returns None when the reduction oracle did not run or produced no value,
    so the caller can fall back to a neutral objective rather than treating a
    missing measurement as a bad one.
    """
    if not tier2_result:
        return None
    proxy = tier2_result.get("reduction_stability_proxy")
    if not isinstance(proxy, dict):
        return None
    value = proxy.get("ea_eV")
    return float(value) if isinstance(value, (int, float)) else None


# Weight constants for the composite score.
# ADR-2026-08-15: li_solvation_reward (donor number) de-rated from 0.20 to 0.05
# due to poor rank correlation (ρ ≈ 0.19). Freed weight redistributed to
# reduction_stability_reward (ΔSCF EA, ρ = 0.91) and dielectric_reward.
# LUMO axis (lumo_eV) is not used as a scoring objective — it was replaced by
# the ΔSCF electron affinity axis (reduction_stability_reward) per ADR-2026-08-10.
# The constant SCORE_WEIGHT_LUMO (0.23) now drives the validated EA axis.
_SCORE_WEIGHT_REDUCTION_STABILITY: float = 0.305  # was 0.23 (SCORE_WEIGHT_LUMO) + 0.075
_SCORE_WEIGHT_HOMO: float = 0.17
_SCORE_WEIGHT_DIELECTRIC: float = 0.245  # was 0.17 + 0.075
_SCORE_WEIGHT_VISCOSITY: float = 0.14
_SCORE_WEIGHT_LI_SOLVATION: float = 0.05  # de-rated from 0.20
_SCORE_WEIGHT_SA: float = 0.01

_OBJECTIVES: list[Objective] = [
    # ADR-2026-08-10: reduction stability is ranked by ΔSCF electron affinity
    # (rho = 0.91 against 40 measured gas-phase EAs), not by the frontier LUMO
    # it replaces (rho = 0.34 against a 0.31 permutation bar).
    # LUMO axis provenance-confounded (audit_label_confound.py);
    # de-rated pending single-method calibration.
    Objective("reduction_stability_reward", "ea_eV", _SCORE_WEIGHT_REDUCTION_STABILITY,
              lambda v: _gaussian(v, EA_TARGET, EA_SIGMA),
              failure_reason_template="EA={value:.3f}eV (reduction-unstable)"),
    Objective("homo_penalty", "homo_eV", _SCORE_WEIGHT_HOMO,
              lambda v: _sigmoid(v, HOMO_THRESHOLD, 5.0, False),
              failure_reason_template="HOMO={value:.3f}eV (oxidative instability)"),
    Objective("dielectric_reward", "dielectric_proxy", _SCORE_WEIGHT_DIELECTRIC,
              lambda v: _sigmoid(v, DIELECTRIC_TARGET, 1.5),
              failure_reason_template="dielectric_proxy={value:.3f} (poor salt dissolution)"),
    Objective("viscosity_penalty", "viscosity_proxy", _SCORE_WEIGHT_VISCOSITY,
              lambda v: _sigmoid(v, VISCOSITY_THRESHOLD, 2.0, False),
              failure_reason_template="viscosity_proxy={value:.3f} (poor ion mobility)"),
    # Donor-number axis ρ ≈ 0.19; de-rated to exploratory.
    Objective("li_solvation_reward", "li_solvation_proxy", _SCORE_WEIGHT_LI_SOLVATION,
              lambda v: _gaussian(v, LI_SOLVATION_TARGET, 1.0),
              failure_reason_template="li_solvation_proxy={value:.3f} (poor Li+ binding)"),
    Objective("sa_penalty", "sa_score", _SCORE_WEIGHT_SA,
              lambda v: _sigmoid(v, SA_THRESHOLD, 2.0, False),
              failure_reason_template="SA score={value:.2f} (hard to synthesize)"),
    Objective("synthesizability_reward", "sa_score", 0.20,
              lambda v: 1.0 - (v / 10.0),
              failure_reason_template="SA score={value:.2f} (hard to synthesize)"),
    Objective("combined_grounding_score", "grounding", 0.15,
              lambda v: v,
              failure_reason_template="grounding={value:.3f} (no BRICS grounding)"),
]


class AureliusPipeline:
    """Full Aurelius screening pipeline orchestrator.

    Coordinates the Filter -> Oracle -> Score pipeline.
    All stages accept a pre-parsed MoleculeContext to avoid redundant
    RDKit parsing.
    """

    def __init__(
        self,
        use_real_models: bool = True,
        use_xtb: bool = True,
    ) -> None:
        self._filter: Filter | None = None
        self._use_real_models = use_real_models
        self._use_xtb = use_xtb
        self._oracle: PropertyOracle | None = None

    def initialize(self) -> None:
        """Initialise all pipeline components."""
        import rdkit  # noqa: F401

        if self._use_real_models:
            try:
                self._filter = Filter()
                logger.info("Tier 1 (Filter): ENABLED")
            except Exception as exc:
                logger.warning("Tier 1 (Filter): DISABLED - %s", exc)
                self._filter = None

        self._oracle = PropertyOracle(use_xtb=self._use_xtb)
        oracle_cache = "oracle_cache.joblib"
        if not self._oracle.load(oracle_cache):
            logger.info("Oracle (PropertyOracle): no cache found — using GC model directly.")
        else:
            logger.info("Oracle (PropertyOracle): loaded from cache (%s).", oracle_cache)
        logger.info("Oracle (PropertyOracle): ENABLED")

    def _generate_failed_run(self, smiles: str, reason: str) -> dict[str, Any]:
        t1_result = {
            "molecule_smiles": smiles,
            "is_viable": False,
            "inference_time_ms": 0.0,
        }
        return {
            "tier1": t1_result,
            "tier2": None,
            "score": {
                "total_score": 0.0,
                "is_viable": False,
                "rejection_reasons": [reason],
            },
        }

    def screen_molecule(
        self,
        ctx: MoleculeContext,
    ) -> dict[str, Any]:
        """Run a single pre-parsed molecule through the Filter -> Oracle pipeline.

        Thin single-candidate wrapper over ``screen_batch``: batch and scalar
        screening share one implementation so the two paths can never drift.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dict with tier results and the final Aurelius score.

        Raises:
            TypeError: If ctx is not a MoleculeContext.
            RuntimeError: If pipeline not initialised.
        """
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"AureliusPipeline.screen_molecule() requires a MoleculeContext, "
                f"got {type(ctx).__name__}. Use MoleculeContext.from_smiles() first."
            )
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialize() first.")

        return self.screen_batch([ctx])[0]

    def screen_mixture(
        self,
        ctx1: MoleculeContext,
        ctx2: MoleculeContext,
        frac1: float = 0.5,
    ) -> dict[str, Any]:
        """Score a binary mixture using ideal mixing rules with synergy bonus.

        Evaluates each component individually via screen_molecule, then
        computes mixture dielectric/viscosity using thermodynamic mixing
        rules and applies a non-linear synergy bonus for complementary pairs
        (high-dielectric + low-viscosity).

        Args:
            ctx1: First component MoleculeContext.
            ctx2: Second component MoleculeContext.
            frac1: Volume fraction of first component.

        Returns:
            Dict with component results, mixture properties, and score.
        """
        res1 = self.screen_molecule(ctx1)
        res2 = self.screen_molecule(ctx2)

        p1 = res1.get("tier2", {})
        p2 = res2.get("tier2", {})
        if p1 is None:
            p1 = {}
        if p2 is None:
            p2 = {}
        f2 = 1.0 - frac1

        d1 = p1.get("dielectric_proxy", 0.0)
        d2 = p2.get("dielectric_proxy", 0.0)
        v1 = p1.get("viscosity_proxy", 99.0)
        v2 = p2.get("viscosity_proxy", 99.0)

        d_mix = predict_mixture_dielectric(d1, d2, frac1)
        v_mix = predict_mixture_viscosity(v1, v2, frac1)
        ls_mix = predict_mixture_li_solvation(
            p1.get("li_solvation_proxy", 0.0),
            p2.get("li_solvation_proxy", 0.0),
            frac1,
        )
        h_mix = frac1 * p1.get("homo_eV", -99.0) + f2 * p2.get("homo_eV", -99.0)
        l_mix = frac1 * p1.get("lumo_eV", -99.0) + f2 * p2.get("lumo_eV", -99.0)

        synergy = mixture_synergy_bonus(d1, d2, v1, v2, frac1)

        # Base score from individual component scores weighted average
        s1 = res1.get("score", {}).get("total_score", 0.0)
        s2 = res2.get("score", {}).get("total_score", 0.0)
        weighted_base = frac1 * s1 + f2 * s2

        total = min(100.0, weighted_base + synergy)
        is_viable = total >= VIABILITY_THRESHOLD

        score: dict[str, Any] = {
            "total_score": total,
            "is_viable": is_viable,
            "synergy_bonus": round(synergy, 4),
            "weighted_base": round(weighted_base, 4),
            "sub_scores": {
                "component1_score": round(s1, 4),
                "component2_score": round(s2, 4),
            },
            "rejection_reasons": [],
        }

        mixture_props: dict[str, float] = {
            "dielectric_proxy": round(d_mix, 4),
            "viscosity_proxy": round(v_mix, 4),
            "li_solvation_proxy": round(ls_mix, 4),
            "homo_eV": round(h_mix, 4),
            "lumo_eV": round(l_mix, 4),
            "synergy_bonus": round(synergy, 4),
        }

        return {
            "component1": res1,
            "component2": res2,
            "mixture_properties": mixture_props,
            "score": score,
        }

    @staticmethod
    def _check_hydrolytic_instability(mol: Chem.Mol) -> float:
        """Check for hydrolytically unstable motifs.

        Returns a multiplier in [0.5, 1.0].
        """
        penalty = 1.0
        for pattern, name, severity in _HYDRO_PATTERNS:
            if pattern is not None and mol.HasSubstructMatch(pattern):
                penalty *= (1.0 - severity)
                logger.debug("Hydrolytic instability detected: %s (penalty %.2f)", name, severity)
        return max(penalty, 0.5)

    @staticmethod
    def _check_hypofluorite_instability(mol: Chem.Mol) -> float:
        """Penalise molecules with O-F (hypofluorite) bonds.

        Hypofluorites are violently reactive oxidisers — they decompose
        exothermically at room temperature and cannot be used as battery
        electrolyte solvents. The EA's methyl-to-fluorine SMARTS reaction
        generates these from carbonate/ether seed molecules, exploiting
        the scoring function's fluorine reward.

        Returns a multiplier in [0.50, 1.0].
        """
        if _HYPOFLUORITE_PATTERN is not None and mol.HasSubstructMatch(_HYPOFLUORITE_PATTERN):
            return _HYPOFLUORITE_PENALTY
        return 1.0

    @staticmethod
    def _check_al_corrosion_risk(mol: Chem.Mol) -> float:
        """Check for Al corrosion risk in high-LUMO fluorinated molecules.

        High-LUMO fluorinated solvents can corrode Al cathode current
        collectors via AlF3 formation. Returns a penalty multiplier in [0.7, 1.0].

        The penalty applies when:
          1. The molecule has AL_CORROSION_MIN_FLUORINE or more fluorine atoms
             AND at least one CF3 group (electron-withdrawing environment), OR
          2. The molecule has >= 3 fluorine atoms attached directly to
             electron-withdrawing groups (carbonyl-adjacent or sulfonyl-adjacent).

        Note: LUMO threshold check is done in _compute_score; this function
        only checks the structural criteria.
        """
        # Count fluorine atoms
        n_f = sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())

        # Count CF3 groups
        n_cf3 = len(mol.GetSubstructMatches(_CF3_PATTERN))

        # Count fluorine adjacent to carbonyl or sulfonyl
        n_f_ewg = len(mol.GetSubstructMatches(_CARBONYL_F_PATTERN))
        n_f_ewg += len(mol.GetSubstructMatches(_SULFONYL_F_PATTERN))

        if n_f >= AL_CORROSION_MIN_FLUORINE and (n_cf3 >= 1 or n_f_ewg >= 1):
            return AL_CORROSION_PENALTY_FACTOR
        return 1.0

    @staticmethod
    def _grounding_score(ctx: MoleculeContext) -> float:
        """Raw synthesizability grounding in [0, 1] for a pre-parsed context.

        Thin wrapper over ``combined_grounding_score`` that never raises, so a
        BRICS decomposition failure degrades to 0.0 (maximally penalised) rather
        than aborting the score. Computed exactly once per molecule and threaded
        through ``_compute_score`` so both the penalty multiplier and the
        selection objective read the same value.
        """
        from aurelius.agent.mutation.brics import combined_grounding_score
        try:
            return float(combined_grounding_score(ctx.mol))
        except Exception:
            return 0.0

    @staticmethod
    def _check_building_block_grounding(grounding: float) -> float:
        """Penalty for molecules whose fragments do not match commercial precursors.

        Uses combined BRICS + functional-group grounding so that molecules with
        novel scaffolds but commercial functional groups are not over-penalised.

        Returns a multiplier in [0.7, 1.0] where 0% grounding → 0.7x
        (softened from 0.5x to avoid strangling novel scaffold discovery)
        and 100% grounding → 1.0x.
        """
        return 0.7 + 0.3 * float(np.clip(grounding, 0.0, 1.0))

    def _compute_score(
        self,
        homo_eV: float = -99.0,
        lumo_eV: float = -99.0,
        dielectric_proxy: float = 0.0,
        viscosity_proxy: float = 99.0,
        li_solvation_proxy: float = 0.0,
        ctx: MoleculeContext | None = None,
        quantum_confidence: str = "unknown",
        ea_eV: float | None = None,
    ) -> dict[str, Any]:
        """Compute the multi-objective composite Aurelius Score.

        ADR-2026-06-02: Added quantum_confidence multiplier. Physical
        justification: The TOM fallback's particle-in-a-box model has MAE
        ~1.07 eV on conjugated/novel scaffolds (quantum_confidence="tom_low").
        Without a penalty, the EA can exploit TOM's blind spots by generating
        highly conjugated molecules that score well on paper but are physically
        unreliable. The 0.85x multiplier softens the score for low-confidence
        predictions, biasing selection toward xTB-validated or simple-TOM
        candidates without hard-rejecting novel scaffolds (which may still
        have genuine merit). The multiplier is intentionally mild (0.85 vs.
        0.70) to avoid strangling discovery while imposing epistemic humility.

        Iterates over the declarative ``_OBJECTIVES`` list, applies
        each objective's mathematical transform to the corresponding
        property value, and aggregates into a weighted composite.

        Args:
            homo_eV: Predicted HOMO energy.
            lumo_eV: Predicted LUMO energy.
            dielectric_proxy: Predicted dielectric proxy.
            viscosity_proxy: Predicted viscosity proxy.
            li_solvation_proxy: Predicted Li+ solvation proxy.
            ctx: Pre-parsed MoleculeContext for substructure checks.
            quantum_confidence: Confidence level from quantum backend
                ("xtb", "tom_high", or "tom_low").

        Returns:
            Dict with total_score, is_viable, sub_scores, rejection_reasons.
        """
        raw_values: dict[str, float] = {
            "lumo_eV": lumo_eV,
            "homo_eV": homo_eV,
            "dielectric_proxy": dielectric_proxy,
            "viscosity_proxy": viscosity_proxy,
            "li_solvation_proxy": li_solvation_proxy,
            # ADR-2026-08-10. When the reduction oracle is unavailable the
            # objective must not silently reward or punish: EA_TARGET makes the
            # Gaussian evaluate to its maximum, so the term drops out of the
            # comparison between candidates rather than injecting a fake signal.
            "ea_eV": EA_TARGET if ea_eV is None else ea_eV,
        }

        sub_scores: dict[str, float] = {}
        total_score = 0.0

        # Compute custom SA score once
        sa_score: float = 5.0
        synthesis_depth: int = 3  # Default moderate depth
        grounding: float = 0.0
        synth_complexity: float = 1.0  # Default: maximally complex
        if ctx is not None:
            try:
                sa_score = electrolyte_synthetic_accessibility(ctx)
            except Exception:
                sa_score = 5.0
            try:
                synthesis_depth = brics_retrosynthetic_depth(ctx.mol)
            except Exception:
                synthesis_depth = 3
            grounding = self._grounding_score(ctx)
            try:
                synth_complexity = _synthesizability_complexity(ctx)
            except Exception:
                synth_complexity = 1.0
        raw_values["sa_score"] = sa_score
        raw_values["synth_complexity"] = synth_complexity
        raw_values["grounding"] = grounding

        for obj in _OBJECTIVES:
            score = obj(raw_values[obj.property_key])
            sub_scores[obj.name] = round(score, 4)
            total_score += obj.weight * score

        total_score *= 100.0

        total_score = self._apply_penalties(total_score, lumo_eV, ctx, grounding)
        total_score = float(np.clip(total_score, 0.0, 100.0))

        if quantum_confidence == "tom_low":
            total_score *= 0.85
            total_score = float(np.clip(total_score, 0.0, 100.0))

        is_viable = total_score >= VIABILITY_THRESHOLD

        rejection_reasons = self._build_rejection_reasons(
            total_score, sub_scores, raw_values, is_viable, ctx
        )

        return {
            "total_score": total_score,
            "is_viable": is_viable,
            "sub_scores": sub_scores,
            "sa_score": round(sa_score, 4),
            "synthesizability_complexity": round(synth_complexity, 4),
            "synthesis_depth": synthesis_depth,
            "grounding": round(grounding, 4),
            "rejection_reasons": rejection_reasons,
        }

    @staticmethod
    def _apply_domain_penalty(score: dict[str, Any], t2_result: dict[str, Any] | None) -> dict[str, Any]:
        if t2_result is not None:
            domain_penalty = t2_result.get("domain_penalty", 1.0)
            if domain_penalty < 1.0:
                score["total_score"] *= domain_penalty
                score["domain_penalty_applied"] = domain_penalty
                reason = t2_result.get("domain_reason", "")
                if reason:
                    score.setdefault("rejection_reasons", []).append(
                        f"Domain penalty {domain_penalty:.2f}: {reason}"
                    )
        score["total_score"] = float(np.clip(score["total_score"], 0.0, 100.0))
        score["is_viable"] = score["total_score"] >= VIABILITY_THRESHOLD
        return score

    @staticmethod
    def _apply_penalties(
        total_score: float,
        lumo_eV: float,
        ctx: MoleculeContext | None,
        grounding: float = 1.0,
    ) -> float:
        al_corrosion_penalty = 1.0
        if ctx is None:
            return total_score
        total_score *= AureliusPipeline._check_hydrolytic_instability(ctx.mol)
        total_score *= AureliusPipeline._check_hypofluorite_instability(ctx.mol)
        total_score *= AureliusPipeline._check_building_block_grounding(grounding)
        if lumo_eV > AL_CORROSION_LUMO_THRESHOLD:
            al_corrosion_penalty = AureliusPipeline._check_al_corrosion_risk(ctx.mol)
        total_score *= al_corrosion_penalty
        return total_score

    @staticmethod
    def _build_rejection_reasons(
        total_score: float,
        sub_scores: dict[str, float],
        raw_values: dict[str, float],
        is_viable: bool,
        ctx: MoleculeContext | None,
    ) -> list[str]:
        if is_viable:
            return []
        reasons = []
        for obj in _OBJECTIVES:
            s = sub_scores.get(obj.name, 0.0)
            if s < 0.3:
                value = raw_values.get(obj.property_key, 0.0)
                reasons.append(obj.failure_reason_template.format(value=value))
        if ctx is not None:
            if AureliusPipeline._check_al_corrosion_risk(ctx.mol) < 1.0:
                reasons.append("Al corrosion risk (high-LUMO fluorinated molecule)")
            if AureliusPipeline._check_hypofluorite_instability(ctx.mol) < 1.0:
                reasons.append("hypofluorite (O-F) bond — violently reactive")
        return [
            f"Aurelius Score {total_score:.1f} below threshold: {'; '.join(reasons)}"
        ]

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"

    def screen_smiles(self, smiles: str) -> dict[str, Any]:
        """Convenience: parse a SMILES string then screen it.

        Args:
            smiles: SMILES string.

        Returns:
            Result dict (same as screen_molecule).
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return self._generate_failed_run(smiles, "Invalid SMILES — parsing failed")
        return self.screen_molecule(ctx)

    def screen_batch(
        self,
        contexts: list[MoleculeContext],
    ) -> list[dict[str, Any]]:
        """Screen a batch of pre-parsed molecules through the full pipeline.

        Evaluates all Tier-1-viable molecules in ONE batch oracle call
        (``PropertyOracle.predict_batch_properties``) so closed-form orbitals,
        the reduction axis and the GC proxies are computed once per batch,
        then assembles per-molecule results that are key-identical to
        ``screen_molecule``. Results preserve input order — Tier-1-rejected
        molecules carry a failed run — and any per-molecule assembly failure
        falls back to the scalar ``screen_molecule`` path so a batch failure
        never kills the caller's loop.

        Args:
            contexts: List of MoleculeContext objects.

        Returns:
            List of result dicts, one per input context, in input order.
        """
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialize() first.")
        if not contexts:
            return []

        results, pending = self._filter_tier1(contexts)

        if pending:
            viable_contexts = [ctx for _, ctx, _ in pending]
            try:
                batch = self._oracle.predict_batch_properties(viable_contexts)
            except Exception as exc:
                logger.warning("Batch oracle failed (%s) — falling back to scalar.", exc)
                batch = None
            batch_results = self._assemble_batch_results(pending, batch)
            for j, (i, _, _) in enumerate(pending):
                results[i] = batch_results[j]

        return [r for r in results if r is not None]

    def _filter_tier1(
        self,
        contexts: list[MoleculeContext],
    ) -> tuple[list[dict[str, Any] | None], list[tuple[int, MoleculeContext, dict[str, Any] | None]]]:
        """Tier-1 filtering loop for a batch of contexts.

        Returns ``(results, pending)`` where *results* has failed-run entries
        for Tier-1-rejected molecules (preserving input order) and *pending*
        contains ``(index, context, t1_result)`` for viable molecules.
        """
        results: list[dict[str, Any] | None] = [None] * len(contexts)
        pending: list[tuple[int, MoleculeContext, dict[str, Any] | None]] = []

        for i, ctx in enumerate(contexts):
            t1_result = None
            if self._filter:
                t1_result = self._filter.screen(ctx)
                if not t1_result.get("is_viable", False):
                    results[i] = self._generate_failed_run(
                        ctx.smiles, "Failed Tier 1 Structural Filter"
                    )
                    continue
            pending.append((i, ctx, t1_result))

        return results, pending

    def _assemble_batch_results(
        self,
        pending: list[tuple[int, MoleculeContext, dict[str, Any] | None]],
        batch: dict[str, Any] | None,
    ) -> list[dict[str, Any] | None]:
        """Assembly for‑loop with fallback.

        For each pending molecule uses the batch oracle when available,
        otherwise falls back to ``screen_molecule``.  Any per-molecule
        assembly exception also falls back to scalar.
        """
        results: list[dict[str, Any] | None] = [None] * len(pending)

        for j, (_i, ctx, t1_result) in enumerate(pending):
            try:
                if batch is not None:
                    results[j] = self._assemble_batch_result(ctx, j, batch, t1_result)
                else:
                    results[j] = self.screen_molecule(ctx)
            except Exception as exc:
                logger.debug(
                    "Batch assembly failed for %s (%s) — falling back to scalar.",
                    ctx.smiles, exc,
                )
                results[j] = self.screen_molecule(ctx)

        return results

    def _assemble_batch_result(
        self,
        ctx: MoleculeContext,
        idx: int,
        batch: dict[str, Any],
        t1_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble one per-molecule pipeline result from a batch oracle response.

        Mirrors the scalar ``screen_molecule`` tier-2 defaults and score
        assembly so batch and scalar outputs are key-identical (16 scalar
        oracle keys, ``li_binding_energy_kcal`` default, ``_extract_ea``,
        ``_compute_score`` and ``_apply_domain_penalty``).
        """
        t2: dict[str, Any] = {
            "homo_eV": round(float(batch["homo_eV"][idx]), 4),
            "lumo_eV": round(float(batch["lumo_eV"][idx]), 4),
            "gap_eV": round(float(batch["gap_eV"][idx]), 4),
            "reduction_stability_proxy": batch["reduction_records"][idx],
            "dielectric_proxy": round(float(batch["dielectric_proxy"][idx]), 4),
            "viscosity_proxy": round(float(batch["viscosity_proxy"][idx]), 4),
            "li_solvation_proxy": round(float(batch["li_solvation_proxy"][idx]), 4),
            "conductivity_proxy": round(float(batch["conductivity_proxy"][idx]), 4),
            "domain_applicable": batch["domain_applicable"][idx],
            "domain_reason": batch["domain_reason"][idx],
            "domain_penalty": round(float(batch["domain_penalty"][idx]), 4),
            "quantum_method": batch["quantum_method"][idx],
            "quantum_confidence": batch["quantum_confidence"][idx],
            "sanity_warning": batch["sanity_warning"][idx],
            "conformal_intervals": batch["conformal_intervals"][idx],
            "conformal_confidence": round(float(batch["conformal_confidence"][idx]), 4),
        }
        # Copy ALL oracle result keys, then set defaults for required fields
        # (identical to the scalar screen_molecule tier-2 block).
        t2.setdefault("homo_eV", -99.0)
        t2.setdefault("lumo_eV", -99.0)
        t2.setdefault("gap_eV", 0.0)
        t2.setdefault("dielectric_proxy", 0.0)
        t2.setdefault("viscosity_proxy", 99.0)
        t2.setdefault("li_solvation_proxy", 0.0)
        t2.setdefault("domain_applicable", True)
        t2.setdefault("domain_reason", "")
        t2.setdefault("domain_penalty", 1.0)
        t2.setdefault("quantum_confidence", "unknown")
        t2.setdefault("li_binding_energy_kcal", 0.0)

        quantum_confidence = t2.get("quantum_confidence", "unknown")
        ea_eV = _extract_ea(t2)
        score = self._compute_score(
            t2["homo_eV"],
            t2["lumo_eV"],
            dielectric_proxy=t2["dielectric_proxy"],
            viscosity_proxy=t2["viscosity_proxy"],
            li_solvation_proxy=t2["li_solvation_proxy"],
            ctx=ctx,
            quantum_confidence=quantum_confidence,
            ea_eV=ea_eV,
        )
        score = self._apply_domain_penalty(score, t2)

        result: dict[str, Any] = {"tier2": t2, "score": score}
        if t1_result is not None:
            result["tier1"] = t1_result
        return result

    def save_result(self, result: dict[str, Any], path: str) -> None:
        """Save a screening result to a JSON file.

        Args:
            result: Result dict from screen_molecule or screen_batch.
            path: Path to JSON file.
        """
        import json

        # Handle both old (with total_score) and new (score dict) result formats
        normalized = dict(result)

        # If result has a 'score' dict with 'total_score', extract it
        if "score" in normalized and isinstance(normalized["score"], dict) and "total_score" not in normalized:
            normalized["total_score"] = normalized["score"].get("total_score")

        with open(path, 'w') as f:
            json.dump(normalized, f, indent=2)

    def load_result(self, path: str) -> dict[str, Any]:
        """Load a screening result from a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Loaded result dict with ``total_score`` normalized.
        """
        import json

        with open(path) as f:
            result = json.load(f)

        # Normalize: ensure total_score is at the top level
        if "score" in result and isinstance(result["score"], dict):
            # Extract total_score from the score dict if present
            if "total_score" not in result and "total_score" in result["score"]:
                result["total_score"] = result["score"]["total_score"]
            # Also copy sub-scores to top level for convenience
            for key in ["sa_score", "synthesizability_complexity", "synthesis_depth",
                       "grounding", "rejection_reasons"]:
                if key not in result and key in result["score"]:
                    result[key] = result["score"][key]

        return result
