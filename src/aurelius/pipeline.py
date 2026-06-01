"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. Filter — Quick structural validity (Tier 1) check with LogP and MW gates.
  2. Oracle — Multi-objective property evaluation via hybrid RF/GC + F/P/S
     correction (HOMO, LUMO, Dielectric proxy, Viscosity proxy, SA Score).
  3.    Score — Declarative multi-objective composite with Al corrosion penalty.
     Each objective is defined as an ``Objective`` dataclass (Gaussian or
     Sigmoid), making the scoring logic purely data-driven.

All stages accept a pre-parsed MoleculeContext to enforce single-point parsing.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.constants import (
    AL_CORROSION_LUMO_THRESHOLD,
    AL_CORROSION_MIN_FLUORINE,
    AL_CORROSION_PENALTY_FACTOR,
    CF3_PATTERN as _CF3_PATTERN,
    CARBONYL_F_PATTERN as _CARBONYL_F_PATTERN,
    DIELECTRIC_SIGMOID_STEEPNESS,
    DIELECTRIC_TARGET,
    HOMO_SIGMOID_STEEPNESS,
    HOMO_THRESHOLD,
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_PATTERNS,
    LI_SOLVATION_SIGMA,
    LI_SOLVATION_TARGET,
    LUMO_SIGMA,
    LUMO_TARGET,
    SA_SIGMOID_STEEPNESS,
    SA_THRESHOLD,
    SCORE_WEIGHT_AL_CORROSION,
    SCORE_WEIGHT_DIELECTRIC,
    SCORE_WEIGHT_HOMO,
    SCORE_WEIGHT_LI_SOLVATION,
    SCORE_WEIGHT_LUMO,
    SCORE_WEIGHT_SA,
    SCORE_WEIGHT_VISCOSITY,
    SULFONYL_F_PATTERN as _SULFONYL_F_PATTERN,
    VIABILITY_THRESHOLD,
    VISCOSITY_SIGMOID_STEEPNESS,
    VISCOSITY_THRESHOLD,
)
from aurelius.scoring.oracle import PropertyOracle
from aurelius.screening.tier1 import Filter
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import electrolyte_synthetic_accessibility

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """A single scoring objective with a mathematical transformation.

    Each objective defines how a raw property value is converted to a
    sub-score via a Gaussian or Sigmoid function, then weighted in the
    final composite.  The ``failure_reason_template`` is used to generate
    human-readable rejection reasons dynamically (e.g. "LUMO={value:.3f}eV
    (poor SEI formation)").
    """

    name: str
    property_key: str
    weight: float
    function: str  # 'gaussian' or 'sigmoid'
    target: float
    sigma: float | None = None
    steepness: float | None = None
    higher_is_better: bool = True
    failure_reason_template: str = "{name}={value:.3f} (below threshold)"

    def __call__(self, value: float) -> float:
        if self.function == "gaussian":
            if self.sigma is None or self.sigma <= 0.0:
                raise ValueError(f"Gaussian objective '{self.name}' requires sigma > 0")
            return float(np.exp(-0.5 * ((value - self.target) / self.sigma) ** 2))
        if self.function == "sigmoid":
            if self.steepness is None:
                raise ValueError(f"Sigmoid objective '{self.name}' requires steepness")
            if self.higher_is_better:
                return float(1.0 / (1.0 + np.exp(-self.steepness * (value - self.target))))
            return float(1.0 / (1.0 + np.exp(self.steepness * (value - self.target))))
        raise ValueError(f"Unknown objective function: '{self.function}'")


# Declarative scoring configuration: add/remove objectives here without
# touching boilerplate math.
_OBJECTIVES: list[Objective] = [
    Objective("lumo_reward", "lumo_eV", SCORE_WEIGHT_LUMO, "gaussian", LUMO_TARGET, sigma=LUMO_SIGMA,
              failure_reason_template="LUMO={value:.3f}eV (poor SEI formation)"),
    Objective("homo_penalty", "homo_eV", SCORE_WEIGHT_HOMO, "sigmoid", HOMO_THRESHOLD,
              steepness=HOMO_SIGMOID_STEEPNESS, higher_is_better=False,
              failure_reason_template="HOMO={value:.3f}eV (oxidative instability)"),
    Objective("dielectric_reward", "dielectric_proxy", SCORE_WEIGHT_DIELECTRIC, "sigmoid", DIELECTRIC_TARGET,
              steepness=DIELECTRIC_SIGMOID_STEEPNESS,
              failure_reason_template="dielectric_proxy={value:.3f} (poor salt dissolution)"),
    Objective("viscosity_penalty", "viscosity_proxy", SCORE_WEIGHT_VISCOSITY, "sigmoid", VISCOSITY_THRESHOLD,
              steepness=VISCOSITY_SIGMOID_STEEPNESS, higher_is_better=False,
              failure_reason_template="viscosity_proxy={value:.3f} (poor ion mobility)"),
    Objective("li_solvation_reward", "li_solvation_proxy", SCORE_WEIGHT_LI_SOLVATION, "gaussian", LI_SOLVATION_TARGET,
              sigma=LI_SOLVATION_SIGMA,
              failure_reason_template="li_solvation_proxy={value:.3f} (poor Li+ binding)"),
    Objective("sa_penalty", "sa_score", SCORE_WEIGHT_SA, "sigmoid", SA_THRESHOLD,
              steepness=SA_SIGMOID_STEEPNESS, higher_is_better=False,
              failure_reason_template="SA score={value:.2f} (hard to synthesize)"),
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
    ) -> None:
        self._filter: Filter | None = None
        self._use_real_models = use_real_models
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

        self._oracle = PropertyOracle()
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

        smiles = ctx.smiles
        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        t1_result = None
        if self._filter:
            t1_start = time.perf_counter()
            t1_result = self._filter.screen(ctx)
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
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter")
        else:
            results = {}
            tier_timings = {}

        t2_result = None
        homo_eV = -99.0
        lumo_eV = -99.0
        dielectric_proxy = 0.0
        viscosity_proxy = 99.0
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(ctx)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            homo_eV = oracle_result.get("homo_eV", -99.0)
            lumo_eV = oracle_result.get("lumo_eV", -99.0)
            dielectric_proxy = oracle_result.get("dielectric_proxy", 0.0)
            viscosity_proxy = oracle_result.get("viscosity_proxy", 99.0)
            li_solvation_proxy = oracle_result.get("li_solvation_proxy", 0.0)

            t2_result = {
                "homo_eV": homo_eV,
                "lumo_eV": lumo_eV,
                "gap_eV": oracle_result.get("gap_eV", 0.0),
                "dielectric_proxy": dielectric_proxy,
                "viscosity_proxy": viscosity_proxy,
                "li_solvation_proxy": li_solvation_proxy,
                "domain_applicable": oracle_result.get("domain_applicable", True),
                "domain_reason": oracle_result.get("domain_reason", ""),
            }
            results["tier2"] = t2_result
            logger.info(
                "Oracle Result: %s -> HOMO=%.3f LUMO=%.3f Dielectric=%.3f Viscosity=%.3f LiSolv=%.3f",
                smiles, homo_eV, lumo_eV, dielectric_proxy, viscosity_proxy, li_solvation_proxy,
            )

        score = self._compute_score(
            homo_eV, lumo_eV,
            dielectric_proxy=dielectric_proxy,
            viscosity_proxy=viscosity_proxy,
            li_solvation_proxy=li_solvation_proxy,
            ctx=ctx,
        )
        results["score"] = score

        logger.debug("Scorecard:\n%s", self._format_score(score))

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            logger.info("Performance: total=%.1fms | %s", total_ms, " | ".join(timing_lines))

        return results

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
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)

        # Count CF3 groups
        n_cf3 = len(mol.GetSubstructMatches(_CF3_PATTERN))

        # Count fluorine adjacent to carbonyl or sulfonyl
        n_f_ewg = len(mol.GetSubstructMatches(_CARBONYL_F_PATTERN))
        n_f_ewg += len(mol.GetSubstructMatches(_SULFONYL_F_PATTERN))

        if n_f >= AL_CORROSION_MIN_FLUORINE and (n_cf3 >= 1 or n_f_ewg >= 1):
            return AL_CORROSION_PENALTY_FACTOR
        return 1.0

    def _compute_score(
        self,
        homo_eV: float = -99.0,
        lumo_eV: float = -99.0,
        dielectric_proxy: float = 0.0,
        viscosity_proxy: float = 99.0,
        li_solvation_proxy: float = 0.0,
        ctx: MoleculeContext | None = None,
    ) -> dict[str, Any]:
        """Compute the multi-objective composite Aurelius Score.

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

        Returns:
            Dict with total_score, is_viable, sub_scores, rejection_reasons.
        """
        raw_values: dict[str, float] = {
            "lumo_eV": lumo_eV,
            "homo_eV": homo_eV,
            "dielectric_proxy": dielectric_proxy,
            "viscosity_proxy": viscosity_proxy,
            "li_solvation_proxy": li_solvation_proxy,
        }

        sub_scores: dict[str, float] = {}
        total_score = 0.0

        # Compute custom SA score once
        sa_score: float = 5.0
        if ctx is not None:
            try:
                sa_score = electrolyte_synthetic_accessibility(ctx)
            except Exception:
                sa_score = 5.0

        for obj in _OBJECTIVES:
            if obj.name == "sa_penalty":
                score = obj(sa_score)
                raw_values["sa_score"] = sa_score
            else:
                score = obj(raw_values[obj.property_key])

            sub_scores[obj.name] = round(score, 4)
            total_score += obj.weight * score

        # Al corrosion penalty
        al_corrosion_penalty = 1.0
        if ctx is not None and lumo_eV > AL_CORROSION_LUMO_THRESHOLD:
            al_corrosion_penalty = self._check_al_corrosion_risk(ctx.mol)
        sub_scores["al_corrosion_penalty"] = round(al_corrosion_penalty, 4)
        total_score += SCORE_WEIGHT_AL_CORROSION * (1.0 - al_corrosion_penalty)

        total_score *= 100.0

        # Multiplicative penalties
        if ctx is not None:
            hydro_penalty = self._check_hydrolytic_instability(ctx.mol)
            total_score *= hydro_penalty
            total_score *= al_corrosion_penalty

        total_score = float(np.clip(total_score, 0.0, 100.0))

        is_viable = total_score >= VIABILITY_THRESHOLD

        # Build rejection reasons (declarative — no hardcoded if/elif chain)
        rejection_reasons: list[str] = []
        if not is_viable:
            reasons = []
            for obj in _OBJECTIVES:
                s = sub_scores.get(obj.name, 0.0)
                if s < 0.3:
                    value = raw_values.get(obj.property_key, sa_score if obj.name == "sa_penalty" else 0.0)
                    reasons.append(obj.failure_reason_template.format(value=value))
            if al_corrosion_penalty < 1.0 and lumo_eV > AL_CORROSION_LUMO_THRESHOLD:
                reasons.append("Al corrosion risk (high-LUMO fluorinated molecule)")
            rejection_reasons.append(
                f"Aurelius Score {total_score:.1f} below threshold: {'; '.join(reasons)}"
            )

        return {
            "total_score": total_score,
            "is_viable": is_viable,
            "sub_scores": sub_scores,
            "sa_score": round(sa_score, 4),
            "rejection_reasons": rejection_reasons,
        }

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
        n_workers: int = 1,
    ) -> list[dict[str, Any]]:
        """Screen a batch of pre-parsed molecules through the full pipeline.

        Args:
            contexts: List of MoleculeContext objects.
            n_workers: Number of parallel workers.

        Returns:
            List of result dicts.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(ctx) for ctx in contexts]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, ctx): i
                for i, ctx in enumerate(contexts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(contexts))]
