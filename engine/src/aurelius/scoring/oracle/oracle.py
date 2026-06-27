"""PropertyOracle — Hybrid fragment-additivity + quantum chemistry oracle.

Usage:
    from aurelius.scoring.oracle.oracle import PropertyOracle
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = oracle.evaluate(ctx)
    print(result["homo_eV"])            # e.g. -7.6 (quantum)
    print(result["li_solvation_proxy"])  # e.g. 2.3 (GC)
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.cache import CacheBackend, DictCache
from aurelius.constants import (
    SEI_LUMO_LOWER,
    SEI_LUMO_UPPER,
    SEI_MOTIF_PENALTY_FACTOR,
    STABLE_SEI_MOTIFS,
)
from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    _UQ_PENALTY,
    _UQ_THRESHOLD_FRACTION,
    BasePropertyModel,
    ElectrolytePack,
    GcUqEnsemble,
    compute_gc_domain_penalty,
)
from aurelius.scoring.oracle.quantum import (
    QuantumOracle,
    compute_quantum_domain_penalty,
)
from aurelius.scoring.oracle.surrogate import SurrogateQuantumOracle
from aurelius.types import (
    DomainResult,
    EvaluationResult,
    GcEvaluation,
    GcResult,
    MoleculeContext,
    OodResult,
    QuantumEvaluation,
    QuantumResult,
    SeiResult,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


def _evaluate_sei_motif(ctx: MoleculeContext, lumo: float) -> tuple[float, str]:
    """Check if a molecule has known stable SEI-forming motifs.

    Physical basis: A molecule with LUMO in the SEI formation window
    (> SEI_LUMO_LOWER and < SEI_LUMO_UPPER) is thermodynamically capable
    of reductive decomposition to form an SEI. However, without known
    stable SEI-forming functional groups (e.g., cyclic carbonates, CF3,
    sultones), the resulting SEI may be unstable or poorly passivating.
    This check penalises molecules in the SEI window that lack these motifs.

    Returns:
        (penalty_multiplier, reason_string)
        Multiplier in [SEI_MOTIF_PENALTY_FACTOR, 1.0]; 1.0 = has stable motif or
        LUMO outside SEI window.
    """
    if not (SEI_LUMO_LOWER < lumo < SEI_LUMO_UPPER):
        return 1.0, ""
    mol = ctx.mol
    for pattern in STABLE_SEI_MOTIFS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return 1.0, ""
    return SEI_MOTIF_PENALTY_FACTOR, "Lacks stable SEI-forming motif"


# ---------------------------------------------------------------------------
# Pure function helpers — stateless, all dependencies passed as args
# ---------------------------------------------------------------------------


def _run_surrogate_check(
    ctx: MoleculeContext,
    surrogate: SurrogateQuantumOracle | None,
) -> tuple[float, float, float, bool]:
    """Pure surrogate pre-filter. Returns (penalty, s_homo, s_lumo, skip_quantum).

    Two gates control whether the surrogate prediction is trusted:
      1. Structural novelty gate: if the molecule's max Tanimoto similarity
         to the calibration pool is below ``similarity_threshold``, the
         surrogate is considered untrustworthy and quantum is forced.
      2. HOMO threshold gate: if surrogate predicts HOMO > -5.0 eV,
         skip quantum (classic fast-skip path).
    """
    if surrogate is None:
        return 1.0, -99.0, 99.0, False
    try:
        if surrogate.is_structurally_novel(ctx):
            logger.info(
                "Surrogate: molecule structurally novel (max Tanimoto < %.2f) — "
                "forcing quantum oracle",
                surrogate.similarity_threshold,
            )
            return 1.0, -99.0, 99.0, False
    except Exception:
        pass

    try:
        s_homo, s_lumo, _uncertainty = surrogate.predict(ctx)
        penalty = surrogate.compute_penalty(s_homo, _uncertainty)
        if penalty < 1.0:
            logger.info(
                "Surrogate: HOMO=%.3f eV > %.1f threshold — skipping quantum oracle",
                s_homo, -5.0,
            )
            return penalty, s_homo, s_lumo, True
        return penalty, s_homo, s_lumo, False
    except Exception:
        logger.debug("Surrogate pre-filter unavailable")
        return 1.0, -99.0, 99.0, False


def _compute_quantum_result(
    ctx: MoleculeContext,
    quantum: QuantumOracle,
    skip_quantum: bool,
    s_homo: float,
    s_lumo: float,
) -> QuantumResult:
    """Pure quantum evaluation. Returns a ``QuantumResult`` NamedTuple."""
    if skip_quantum:
        gap = s_lumo - s_homo
        return QuantumResult(
            homo_eV=s_homo,
            lumo_eV=s_lumo,
            gap_eV=gap,
            method="surrogate",
            confidence="surrogate",
        )
    qr = quantum.evaluate(ctx.mol)
    gap = qr["lumo_eV"] - qr["homo_eV"]
    confidence_val = qr.get("quantum_confidence", "unknown")
    confidence = str(confidence_val) if not isinstance(confidence_val, str) else confidence_val
    return QuantumResult(
        homo_eV=qr["homo_eV"],
        lumo_eV=qr["lumo_eV"],
        gap_eV=gap,
        method=quantum.method,
        confidence=confidence,
    )


def _compute_gc_properties(
    ctx: MoleculeContext,
    property_pack: BasePropertyModel,
) -> GcResult:
    """Pure GC property computation. Delegates to the property pack.

    Returns a ``GcResult`` NamedTuple with gc_props, uq_penalty, diel_std, visc_std.
    """
    gc_props = property_pack.predict_all(ctx)
    return GcResult(
        gc_props=gc_props,
        uq_penalty=1.0,
        diel_std=0.0,
        visc_std=0.0,
    )


def _compute_uq_penalty(
    ctx: MoleculeContext,
    gc_uq: GcUqEnsemble | None,
) -> tuple[float, float, float]:
    """Pure GC uncertainty penalty. Returns (uq_penalty, diel_std, visc_std)."""
    if gc_uq is None:
        return 1.0, 0.0, 0.0
    try:
        _diel_mean, diel_std, _ = gc_uq.predict_dielectric(ctx)
        _visc_mean, visc_std, _ = gc_uq.predict_viscosity(ctx)
        n_flags = 0
        if diel_std > max(1.0, abs(_diel_mean)) * _UQ_THRESHOLD_FRACTION:
            n_flags += 1
        if visc_std > max(1.0, abs(_visc_mean)) * _UQ_THRESHOLD_FRACTION:
            n_flags += 1
        if n_flags > 0:
            return _UQ_PENALTY ** n_flags, diel_std, visc_std
        return 1.0, diel_std, visc_std
    except Exception:
        logger.debug("GC UQ unavailable")
    return 1.0, 0.0, 0.0


def _compute_ood_penalty(
    ctx: MoleculeContext,
    gc_uq: GcUqEnsemble | None,
) -> OodResult:
    """Pure centroid-based OOD penalty. Returns an ``OodResult`` NamedTuple."""
    if gc_uq is None:
        return OodResult(ood_penalty=1.0, ood_reason="")
    try:
        ood_dist, is_ood = gc_uq.compute_domain_distance(ctx)
        if is_ood:
            return OodResult(ood_penalty=0.9, ood_reason=f"OOD centroid (dist={ood_dist:.3f})")
    except Exception:
        pass
    return OodResult(ood_penalty=1.0, ood_reason="")


def _build_domain(
    ctx: MoleculeContext,
    skip_quantum: bool,
    surrogate_penalty: float,
    s_homo: float,
    uq_penalty: float,
    gc_uq: GcUqEnsemble | None,
) -> DomainResult:
    """Pure domain penalty builder.

    Combines quantum DoA, GC DoA, surrogate, UQ variance, and
    centroid-based OOD penalties into a single domain penalty.

    Returns a ``DomainResult`` NamedTuple.
    """
    q_penalty, q_reason = (1.0, "skipped — surrogate") if skip_quantum else compute_quantum_domain_penalty(ctx)
    gc_penalty, gc_reason = compute_gc_domain_penalty(ctx)
    ood_penalty, ood_reason = _compute_ood_penalty(ctx, gc_uq)
    domain_penalty = min(q_penalty, gc_penalty, surrogate_penalty, uq_penalty, ood_penalty)

    reasons: list[str] = []
    if q_penalty < 1.0 and not skip_quantum:
        reasons.append(f"quantum: {q_reason}")
    if gc_penalty < 1.0:
        reasons.append(f"GC: {gc_reason}")
    if surrogate_penalty < 1.0:
        reasons.append(f"surrogate: HOMO={s_homo:.2f} eV > threshold")
    if uq_penalty < 1.0:
        reasons.append("High UQ Variance")
    if ood_penalty < 1.0:
        reasons.append(ood_reason)
    return DomainResult(
        domain_penalty=domain_penalty,
        domain_reason_str="; ".join(reasons) if reasons else _DATA_SOURCE,
        domain_applicable=domain_penalty >= 0.85,
    )


def _apply_sei_penalty(
    ctx: MoleculeContext,
    lumo: float,
    domain_penalty: float,
    domain_reason_str: str,
) -> SeiResult:
    """Pure SEI motif penalty integration.

    Returns a ``SeiResult`` NamedTuple with sei_penalty, sei_reason.
    """
    sei_penalty, sei_reason = _evaluate_sei_motif(ctx, lumo)
    if sei_penalty >= 1.0:
        return SeiResult(
            sei_penalty=domain_penalty,
            sei_reason=domain_reason_str,
            domain_applicable=domain_penalty >= 0.85,
        )
    domain_penalty *= sei_penalty
    if domain_reason_str and domain_reason_str != _DATA_SOURCE:
        domain_reason_str = domain_reason_str + "; " + sei_reason
    else:
        domain_reason_str = sei_reason
    return SeiResult(
        sei_penalty=domain_penalty,
        sei_reason=domain_reason_str,
        domain_applicable=domain_penalty >= 0.85,
    )


def _assemble_result(
    homo: float, lumo: float, gap: float,
    domain_penalty: float, domain_reason_str: str, domain_applicable: bool,
    quantum_method: str, quantum_confidence_val: str,
    skip_quantum: bool,
    diel_std: float = 0.0,
    visc_std: float = 0.0,
    dielectric: float = 0.0,
    viscosity: float = 0.0,
    li_solvation: float = 0.0,
    ced: float = 0.0,
    li_dissociation: float = 0.0,
    hydrolysis_risk: float = 0.0,
) -> EvaluationResult:
    """Pure result assembly into the final ``EvaluationResult`` NamedTuple.

    Convert to dict via ``._asdict()`` for JSON serialization.
    """
    uncertainty_flag = (
        diel_std > abs(dielectric) * _UQ_THRESHOLD_FRACTION
        or visc_std > abs(viscosity) * _UQ_THRESHOLD_FRACTION
    )
    diel_norm = diel_std / max(abs(dielectric), 1.0)
    visc_norm = visc_std / max(abs(viscosity), 1.0)
    uncertainty_score = round((diel_norm + visc_norm) / 2.0, 4)

    return EvaluationResult(
        homo_eV=round(homo, 4),
        lumo_eV=round(lumo, 4),
        gap_eV=round(gap, 4),
        domain_applicable=domain_applicable,
        domain_drift_risk=domain_penalty < 0.85,
        domain_reason=domain_reason_str,
        domain_penalty=round(domain_penalty, 4),
        quantum_method=quantum_method,
        quantum_confidence=quantum_confidence_val,
        uncertainty_flag=uncertainty_flag,
        uncertainty_score=uncertainty_score,
        dielectric_proxy=round(dielectric, 4),
        viscosity_proxy=round(viscosity, 4),
        li_solvation_proxy=round(li_solvation, 4),
        ced_proxy=round(ced, 4),
        li_dissociation_proxy=round(li_dissociation, 4),
        hydrolysis_risk_proxy=round(hydrolysis_risk, 4),
        surrogate_skipped=skip_quantum,
    )


class PropertyOracle:
    """Multi-objective property oracle with a hybrid physics model.

    Architecture:
      - HOMO / LUMO / Gap: QuantumOracle (xTB preferred, TOM fallback)
      - Dielectric proxy: GC fragment-additivity + TPSA-based cap
      - Viscosity proxy: GC fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy: GC fragment-additivity
    """

    def __init__(
        self,
        use_xtb: bool = True,
        use_surrogate: bool = True,
        use_gc_uq: bool = True,
        property_pack: BasePropertyModel | None = None,
        l2_cache: CacheBackend | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)
        self._use_surrogate = use_surrogate
        self._surrogate: SurrogateQuantumOracle | None = (
            SurrogateQuantumOracle() if use_surrogate else None
        )
        self._use_gc_uq = use_gc_uq
        self._gc_uq: GcUqEnsemble | None = GcUqEnsemble() if use_gc_uq else None
        self._property_pack: BasePropertyModel = property_pack or ElectrolytePack()
        self._cache: dict[str, dict[str, Any]] = {}
        if redis_url is not None and l2_cache is None:
            from aurelius.cache.redis_cache import RedisCache
            self._disk_cache = RedisCache(url=redis_url)  # type: ignore[assignment]
        else:
            self._disk_cache = l2_cache or DictCache()  # type: ignore[assignment]
        self._n_surrogate_skips = 0

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    @property
    def property_pack(self) -> BasePropertyModel:
        return self._property_pack

    def _run_surrogate(self, ctx: MoleculeContext) -> tuple[float, float, float, bool]:
        """Run surrogate pre-filter. Delegates to ``_run_surrogate_check``."""
        penalty, s_homo, s_lumo, skip = _run_surrogate_check(ctx, self._surrogate)
        if penalty < 1.0:
            self._n_surrogate_skips += 1
        return penalty, s_homo, s_lumo, skip

    def _compute_quantum(self, ctx: MoleculeContext, skip_quantum: bool, s_homo: float, s_lumo: float) -> QuantumResult:
        """Compute or skip quantum evaluation. Delegates to ``_compute_quantum_result``.

        Returns a ``QuantumResult`` NamedTuple with homo_eV, lumo_eV, gap_eV, method, confidence.
        """
        r = _compute_quantum_result(ctx, self._quantum, skip_quantum, s_homo, s_lumo)
        return QuantumResult(
            homo_eV=r.homo_eV,
            lumo_eV=r.lumo_eV,
            gap_eV=r.gap_eV,
            method=r.method,
            confidence=r.confidence,
        )

    def _compute_uq_penalty(self, ctx: MoleculeContext) -> tuple[float, float, float]:
        """Compute GC uncertainty penalty. Delegates to ``_compute_uq_penalty``.

        Returns (uq_penalty, diel_std, visc_std).
        """
        result = _compute_uq_penalty(ctx, self._gc_uq)
        return result

    def _compute_gc_properties(self, ctx: MoleculeContext) -> GcResult:
        """Compute all GC-based bulk property proxies. Delegates to pure ``_compute_gc_properties``.

        Returns a ``GcResult`` NamedTuple.
        """
        result = _compute_gc_properties(ctx, self._property_pack)
        return GcResult(
            gc_props=result.gc_props,
            uq_penalty=result.uq_penalty,
            diel_std=result.diel_std,
            visc_std=result.visc_std,
        )

    def _evaluate_quantum(self, ctx: MoleculeContext) -> QuantumEvaluation:
        """Evaluate quantum properties (HOMO/LUMO/gap) via surrogate or xTB/TOM.

        Bundles the surrogate pre-filter and quantum computation into one call.
        Returns a ``QuantumEvaluation`` NamedTuple with quantum results plus surrogate metadata.
        """
        surrogate_penalty, s_homo, s_lumo, skip_quantum = self._run_surrogate(ctx)
        homo, lumo, gap, quantum_method, quantum_confidence_val = self._compute_quantum(
            ctx, skip_quantum=skip_quantum, s_homo=s_homo, s_lumo=s_lumo,
        )
        return QuantumEvaluation(
            homo_eV=homo,
            lumo_eV=lumo,
            gap_eV=gap,
            surrogate_penalty=surrogate_penalty,
            s_homo=s_homo,
            skip_quantum=skip_quantum,
            quantum_method=quantum_method,
            quantum_confidence_val=quantum_confidence_val,
        )

    def _evaluate_gc(self, ctx: MoleculeContext) -> GcEvaluation:
        """Evaluate GC bulk properties and UQ penalty.

        Bundles GC property prediction and uncertainty quantification.
        Returns a ``GcEvaluation`` NamedTuple.
        """
        gc_props = self._compute_gc_properties(ctx)
        uq_penalty, diel_std, visc_std = self._compute_uq_penalty(ctx)
        return GcEvaluation(
            gc_props=gc_props.gc_props,
            uq_penalty=uq_penalty,
            diel_std=diel_std,
            visc_std=visc_std,
        )

    def _compute_ood_penalty(self, ctx: MoleculeContext) -> OodResult:
        """Compute centroid-based OOD penalty. Delegates to pure ``_compute_ood_penalty``.

        Returns an ``OodResult`` NamedTuple.
        """
        result = _compute_ood_penalty(ctx, self._gc_uq)
        return OodResult(
            ood_penalty=result.ood_penalty,
            ood_reason=result.ood_reason,
        )

    def _build_domain(self, ctx: MoleculeContext, skip_quantum: bool, surrogate_penalty: float, s_homo: float, uq_penalty: float) -> DomainResult:
        """Build domain penalty. Delegates to pure ``_build_domain``.

        Returns a ``DomainResult`` NamedTuple.
        """
        result = _build_domain(ctx, skip_quantum, surrogate_penalty, s_homo, uq_penalty, self._gc_uq)
        return DomainResult(
            domain_penalty=result.domain_penalty,
            domain_reason_str=result.domain_reason_str,
            domain_applicable=result.domain_applicable,
        )

    def _apply_sei_penalty(self, ctx: MoleculeContext, lumo: float, domain_penalty: float, domain_reason_str: str) -> SeiResult:
        """Apply SEI motif penalty. Delegates to pure ``_apply_sei_penalty``.

        Returns a ``SeiResult`` NamedTuple.
        """
        result = _apply_sei_penalty(ctx, lumo, domain_penalty, domain_reason_str)
        return SeiResult(
            sei_penalty=result.sei_penalty,
            sei_reason=result.sei_reason,
            domain_applicable=result.domain_applicable,
        )

    def _assemble_result(
        self,
        homo: float, lumo: float, gap: float,
        domain_penalty: float, domain_reason_str: str, domain_applicable: bool,
        quantum_method: str, quantum_confidence_val: str,
        skip_quantum: bool,
        diel_std: float = 0.0,
        visc_std: float = 0.0,
        dielectric: float = 0.0,
        viscosity: float = 0.0,
        li_solvation: float = 0.0,
        ced: float = 0.0,
        li_dissociation: float = 0.0,
        hydrolysis_risk: float = 0.0,
    ) -> EvaluationResult:
        """Assemble the final evaluation result NamedTuple. Delegates to pure ``_assemble_result``.

        Returns an ``EvaluationResult`` NamedTuple.
        Convert to dict via ``._asdict()`` for JSON serialization.
        """
        result = _assemble_result(
            homo=homo,
            lumo=lumo,
            gap=gap,
            domain_penalty=domain_penalty,
            domain_reason_str=domain_reason_str,
            domain_applicable=domain_applicable,
            quantum_method=quantum_method,
            quantum_confidence_val=quantum_confidence_val,
            skip_quantum=skip_quantum,
            diel_std=diel_std,
            visc_std=visc_std,
            dielectric=dielectric,
            viscosity=viscosity,
            li_solvation=li_solvation,
            ced=ced,
            li_dissociation=li_dissociation,
            hydrolysis_risk=hydrolysis_risk,
        )
        return result

    def _cache_key(self, smiles: str) -> str:
        return f"{smiles}::{self._property_pack.name}"

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )
        smiles = ctx.smiles
        key = self._cache_key(smiles)
        if key in self._cache:
            return self._cache[key]
        cached = self._disk_cache.get(key)
        if cached is not None:
            self._cache[key] = cached
            return cached
        if key != smiles:
            cached = self._disk_cache.get(smiles)
            if cached is not None:
                self._cache[key] = cached
                return cached

        q = self._evaluate_quantum(ctx)
        g = self._evaluate_gc(ctx)
        domain_penalty, domain_reason_str, domain_applicable = self._build_domain(
            ctx, q.skip_quantum, q.surrogate_penalty, q.s_homo, g.uq_penalty,
        )
        sei_result = self._apply_sei_penalty(
            ctx, q.lumo_eV, domain_penalty, domain_reason_str,
        )
        domain_penalty = sei_result.sei_penalty
        domain_reason_str = sei_result.sei_reason
        domain_applicable = sei_result.domain_applicable

        if domain_penalty < 0.85:
            logger.warning(
                "Warning: Molecule is outside the calibrated domain of the current kernel. "
                "Consider retuning via Aurelius Certification Lab. "
                "(smiles=%s, domain_penalty=%.4f, reason=%s)",
                smiles, domain_penalty, domain_reason_str,
            )

        result = self._assemble_result(
            homo=q.homo_eV,
            lumo=q.lumo_eV,
            gap=q.gap_eV,
            domain_penalty=domain_penalty,
            domain_reason_str=domain_reason_str,
            domain_applicable=domain_applicable,
            quantum_method=q.quantum_method,
            quantum_confidence_val=q.quantum_confidence_val,
            skip_quantum=q.skip_quantum,
            diel_std=g.diel_std,
            visc_std=g.visc_std,
            dielectric=g.gc_props.get("dielectric_proxy", 0.0),
            viscosity=g.gc_props.get("viscosity_proxy", 0.0),
            li_solvation=g.gc_props.get("li_solvation_proxy", 0.0),
            ced=g.gc_props.get("ced_proxy", 0.0),
            li_dissociation=g.gc_props.get("li_dissociation_proxy", 0.0),
            hydrolysis_risk=g.gc_props.get("hydrolysis_risk_proxy", 0.0),
        )
        self._cache[key] = result._asdict()
        self._disk_cache[key] = result._asdict()
        return result._asdict()

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        self._cache.clear()
        for key in self._disk_cache:
            self._cache[key] = self._disk_cache[key]
        import joblib
        payload: dict[str, Any] = {
            "cache": self._cache,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: cache saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        try:
            import joblib
            payload = joblib.load(path)
        except (FileNotFoundError, Exception) as exc:
            logger.debug("PropertyOracle: no cached oracle at %s (%s)", path, exc)
            return False

        loaded_cache = payload.get("cache")
        if loaded_cache is not None:
            self._cache.update(loaded_cache)
            for key, value in loaded_cache.items():
                self._disk_cache[key] = value
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        self._cache.clear()
        self._disk_cache.clear()
        self._quantum.clear_cache()
