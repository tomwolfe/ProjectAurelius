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
) -> dict[str, Any]:
    """Pure quantum evaluation. Returns dict with keys
    ``homo_eV``, ``lumo_eV``, ``gap_eV``, ``method``, ``confidence``.
    """
    if skip_quantum:
        gap = s_lumo - s_homo
        return {
            "homo_eV": s_homo,
            "lumo_eV": s_lumo,
            "gap_eV": gap,
            "method": "surrogate",
            "confidence": "surrogate",
        }
    qr = quantum.evaluate(ctx.mol)
    gap = qr["lumo_eV"] - qr["homo_eV"]
    return {
        "homo_eV": qr["homo_eV"],
        "lumo_eV": qr["lumo_eV"],
        "gap_eV": gap,
        "method": quantum.method,
        "confidence": qr.get("quantum_confidence", "unknown"),
    }


def _compute_gc_properties(
    ctx: MoleculeContext,
    property_pack: BasePropertyModel,
) -> dict[str, float]:
    """Pure GC property computation. Delegates to the property pack."""
    return property_pack.predict_all(ctx)


def _compute_uq_penalty(
    ctx: MoleculeContext,
    gc_uq: GcUqEnsemble | None,
) -> tuple[float, float, float]:
    """Pure GC uncertainty penalty. Returns (penalty, diel_std, visc_std)."""
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
) -> tuple[float, str]:
    """Pure centroid-based OOD penalty."""
    if gc_uq is None:
        return 1.0, ""
    try:
        ood_dist, is_ood = gc_uq.compute_domain_distance(ctx)
        if is_ood:
            return 0.9, f"OOD centroid (dist={ood_dist:.3f})"
    except Exception:
        pass
    return 1.0, ""


def _build_domain(
    ctx: MoleculeContext,
    skip_quantum: bool,
    surrogate_penalty: float,
    s_homo: float,
    uq_penalty: float,
    gc_uq: GcUqEnsemble | None,
) -> tuple[float, str, bool]:
    """Pure domain penalty builder.

    Combines quantum DoA, GC DoA, surrogate, UQ variance, and
    centroid-based OOD penalties into a single domain penalty.
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
    return domain_penalty, "; ".join(reasons) if reasons else _DATA_SOURCE, domain_penalty >= 0.85


def _apply_sei_penalty(
    ctx: MoleculeContext,
    lumo: float,
    domain_penalty: float,
    domain_reason_str: str,
) -> tuple[float, str, bool]:
    """Pure SEI motif penalty integration."""
    sei_penalty, sei_reason = _evaluate_sei_motif(ctx, lumo)
    if sei_penalty >= 1.0:
        return domain_penalty, domain_reason_str, domain_penalty >= 0.85
    domain_penalty *= sei_penalty
    if domain_reason_str and domain_reason_str != _DATA_SOURCE:
        domain_reason_str += "; " + sei_reason
    else:
        domain_reason_str = sei_reason
    return domain_penalty, domain_reason_str, domain_penalty >= 0.85


def _assemble_result(
    smiles: str,
    homo: float, lumo: float, gap: float,
    gc_props: dict[str, float],
    domain_penalty: float, domain_reason_str: str, domain_applicable: bool,
    quantum_method: str, quantum_confidence_val: str,
    skip_quantum: bool,
    has_surrogate: bool = False,
    diel_std: float = 0.0,
    visc_std: float = 0.0,
) -> dict[str, Any]:
    """Pure result assembly into the final evaluation dict."""
    result: dict[str, Any] = {
        "homo_eV": round(homo, 4),
        "lumo_eV": round(lumo, 4),
        "gap_eV": round(gap, 4),
        "domain_applicable": domain_applicable,
        "domain_drift_risk": domain_penalty < 0.85,
        "domain_reason": domain_reason_str,
        "domain_penalty": round(domain_penalty, 4),
        "quantum_method": quantum_method,
        "quantum_confidence": quantum_confidence_val,
    }
    for key, value in gc_props.items():
        result[key] = round(value, 4)

    dielectric = gc_props.get("dielectric_proxy", 0.0)
    viscosity = gc_props.get("viscosity_proxy", 0.0)
    result["uncertainty_flag"] = (
        diel_std > abs(dielectric) * _UQ_THRESHOLD_FRACTION
        or visc_std > abs(viscosity) * _UQ_THRESHOLD_FRACTION
    )
    diel_norm = diel_std / max(abs(dielectric), 1.0)
    visc_norm = visc_std / max(abs(viscosity), 1.0)
    result["uncertainty_score"] = round((diel_norm + visc_norm) / 2.0, 4)
    if has_surrogate:
        result["surrogate_skipped"] = skip_quantum
    return result


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
            self._disk_cache: CacheBackend = RedisCache(url=redis_url)  # type: ignore[assignment]
        else:
            self._disk_cache: CacheBackend = l2_cache or DictCache()
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

    def _compute_quantum(self, ctx: MoleculeContext, skip_quantum: bool, s_homo: float, s_lumo: float) -> tuple[float, float, float, str, Any]:
        """Compute or skip quantum evaluation. Delegates to ``_compute_quantum_result``."""
        r = _compute_quantum_result(ctx, self._quantum, skip_quantum, s_homo, s_lumo)
        return r["homo_eV"], r["lumo_eV"], r["gap_eV"], r["method"], r["confidence"]

    def _compute_uq_penalty(self, ctx: MoleculeContext) -> tuple[float, float, float]:
        """Compute GC uncertainty penalty. Delegates to ``_compute_uq_penalty``."""
        return _compute_uq_penalty(ctx, self._gc_uq)

    def _compute_gc_properties(self, ctx: MoleculeContext) -> dict[str, float]:
        """Compute all GC-based bulk property proxies. Delegates to pure ``_compute_gc_properties``."""
        return _compute_gc_properties(ctx, self._property_pack)

    def _evaluate_quantum(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Evaluate quantum properties (HOMO/LUMO/gap) via surrogate or xTB/TOM.

        Bundles the surrogate pre-filter and quantum computation into one call.
        Returns a dict with quantum results plus surrogate metadata.
        """
        surrogate_penalty, s_homo, s_lumo, skip_quantum = self._run_surrogate(ctx)
        homo, lumo, gap, quantum_method, quantum_confidence_val = self._compute_quantum(
            ctx, skip_quantum=skip_quantum, s_homo=s_homo, s_lumo=s_lumo,
        )
        return {
            "homo_eV": homo,
            "lumo_eV": lumo,
            "gap_eV": gap,
            "surrogate_penalty": surrogate_penalty,
            "s_homo": s_homo,
            "skip_quantum": skip_quantum,
            "quantum_method": quantum_method,
            "quantum_confidence_val": quantum_confidence_val,
        }

    def _evaluate_gc(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Evaluate GC bulk properties and UQ penalty.

        Bundles GC property prediction and uncertainty quantification.
        """
        gc_props = self._compute_gc_properties(ctx)
        uq_penalty, diel_std, visc_std = self._compute_uq_penalty(ctx)
        return {
            "gc_props": gc_props,
            "uq_penalty": uq_penalty,
            "diel_std": diel_std,
            "visc_std": visc_std,
        }

    def _compute_ood_penalty(self, ctx: MoleculeContext) -> tuple[float, str]:
        """Compute centroid-based OOD penalty. Delegates to pure ``_compute_ood_penalty``."""
        return _compute_ood_penalty(ctx, self._gc_uq)

    def _build_domain(self, ctx: MoleculeContext, skip_quantum: bool, surrogate_penalty: float, s_homo: float, uq_penalty: float) -> tuple[float, str, bool]:
        """Build domain penalty. Delegates to pure ``_build_domain``."""
        return _build_domain(ctx, skip_quantum, surrogate_penalty, s_homo, uq_penalty, self._gc_uq)

    def _apply_sei_penalty(self, ctx: MoleculeContext, lumo: float, domain_penalty: float, domain_reason_str: str) -> tuple[float, str, bool]:
        """Apply SEI motif penalty. Delegates to pure ``_apply_sei_penalty``."""
        return _apply_sei_penalty(ctx, lumo, domain_penalty, domain_reason_str)

    def _assemble_result(
        self,
        smiles: str,
        homo: float, lumo: float, gap: float,
        gc_props: dict[str, float],
        domain_penalty: float, domain_reason_str: str, domain_applicable: bool,
        quantum_method: str, quantum_confidence_val: str,
        skip_quantum: bool,
        diel_std: float = 0.0,
        visc_std: float = 0.0,
    ) -> dict[str, Any]:
        """Assemble the final evaluation result dict. Delegates to pure ``_assemble_result``."""
        return _assemble_result(
            smiles, homo, lumo, gap, gc_props,
            domain_penalty, domain_reason_str, domain_applicable,
            quantum_method, quantum_confidence_val, skip_quantum,
            has_surrogate=self._surrogate is not None,
            diel_std=diel_std, visc_std=visc_std,
        )

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
            ctx, q["skip_quantum"], q["surrogate_penalty"], q["s_homo"], g["uq_penalty"],
        )
        domain_penalty, domain_reason_str, domain_applicable = self._apply_sei_penalty(
            ctx, q["lumo_eV"], domain_penalty, domain_reason_str,
        )

        if domain_penalty < 0.85:
            logger.warning(
                "Warning: Molecule is outside the calibrated domain of the current kernel. "
                "Consider retuning via Aurelius Certification Lab. "
                "(smiles=%s, domain_penalty=%.4f, reason=%s)",
                smiles, domain_penalty, domain_reason_str,
            )

        result = self._assemble_result(
            smiles, q["homo_eV"], q["lumo_eV"], q["gap_eV"],
            g["gc_props"],
            domain_penalty, domain_reason_str, domain_applicable,
            q["quantum_method"], q["quantum_confidence_val"],
            q["skip_quantum"],
            diel_std=g["diel_std"],
            visc_std=g["visc_std"],
        )
        self._cache[key] = result
        self._disk_cache[key] = result
        return result

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
