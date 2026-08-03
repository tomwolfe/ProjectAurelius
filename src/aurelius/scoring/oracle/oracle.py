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

from aurelius.scoring.oracle.conformal import get_conformal_predictor
from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    compute_gc_domain_penalty,
    predict_dielectric_proxy,
    predict_ionic_conductivity_proxy,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)
from aurelius.scoring.oracle.quantum import (
    QuantumOracle,
    compute_quantum_domain_penalty,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "dielectric_proxy": (1.0, 100.0),
    "viscosity_proxy": (0.1, 50.0),
    "homo_eV": (-12.0, -3.0),
    "lumo_eV": (-5.0, 5.0),
}


def _apply_physical_bounds(
    raw: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Clamp each property to its physical bounds, collecting warnings.

    Returns the clamped dict and a list of warning messages for any value
    that fell outside its bounds.
    """
    clamped: dict[str, float] = {}
    warnings_list: list[str] = []
    for key, value in raw.items():
        lo, hi = _PHYSICAL_BOUNDS.get(key, (float("-inf"), float("inf")))
        if value < lo:
            clamped[key] = lo
            warnings_list.append(f"{key} below physical minimum (clamped to {lo})")
        elif value > hi:
            clamped[key] = hi
            warnings_list.append(f"{key} above physical maximum (clamped to {hi})")
        else:
            clamped[key] = value
    return clamped, warnings_list


class PropertyOracle:
    """Multi-objective property oracle with a hybrid physics model.

    Architecture:
      - HOMO / LUMO / Gap: QuantumOracle (xTB preferred, TOM fallback)
      - Dielectric proxy: GC fragment-additivity + TPSA-based cap
      - Viscosity proxy: GC fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy: GC fragment-additivity
    """

    def __init__(self, use_xtb: bool = True) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )

        smiles = ctx.smiles
        if smiles in self._cache:
            return self._cache[smiles]

        quantum_result = self._quantum.evaluate(ctx.mol)
        homo = quantum_result["homo_eV"]
        lumo = quantum_result["lumo_eV"]
        gap = lumo - homo

        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)
        conductivity = predict_ionic_conductivity_proxy(dielectric, viscosity, li_solvation)

        # Domain of applicability penalties
        q_penalty, q_reason = compute_quantum_domain_penalty(ctx)
        gc_penalty, gc_reason = compute_gc_domain_penalty(ctx)
        domain_penalty = min(q_penalty, gc_penalty)
        domain_reasons: list[str] = []
        if q_penalty < 1.0:
            domain_reasons.append(f"quantum: {q_reason}")
        if gc_penalty < 1.0:
            domain_reasons.append(f"GC: {gc_reason}")
        domain_applicable = domain_penalty >= 0.85
        domain_reason_str = "; ".join(domain_reasons) if domain_reasons else _DATA_SOURCE

        clamped_values, sanity_warning = _apply_physical_bounds({
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "homo_eV": homo,
            "lumo_eV": lumo,
        })

        cp = get_conformal_predictor()
        intervals = {
            "homo": cp.predict_interval("homo", clamped_values["homo_eV"]),
            "lumo": cp.predict_interval("lumo", clamped_values["lumo_eV"]),
            "dielectric": cp.predict_interval(
                "dielectric", clamped_values["dielectric_proxy"]
            ),
            "viscosity": cp.predict_interval(
                "viscosity", clamped_values["viscosity_proxy"]
            ),
        }
        conformal_confidence = cp.confidence_discount(intervals)

        result: dict[str, Any] = {
            "homo_eV": round(clamped_values["homo_eV"], 4),
            "lumo_eV": round(clamped_values["lumo_eV"], 4),
            "gap_eV": round(clamped_values["lumo_eV"] - clamped_values["homo_eV"], 4),
            "dielectric_proxy": round(clamped_values["dielectric_proxy"], 4),
            "viscosity_proxy": round(clamped_values["viscosity_proxy"], 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "conductivity_proxy": round(conductivity, 4),
            "domain_applicable": domain_applicable,
            "domain_reason": domain_reason_str,
            "domain_penalty": round(domain_penalty, 4),
            "quantum_method": self._quantum.method,
            "quantum_confidence": quantum_result.get("quantum_confidence", "unknown"),
            "sanity_warning": sanity_warning,
            "conformal_intervals": {
                prop: [round(lo, 4), round(hi, 4)]
                for prop, (lo, hi) in intervals.items()
            },
            "conformal_confidence": round(conformal_confidence, 4),
        }

        self._cache[smiles] = result
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
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
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        self._cache.clear()
        self._quantum.clear_cache()
