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

from rdkit import Chem

from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    predict_dielectric_proxy,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)
from aurelius.scoring.oracle.quantum import QuantumOracle
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


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

        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "dielectric_proxy": round(dielectric, 4),
            "viscosity_proxy": round(viscosity, 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "domain_applicable": True,
            "domain_reason": _DATA_SOURCE,
            "quantum_method": self._quantum.method,
        }

        self._cache[smiles] = result
        return result

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
