from __future__ import annotations

import logging
from typing import Any

import numpy as np
from aurelius.scoring.oracle import PropertyOracle
from aurelius.scoring.oracle.gc import BasePropertyModel, ElectrolytePack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Property pack registry - maps domain names to BasePropertyModel classes
# ---------------------------------------------------------------------------

_PROPERTY_PACK_REGISTRY: dict[str, type[BasePropertyModel]] = {}


def _build_pack_registry() -> dict[str, type[BasePropertyModel]]:
    """Discover and cache property pack subclasses.

    Scans ``aurelius.scoring.oracle.packs`` and the core ``gc`` module for
    all concrete ``BasePropertyModel`` subclasses and indexes them by
    ``name``.
    """
    packs: dict[str, type[BasePropertyModel]] = {}

    packs[ElectrolytePack.name] = ElectrolytePack  # type: ignore[assignment]

    try:
        from aurelius.scoring.oracle.packs import organic_electronics  # noqa: F401
    except ImportError:
        pass

    for subclass in BasePropertyModel.__subclasses__():
        name: str = getattr(subclass, "name", "")
        if name and name not in packs:
            packs[name] = subclass

    return packs


def get_property_pack(pack_name: str) -> BasePropertyModel:
    """Return a ``BasePropertyModel`` instance for the given pack name.

    Falls back to ``ElectrolytePack`` if the pack name is unknown.
    """
    if not _PROPERTY_PACK_REGISTRY:
        _PROPERTY_PACK_REGISTRY.update(_build_pack_registry())

    cls = _PROPERTY_PACK_REGISTRY.get(pack_name)
    if cls is None:
        logger.warning(
            "Unknown property pack '%s', falling back to 'electrolyte'. "
            "Available packs: %s",
            pack_name, list(_PROPERTY_PACK_REGISTRY),
        )
        cls = ElectrolytePack
    return cls()


def resolve_oracle_key(pack: BasePropertyModel, property_name: str) -> str:
    """Resolve a short property name to the oracle result key for this pack."""
    keys = pack.property_keys()
    return keys.get(property_name, keys.get(property_name.lower(), "homo_eV"))


def _adjust_prediction(
    oracle: PropertyOracle,
    smiles: str,
    property_name: str,
    homo_offset: float,
    lumo_offset: float,
    gc_scale: float,
    property_pack: BasePropertyModel | None = None,
) -> float:
    result = oracle.evaluate_smiles(smiles)
    pack = property_pack or oracle.property_pack
    key = resolve_oracle_key(pack, property_name)
    raw = result.get(key, 0.0)
    if not raw:
        raw = result.get("homo_eV", 0.0)
    if property_name in ("homo", "homo_eV"):
        return raw + homo_offset
    if property_name in ("lumo", "lumo_eV"):
        return raw + lumo_offset
    return raw * gc_scale


def _parse_training_pairs(
    training_pairs: list[tuple[str, ...]],
) -> tuple[list[str], list[str], np.ndarray]:
    smiles_list: list[str] = []
    prop_list: list[str] = []
    exp_list: list[float] = []

    for pair in training_pairs:
        if len(pair) == 3:
            smi, prop, val = pair
            smiles_list.append(str(smi))
            prop_list.append(str(prop))
            exp_list.append(float(val))
        elif len(pair) == 2:
            smiles_list.append(str(pair[0]))
            prop_list.append("homo")
            exp_list.append(float(pair[1]))
        else:
            raise ValueError(
                f"Each training pair must have 2 (SMILES, value) or 3 (SMILES, property, value) elements, "
                f"got {len(pair)}: {pair}"
            )

    return smiles_list, prop_list, np.array(exp_list, dtype=np.float64)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    d = x_rank - y_rank
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1.0))
    if np.isnan(rho):
        return 0.0
    return float(rho)


def _nelder_mead(
    f,
    x0: np.ndarray,
    args: tuple = (),
    bounds: list[tuple[float, float]] | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    n = len(x0)

    def clamp(x: np.ndarray) -> np.ndarray:
        if bounds is None:
            return x
        return np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])

    def obj(x: np.ndarray) -> float:
        return float(f(x, *args))

    simplex: list[np.ndarray] = [clamp(np.array(x0, dtype=np.float64))]
    for i in range(n):
        p = np.array(x0, dtype=np.float64)
        p[i] = x0[i] + 0.5 if abs(x0[i]) < 0.01 else x0[i] * 1.05
        simplex.append(clamp(p))

    f_vals: list[float] = [obj(simplex[i]) for i in range(n + 1)]

    alpha = 1.0
    gamma = 2.0
    rho_nm = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        indices = np.argsort(f_vals)
        simplex = [simplex[i] for i in indices]
        f_vals = [f_vals[i] for i in indices]

        if np.std(f_vals) < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)

        xr = clamp(centroid + alpha * (centroid - simplex[-1]))
        fr = obj(xr)

        if f_vals[0] <= fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        elif fr < f_vals[0]:
            xe = clamp(centroid + gamma * (xr - centroid))
            fe = obj(xe)
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        else:
            if fr < f_vals[-1]:
                xc = clamp(centroid + rho_nm * (xr - centroid))
                fc = obj(xc)
                if fc < fr:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clamp(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        f_vals[i] = obj(simplex[i])
            else:
                xc = clamp(centroid + rho_nm * (centroid - simplex[-1]))
                fc = obj(xc)
                if fc < f_vals[-1]:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clamp(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        f_vals[i] = obj(simplex[i])

    best_idx = int(np.argmin(f_vals))
    return simplex[best_idx]


class KernelOptimizer:
    """Optimises kernel TOM parameters using Nelder-Mead minimisation."""

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 200,
        tolerance: float = 1e-6,
        property_pack_name: str = "electrolyte",
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.property_pack_name = property_pack_name
        self._oracle: PropertyOracle | None = None
        self._property_pack: BasePropertyModel | None = None

    def _get_oracle(self) -> PropertyOracle:
        if self._oracle is None:
            pack = self._get_property_pack()
            self._oracle = PropertyOracle(
                use_xtb=False, use_surrogate=True, use_gc_uq=True,
                property_pack=pack,
            )
        return self._oracle

    def _get_property_pack(self) -> BasePropertyModel:
        if self._property_pack is None:
            self._property_pack = get_property_pack(self.property_pack_name)
        return self._property_pack

    def optimize(
        self,
        training_pairs: list[tuple[str, ...]],
        domain_boundary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        smiles_list, property_names, exp_values = _parse_training_pairs(training_pairs)

        x0 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.1, 10.0), (0.1, 10.0)]

        result_x = _nelder_mead(
            self._objective,
            x0,
            args=(smiles_list, property_names, exp_values),
            bounds=bounds,
            max_iter=self.max_iter,
            tol=self.tolerance,
        )

        tom_parameters = {
            "homo_offset": float(result_x[0]),
            "lumo_offset": float(result_x[1]),
            "gc_scale": float(result_x[2]),
            "uq_scale": float(result_x[3]),
        }

        metrics = self._compute_metrics(
            smiles_list, property_names, exp_values, tom_parameters
        )

        pack = self._get_property_pack()
        kernel: dict[str, Any] = {
            "version": "1.0.0",
            "domain_boundary": domain_boundary or {"domain": self.property_pack_name},
            "tom_parameters": tom_parameters,
            "gc_fragments": pack.get_fragment_names(),
            "uq_weights": {"ensemble_weight": 0.5},
            "validation_metrics": metrics,
        }
        return kernel

    def _objective(
        self,
        params: np.ndarray,
        smiles_list: list[str],
        property_names: list[str],
        exp_values: np.ndarray,
    ) -> float:
        if not smiles_list or len(exp_values) == 0:
            return 0.0

        oracle = self._get_oracle()
        pack = self._get_property_pack()
        predictions: list[float] = []
        for smi, prop in zip(smiles_list, property_names, strict=False):
            pred = _adjust_prediction(
                oracle, smi, prop,
                homo_offset=float(params[0]),
                lumo_offset=float(params[1]),
                gc_scale=float(params[2]),
                property_pack=pack,
            )
            predictions.append(pred)

        pred_array = np.array(predictions, dtype=np.float64)
        mse = float(np.mean((pred_array - exp_values) ** 2))
        return float(np.sqrt(mse))

    def _compute_metrics(
        self,
        smiles_list: list[str],
        property_names: list[str],
        exp_values: np.ndarray,
        tom_parameters: dict[str, float],
    ) -> dict[str, float]:
        if not smiles_list or len(exp_values) == 0:
            return {"spearman_rho": 0.0, "mae": 0.0, "rmse": 0.0, "n_training": 0}

        oracle = self._get_oracle()
        pack = self._get_property_pack()
        predictions: list[float] = []
        for smi, prop in zip(smiles_list, property_names, strict=False):
            pred = _adjust_prediction(
                oracle, smi, prop,
                homo_offset=tom_parameters.get("homo_offset", 0.0),
                lumo_offset=tom_parameters.get("lumo_offset", 0.0),
                gc_scale=tom_parameters.get("gc_scale", 1.0),
                property_pack=pack,
            )
            predictions.append(pred)

        pred_array = np.array(predictions, dtype=np.float64)
        residuals = pred_array - exp_values
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        spearman = _spearman_rho(pred_array, exp_values)

        return {
            "spearman_rho": spearman,
            "mae": mae,
            "rmse": rmse,
            "n_training": len(smiles_list),
        }
