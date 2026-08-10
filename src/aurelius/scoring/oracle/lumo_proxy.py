"""Δ-learning correction layer for LUMO (reduction stability proxy).

Physical justification
----------------------
LUMO ranking is provenance-confounded (unseen Spearman ρ ≈ 0.06): 69% of label
variance is between-source rather than between-molecule. A predictor given only
the citation string scores ρ = 0.837, beating every real model by 10×. The
cause: the DFT LUMO labels in ``orbital_calibration.json`` were compiled from
~50 different functionals and basis sets.

ADR-2026-08-09-02: The calibration set is now ``lumo_calibration_xtb.json`` —
GFN2-xTB single-point LUMO values calibrated to the B3LYP/6-311++G** scale via
the OLS affine map in ``quantum.py``. Because every value is produced by the
*same* quantum-chemical method, the set is free of provenance confound
(verified: citation-only ρ = 0.0, between-source fraction = 0.0). Ranking on
this set reflects chemistry, not which journal a value originated from.

MAE is comparatively robust to a constant per-source offset. The Δ-layer
improves LUMO *calibration*: the residual (``xTB_LUMO − TOM_LUMO``) is
structured, so a GPR can learn meaningful corrections while keeping the
interpretable TOM as the base model.

Architecture
------------
GPR residual model mapping ECFP4 fingerprints to LUMO prediction errors:
    Δ = LUMO_ref − TOM_LUMO
    corrected_LUMO = TOM_LUMO + shrinkage(Δ̂)

The shrinkage factor uses the normal-normal posterior mean:
    conf = σ²_prior / (σ²_prior + σ²_pred)
so OOD molecules gracefully revert to raw TOM.

Validation
----------
Scaffold-disjoint (Murcko) split ensures every molecule from a given scaffold
group is either entirely in training or entirely in test. Random splits leak
structural similarity and overestimate performance.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
)

from aurelius.scoring.oracle.quantum import predict_tom_orbitals

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "lumo_calibration_xtb.json",
)

# Fallback to the original confounded set if the xTB file is missing.
_FALLBACK_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "orbital_calibration.json",
)

_GPR_KERNEL = ConstantKernel(1.0) * RBF(
    length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

_GPR_KWARGS: dict[str, Any] = {
    "kernel": _GPR_KERNEL,
    "alpha": 1e-6,
    "normalize_y": True,
    "n_restarts_optimizer": 2,
    "random_state": 42,
}


def _load_calibration() -> list[dict[str, float]]:
    """Load the internally consistent xTB LUMO calibration set.

    Falls back to the original ``orbital_calibration.json`` (DFT labels,
    provenance-confounded) if the xTB file is unavailable, emitting a
    warning. ADR-2026-08-09-02: the xTB set is preferred because all LUMO
    labels come from a single quantum-chemical method, making the set
    free of between-source confound (citation-only rho = 0.0, verified).
    """
    try:
        with open(_CALIBRATION_PATH) as f:
            data = json.load(f)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if entries:
            return entries
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    logger = logging.getLogger(__name__)
    logger.warning(
        "LUMO proxy fell back to orbital_calibration.json (confounded DFT labels); "
        "lumo_calibration_xtb.json not found."
    )
    with open(_FALLBACK_CALIBRATION_PATH) as f:
        return json.load(f)


def _ecfp4_vector(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Encode a molecule as a dense ECFP4 bit vector."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float64)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


def _get_murcko_scaffold(mol: Chem.Mol) -> str:
    """Get the Murcko scaffold SMILES for scaffold-disjoint splitting.

    Falls back to the molecule's own SMILES if scaffold extraction fails.
    """
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(core)
        if scaffold:
            return scaffold
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


class LumoProxy:
    """GPR residual model for LUMO (reduction stability proxy).

    Fits a Gaussian Process on ECFP4 → (xTB_LUMO − TOM_LUMO) residuals using
    the internally consistent ``lumo_calibration_xtb.json`` set (ADR-2026-08-09-02).
    Provides shrinkage-damped corrections that revert to TOM for OOD molecules.

    The xTB labels are free of provenance confound because all values are
    computed with the same GFN2-xTB single-point method. Ranking on this set
    is chemically meaningful (verified: citation-only ρ = 0.0).

    IMPORTANT: This is an MAE-only proxy. Do NOT use the output for ranking
    claims against the confounded external benchmark set.
    """

    def __init__(
        self,
        calib: list[dict[str, float]] | None = None,
    ) -> None:
        self._calib = calib if calib is not None else _load_calibration()

        valid_entries = []
        for entry in self._calib:
            mol = Chem.MolFromSmiles(entry.get("smiles", ""))
            if mol is not None:
                valid_entries.append(entry)
        self._calib = valid_entries

        n = len(self._calib)
        self._X = np.zeros((n, 2048), dtype=np.float64)
        self._y_lumo = np.zeros(n, dtype=np.float64)

        for i, entry in enumerate(self._calib):
            mol = Chem.MolFromSmiles(entry["smiles"])
            self._X[i] = _ecfp4_vector(mol)
            _, tom_lumo = predict_tom_orbitals(mol)
            self._y_lumo[i] = entry["lumo_eV"] - tom_lumo

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._lumo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(
                self._X, self._y_lumo
            )
        self._prior_std_lumo = float(np.std(self._y_lumo)) or 1.0

    def predict_residual(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (delta_lumo_mean, std) for a molecule."""
        x = _ecfp4_vector(mol).reshape(1, -1)
        d_lumo, std_lumo = self._lumo_model.predict(x, return_std=True)
        return float(d_lumo[0]), float(std_lumo[0])

    def predict_corrected(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (corrected_lumo_eV, confidence) for a molecule.

        confidence ∈ [0, 1] from the GPR shrinkage factor. High confidence
        means the molecule is in-distribution; low confidence means the
        correction is uncertain and the result is shrunk toward TOM.
        """
        _, tom_lumo = predict_tom_orbitals(mol)
        d_lumo, std_lumo = self.predict_residual(mol)
        var_prior = self._prior_std_lumo ** 2
        conf = var_prior / (var_prior + std_lumo ** 2)
        corrected = tom_lumo + d_lumo * conf
        return corrected, round(conf, 4)

    def scaffold_disjoint_mae(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Scaffold-disjoint cross-validation MAE for the LUMO correction.

        Splits by Murcko scaffold groups so that structurally similar
        molecules never appear in both train and test. This is the honest
        estimate of generalization to novel chemistries.

        Returns dict with: mae_raw_tom, mae_corrected, n_test_total.
        """
        from sklearn.model_selection import GroupKFold

        scaffold_groups: list[str] = []
        for entry in self._calib:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                scaffold_groups.append("")
            else:
                scaffold_groups.append(_get_murcko_scaffold(mol))

        unique_scaffolds: dict[str, int] = {}
        group_ids: list[int] = []
        for s in scaffold_groups:
            if s not in unique_scaffolds:
                unique_scaffolds[s] = len(unique_scaffolds)
            group_ids.append(unique_scaffolds[s])

        group_arr = np.array(group_ids)
        n_groups = len(unique_scaffolds)
        n_splits = min(n_splits, n_groups)

        if n_splits < 2:
            return {
                "mae_raw_tom": float("nan"),
                "mae_corrected": float("nan"),
                "n_test_total": 0,
                "n_splits": n_groups,
            }

        kf = GroupKFold(n_splits=n_splits)
        errors_raw: list[float] = []
        errors_corrected: list[float] = []

        for train_idx, test_idx in kf.split(self._X, self._y_lumo, group_arr):
            X_train, X_test = self._X[train_idx], self._X[test_idx]
            y_train, y_test = self._y_lumo[train_idx], self._y_lumo[test_idx]

            model = GaussianProcessRegressor(**_GPR_KWARGS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            prior_std = float(np.std(y_train)) or 1.0
            var_prior = prior_std ** 2

            d_test, std_test = model.predict(X_test, return_std=True)
            for i in range(len(test_idx)):
                conf = var_prior / (var_prior + std_test[i] ** 2)
                corrected_delta = d_test[i] * conf
                raw_err = abs(y_test[i])
                corr_err = abs(y_test[i] - corrected_delta)
                errors_raw.append(raw_err)
                errors_corrected.append(corr_err)

        return {
            "mae_raw_tom": float(np.mean(errors_raw)),
            "mae_corrected": float(np.mean(errors_corrected)),
            "n_test_total": len(errors_raw),
            "n_splits": n_splits,
        }


_DEFAULT: LumoProxy | None = None


def get_lumo_proxy() -> LumoProxy:
    """Return the process-wide singleton LUMO proxy (lazy init)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LumoProxy()
    return _DEFAULT


def predict_reduction_stability(mol: Chem.Mol) -> dict[str, float]:
    """Public convenience wrapper returning LUMO proxy results.

    Returns dict with: lumo_eV, confidence.
    """
    proxy = get_lumo_proxy()
    lumo, conf = proxy.predict_corrected(mol)
    return {"lumo_eV": round(lumo, 4), "confidence": conf}
