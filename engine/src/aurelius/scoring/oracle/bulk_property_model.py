"""Bulk Property Model — ML QSPR model for dielectric, viscosity, and donor number.

Trains an XGBoost model on ECFP4 fingerprints + 10 physical-chemical
descriptors to predict bulk electrolyte properties. Uses 5-fold Murcko
scaffold cross-validation with reported MAE, RMSE, and Spearman ρ.

Replaces the hand-tuned 40-fragment GC functions in gc.py with this
trained ML model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_N_SPLIT = 5
_N_BITS = 2048
_N_JOBS = -1

_TRAINING_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "external_property_benchmark.json"

_PHYSICAL_DESCRIPTORS = [
    "mw",
    "tpsa",
    "logp",
    "rotatable_bonds",
    "n_heavy_atoms",
    "n_carbons",
    "n_oxygens",
    "n_nitrogens",
    "n_fluorines",
    "n_sulfurs",
]

_PROPERTY_TARGETS = ["dielectric_constant", "viscosity_cP", "donor_number"]

# Mapping from ML model property names to GC proxy keys for blending.
_ML_TO_GC_KEY: dict[str, str] = {
    "dielectric_constant": "dielectric_proxy",
    "viscosity_cP": "viscosity_proxy",
    "donor_number": "li_solvation_proxy",
}


def _compute_ecfp4(mol: Chem.Mol) -> np.ndarray:
    """Compute ECFP4 fingerprint as a numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_N_BITS)
    arr = np.zeros(_N_BITS, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr


def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute 10 physical-chemical descriptors as a numpy array."""
    from rdkit.Chem import Descriptors, Lipinski

    vals = np.array([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumRotatableBonds(mol),
        Descriptors.HeavyAtomCount(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
    ], dtype=np.float32)
    return vals


def _compute_features(mol: Chem.Mol) -> np.ndarray:
    """Compute combined ECFP4 + physical descriptors feature vector."""
    ecfp = _compute_ecfp4(mol)
    desc = _compute_descriptors(mol)
    return np.concatenate([ecfp, desc])


def _murcko_scaffold(smiles: str) -> str | None:
    """Compute Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except Exception:
        return None


def _scaffold_split(
    smiles_list: list[str],
    n_splits: int = _N_SPLIT,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Split indices by Murcko scaffold for cross-validation."""
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles_list):
        scaffold = _murcko_scaffold(smi)
        if scaffold is None:
            scaffold = f"__unknown_{idx}__"
        scaffolds.setdefault(scaffold, []).append(idx)

    scaffold_list = list(scaffolds.values())
    scaffold_list.sort(key=lambda s: -len(s))

    splits: list[tuple[list[int], list[int]]] = []

    for fold_idx in range(n_splits):
        test_indices: list[int] = []
        train_indices: list[int] = []

        for s_idx, scaffold_indices in enumerate(scaffold_list):
            if s_idx % n_splits == fold_idx:
                test_indices.extend(scaffold_indices)
            else:
                train_indices.extend(scaffold_indices)

        splits.append((train_indices, test_indices))

    return splits


def _evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, and Spearman ρ."""
    from scipy.stats import spearmanr

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, _p = spearmanr(y_true, y_pred)
    rho = float(rho) if not np.isnan(rho) else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "spearman_rho": round(rho, 4),
    }


class BulkPropertyModel:
    """XGBoost-based QSPR model for bulk electrolyte properties.

    Trains separate models for dielectric constant, viscosity, and donor
    number using ECFP4 fingerprints + 10 physical-chemical descriptors.
    Uses 5-fold Murcko scaffold cross-validation.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self._models: dict[str, xgb.XGBRegressor] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._cv_metrics: dict[str, list[dict[str, float]]] = {}
        self._trained = False

    def _load_training_data(self) -> tuple[np.ndarray, dict[str, list[float]], list[str]]:
        """Load the training set from JSON.

        Returns:
            X: Feature array (N, n_features), one row per valid molecule.
            target_dict: Dict mapping each target name to a list of float
                (or NaN) aligned with X rows.
            smiles_list: SMILES strings aligned with X rows.
        """
        if not _TRAINING_SET_PATH.exists():
            msg = f"Training set not found at {_TRAINING_SET_PATH}"
            raise FileNotFoundError(msg)

        with open(_TRAINING_SET_PATH) as f:
            data = json.load(f)

        results = data.get("results", []) if isinstance(data, dict) else data
        if not results:
            msg = "Training set is empty"
            raise ValueError(msg)

        smiles_list: list[str] = []
        feature_list: list[np.ndarray] = []
        target_dict: dict[str, list[float]] = {k: [] for k in _PROPERTY_TARGETS}

        for entry in results:
            smiles = entry.get("smiles", "")
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles_list.append(smiles)
            feature_list.append(_compute_features(mol))

            for target in _PROPERTY_TARGETS:
                val = entry.get(target)
                if val is None:
                    target_dict[target].append(np.nan)
                else:
                    target_dict[target].append(float(val))

        X = np.array(feature_list, dtype=np.float32)
        return X, target_dict, smiles_list

    def train(self) -> dict[str, Any]:
        """Train the model with 5-fold Murcko scaffold cross-validation.

        Returns:
            Dict with CV metrics for each property.
        """
        X, target_dict, smiles_list = self._load_training_data()

        if X.shape[0] == 0:
            msg = "No valid training data"
            raise ValueError(msg)

        for target in _PROPERTY_TARGETS:
            vals = np.array(target_dict[target], dtype=np.float32)
            mask = ~np.isnan(vals)
            if mask.sum() < _N_SPLIT:
                logger.warning(
                    "Not enough valid values for %s (%d). Skipping.",
                    target,
                    mask.sum(),
                )
                continue

            X_target = X[mask]
            y_target = vals[mask]
            smiles_target = [s for s, m in zip(smiles_list, mask, strict=False) if m]
            splits = _scaffold_split(smiles_target, n_splits=_N_SPLIT, seed=self.random_state)

            self._cv_metrics.setdefault(target, [])

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train = X_target[train_idx]
                X_test = X_target[test_idx]
                y_train = y_target[train_idx]
                y_test = y_target[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    n_jobs=_N_JOBS,
                    verbosity=0,
                )
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                metrics = _evaluate_regression(y_test, y_pred)
                self._cv_metrics[target].append(metrics)

                logger.info(
                    "%s fold %d: MAE=%.4f RMSE=%.4f ρ=%.4f",
                    target,
                    fold_idx,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["spearman_rho"],
                )

            # Train final model on all data for this target
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_target)
            final_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=_N_JOBS,
                verbosity=0,
            )
            final_model.fit(X_scaled, y_target)
            self._models[target] = final_model
            self._scalers[target] = scaler

        self._trained = True

        summary: dict[str, Any] = {}
        for target in _PROPERTY_TARGETS:
            if target in self._cv_metrics:
                metrics_list = self._cv_metrics[target]
                avg_mae = float(np.mean([m["mae"] for m in metrics_list]))
                avg_rmse = float(np.mean([m["rmse"] for m in metrics_list]))
                avg_rho = float(np.mean([m["spearman_rho"] for m in metrics_list]))
                summary[target] = {
                    "cv_mae": round(avg_mae, 4),
                    "cv_rmse": round(avg_rmse, 4),
                    "cv_spearman_rho": round(avg_rho, 4),
                    "n_folds": len(metrics_list),
                }

        return summary

    def predict(
        self,
        ctx: MoleculeContext,
        property_name: str = "dielectric_constant",
    ) -> float:
        """Predict a bulk property for a molecule."""
        if not self._trained:
            msg = "Model not trained. Call train() first."
            raise RuntimeError(msg)

        if property_name not in self._models:
            msg = f"Unknown property: {property_name}"
            raise ValueError(msg)

        features = _compute_features(ctx.mol).reshape(1, -1)
        X_scaled = self._scalers[property_name].transform(features)
        return float(self._models[property_name].predict(X_scaled)[0])

    def predict_all(self, ctx: MoleculeContext) -> dict[str, float]:
        """Predict all bulk properties for a molecule."""
        return {
            prop: self.predict(ctx, prop)
            for prop in _PROPERTY_TARGETS
            if prop in self._models
        }

    def save(self, path: str = "bulk_property_model.json") -> None:
        """Save model metadata (not the XGBoost model itself for simplicity)."""
        metadata: dict[str, Any] = {
            "trained": self._trained,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "cv_metrics": {
                k: v for k, v in self._cv_metrics.items()
            },
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("BulkPropertyModel metadata saved to %s", path)

    def load(self, path: str = "bulk_property_model.json") -> bool:
        """Load model metadata."""
        try:
            with open(path) as f:
                metadata = json.load(f)
            self._trained = metadata.get("trained", False)
            self.n_estimators = metadata.get("n_estimators", self.n_estimators)
            self.max_depth = metadata.get("max_depth", self.max_depth)
            self.learning_rate = metadata.get("learning_rate", self.learning_rate)
            self._cv_metrics = metadata.get("cv_metrics", {})
            return True
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to load BulkPropertyModel: %s", exc)
            return False

    @property
    def name(self) -> str:
        return "bulk_property_model"
