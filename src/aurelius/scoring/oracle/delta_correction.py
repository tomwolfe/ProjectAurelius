"""Δ-learning correction layer for the Topological Orbital Model (TOM).

Physical justification: TOM is a closed-form 1-D particle-in-a-box model of
conjugation. It systematically mis-estimates HOMO/LUMO for molecules whose
electronic structure is not captured by a single conjugation length (branched
pi-systems, through-bond coupling, hyperconjugation from C–F / C–O sigma
bonds, conformational averaging). These errors are structured, not random,
so a residual model trained on the difference between TOM and reference DFT
values can correct them while keeping the interpretable TOM as the base model.

The residual (Δ = DFT − TOM) is regressed from ECFP4 fingerprints with a
regularized kernel ridge (RBF) model. The correction is only trusted in
proportion to how close a molecule is to the calibration domain — molecules
with fingerprints far from every calibration molecule get a residual near
zero (regularization pulls the KRR prediction toward the mean of the data),
so out-of-domain predictions degrade gracefully back to raw TOM.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.kernel_ridge import KernelRidge

from aurelius.scoring.oracle.quantum import predict_tom_orbitals

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "orbital_calibration.json",
)

_KRR_PARAMS: dict[str, Any] = {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1}


def _load_calibration() -> list[dict[str, float]]:
    """Load the DFT HOMO/LUMO calibration set."""
    with open(_CALIBRATION_PATH) as f:
        return json.load(f)


def _ecfp4_vector(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Encode a molecule as a dense ECFP4 bit vector.

    ECFP4 (Morgan radius 2, 2048 bits) captures local topology around each
    atom, which is the feature space in which TOM residuals are smooth.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


class DeltaCorrection:
    """KRR residual model mapping ECFP4 fingerprints to TOM HOMO/LUMO errors.

    Two independent kernel-ridge models are fit on the calibration residuals
    (DFT − TOM) for HOMO and LUMO. A molecule's predicted residual is added
    to its raw TOM prediction to produce the corrected orbital energy.
    """

    def __init__(self, calib: list[dict[str, float]] | None = None) -> None:
        self._calib = calib if calib is not None else _load_calibration()
        self._X = np.zeros((len(self._calib), 2048), dtype=np.float32)
        self._y_homo = np.zeros(len(self._calib), dtype=np.float64)
        self._y_lumo = np.zeros(len(self._calib), dtype=np.float64)
        for i, entry in enumerate(self._calib):
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                raise ValueError(f"Unparseable calibration SMILES: {entry['smiles']}")
            self._X[i] = _ecfp4_vector(mol)
            tom_homo, tom_lumo = predict_tom_orbitals(mol)
            self._y_homo[i] = entry["homo_eV"] - tom_homo
            self._y_lumo[i] = entry["lumo_eV"] - tom_lumo
        self._homo_model = KernelRidge(**_KRR_PARAMS).fit(self._X, self._y_homo)
        self._lumo_model = KernelRidge(**_KRR_PARAMS).fit(self._X, self._y_lumo)

    def predict_deltas(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (delta_homo, delta_lumo) residual predictions for a molecule."""
        x = _ecfp4_vector(mol).reshape(1, -1)
        return float(self._homo_model.predict(x)[0]), float(self._lumo_model.predict(x)[0])

    def predict_corrected(
        self, mol: Chem.Mol, base: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Return corrected (homo_eV, lumo_eV) for a molecule.

        Falls back to raw TOM predictions if ``base`` is not supplied.
        """
        tom_homo, tom_lumo = base if base is not None else predict_tom_orbitals(mol)
        d_homo, d_lumo = self.predict_deltas(mol)
        return tom_homo + d_homo, tom_lumo + d_lumo

    def loo_mae(self) -> float:
        """Leave-one-out cross-validation MAE (mean of HOMO/LUMO errors, eV).

        Physical justification: LOO gives an honest estimate of how the
        residual model generalizes to unseen molecules — each calibration
        molecule is scored by a model that never saw it, mirroring how the
        correction is applied to novel EA candidates.
        """
        errors: list[float] = []
        for i in range(len(self._calib)):
            mask = np.ones(len(self._calib), dtype=bool)
            mask[i] = False
            homo_model = KernelRidge(**_KRR_PARAMS).fit(self._X[mask], self._y_homo[mask])
            lumo_model = KernelRidge(**_KRR_PARAMS).fit(self._X[mask], self._y_lumo[mask])
            d_homo = float(homo_model.predict(self._X[i : i + 1])[0])
            d_lumo = float(lumo_model.predict(self._X[i : i + 1])[0])
            homo_err = abs(self._y_homo[i] - d_homo)
            lumo_err = abs(self._y_lumo[i] - d_lumo)
            errors.append((homo_err + lumo_err) / 2.0)
        return float(np.mean(errors))


_DEFAULT: DeltaCorrection | None = None


def get_delta_correction() -> DeltaCorrection:
    """Return the process-wide singleton Δ-correction model (lazy init)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DeltaCorrection()
    return _DEFAULT


def predict_corrected_orbitals(mol: Chem.Mol) -> tuple[float, float]:
    """Public convenience wrapper: corrected (homo_eV, lumo_eV) for a molecule."""
    return get_delta_correction().predict_corrected(mol)
