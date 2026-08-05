"""Δ-learning correction layer for the Topological Orbital Model (TOM).

Physical justification: TOM is a closed-form 1-D particle-in-a-box model of
conjugation. It systematically mis-estimates HOMO/LUMO for molecules whose
electronic structure is not captured by a single conjugation length (branched
pi-systems, through-bond coupling, hyperconjugation from C–F / C–O sigma
bonds, conformational averaging). These errors are structured, not random,
so a residual model trained on the difference between TOM and reference DFT
values can correct them while keeping the interpretable TOM as the base model.

The residual (Δ = DFT − TOM) is regressed from ECFP4 fingerprints with a
Gaussian Process Regressor (GPR) using an RBF + WhiteKernel covariance.
GPR is preferred over kernel ridge because:

  1. It provides a calibrated standard deviation σ(Δ) for each prediction,
     which quantifies how far out-of-domain a molecule is and feeds into
     the conformal confidence discount (see W5).
  2. Out-of-domain predictions naturally shrink toward the mean residual
     (≈ 0 for a well-centred calibration set), so corrections degrade
     gracefully back to raw TOM without manual regularisation tuning.
  3. The marginal log-likelihood is optimised during hyperparameter search,
     balancing data fit against model complexity automatically.

Molecules with fingerprints far from every calibration molecule get a residual
near zero (GPR posterior mean → prior mean ≈ 0), so out-of-domain predictions
degrade gracefully back to raw TOM.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
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
    """Load the DFT HOMO/LUMO calibration set."""
    with open(_CALIBRATION_PATH) as f:
        return json.load(f)


def _ecfp4_vector(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Encode a molecule as a dense ECFP4 bit vector.

    ECFP4 (Morgan radius 2, 2048 bits) captures local topology around each
    atom, which is the feature space in which TOM residuals are smooth.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float64)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


class DeltaCorrection:
    """GPR residual model mapping ECFP4 fingerprints to TOM HOMO/LUMO errors.

    Two independent Gaussian Process models are fit on the calibration residuals
    (DFT − TOM) for HOMO and LUMO. A molecule's predicted mean residual is added
    to its raw TOM prediction to produce the corrected orbital energy. The
    associated GPR standard deviation quantifies prediction uncertainty and
    feeds into the conformal confidence discount in the selection layer.

    Optional OOD calibration set support: molecules from out-of-distribution
    chemical scaffolds can be included with 2× weight during training, improving
    correction accuracy for novel chemistries where TOM's particle-in-a-box
    model systematically fails (branched π-systems, through-bond coupling,
    hyperconjugation from C–F / C–O sigma bonds).
    """

    def __init__(
        self,
        calib: list[dict[str, float]] | None = None,
        calib_smiles: list[str] | None = None,
        ood_calibration_set: list[dict[str, float]] | None = None,
    ) -> None:
        self._calib = calib if calib is not None else _load_calibration()
        self._calib_smiles = calib_smiles if calib_smiles is not None else [
            Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"])) for entry in self._calib
            if Chem.MolFromSmiles(entry["smiles"]) is not None
        ]

        # Build base calibration arrays
        base_entries = list(self._calib)
        base_smiles = list(self._calib_smiles)

        # Append OOD calibration entries with 2× weight (duplicated)
        ood_entries: list[dict[str, float]] = []
        ood_smiles: list[str] = []
        if ood_calibration_set is not None:
            for entry in ood_calibration_set:
                mol = Chem.MolFromSmiles(entry.get("smiles", ""))
                if mol is None:
                    continue
                ood_entries.append(entry)
                ood_smiles.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"]))
                )

        # Combine: OOD entries are duplicated for 2× weight
        all_entries = base_entries + ood_entries + ood_entries
        all_smiles = base_smiles + ood_smiles + ood_smiles

        n = len(all_entries)
        self._X = np.zeros((n, 2048), dtype=np.float64)
        self._y_homo = np.zeros(n, dtype=np.float64)
        self._y_lumo = np.zeros(n, dtype=np.float64)
        for i, entry in enumerate(all_entries):
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                raise ValueError(f"Unparseable calibration SMILES: {entry['smiles']}")
            self._X[i] = _ecfp4_vector(mol)
            tom_homo, tom_lumo = predict_tom_orbitals(mol)
            self._y_homo[i] = entry["homo_eV"] - tom_homo
            self._y_lumo[i] = entry["lumo_eV"] - tom_lumo
        self._homo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(self._X, self._y_homo)
        self._lumo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(self._X, self._y_lumo)

    def predict_deltas(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (delta_homo, delta_lumo) mean residual predictions for a molecule."""
        x = _ecfp4_vector(mol).reshape(1, -1)
        return float(self._homo_model.predict(x)[0]), float(self._lumo_model.predict(x)[0])

    def predict_deltas_with_uncertainty(
        self, mol: Chem.Mol
    ) -> tuple[float, float, float, float]:
        """Return (delta_homo, delta_lumo, std_homo, std_lumo) for a molecule.

        The standard deviations quantify how far out-of-domain the molecule is
        in ECFP4 space relative to the calibration set. Large σ means the
        correction is uncertain and should be damped (see ``predict_corrected``).
        """
        x = _ecfp4_vector(mol).reshape(1, -1)
        d_homo, std_homo = self._homo_model.predict(x, return_std=True)
        d_lumo, std_lumo = self._lumo_model.predict(x, return_std=True)
        return (
            float(d_homo[0]),
            float(d_lumo[0]),
            float(std_homo[0]),
            float(std_lumo[0]),
        )

    def predict_corrected(
        self, mol: Chem.Mol, base: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Return corrected (homo_eV, lumo_eV) for a molecule.

        Falls back to raw TOM predictions if ``base`` is not supplied.
        Out-of-domain corrections are damped using the GPR uncertainty:
        when σ is large the residual is shrunk toward zero so the result
        reverts to raw TOM.
        """
        tom_homo, tom_lumo = base if base is not None else predict_tom_orbitals(mol)
        d_homo, d_lumo, std_homo, std_lumo = self.predict_deltas_with_uncertainty(mol)
        # Shrinkage factor: exp(-σ² / (2 * cutoff²)) maps σ∈[0,∞) → confidence∈(0,1].
        # A molecule at the calibration domain center (σ≈0) gets full correction;
        # a molecule far out-of-domain gets ≈ 0 correction (reverts to raw TOM).
        cutoff = 0.5
        conf_homo = float(np.exp(-(std_homo / cutoff) ** 2))
        conf_lumo = float(np.exp(-(std_lumo / cutoff) ** 2))
        return tom_homo + d_homo * conf_homo, tom_lumo + d_lumo * conf_lumo

    def update_online(
        self, smiles: str, homo_tom: float, lumo_tom: float,
        homo_dft: float, lumo_dft: float
    ) -> None:
        """Online update of the GPR residual model with new data.

        Physical justification: In active learning, low-confidence TOM predictions
        are escalated to xTB (and eventually DFT) for more accurate evaluation.
        The accumulated feedback should immediately update the Δ-correction
        model so that subsequent predictions benefit from this new information
        without waiting for a periodic refit. The incremental update preserves
        the probabilistic properties of the GPR while adapting to new data
        patterns.

        Args:
            smiles: SMILES string for the molecule
            homo_tom: Raw TOM HOMO prediction (eV)
            lumo_tom: Raw TOM LUMO prediction (eV)
            homo_dft: DFT HOMO reference (eV)
            lumo_dft: DFT LUMO reference (eV)
        """
        try:
            import sklearn.gaussian_process
            from sklearn.gaussian_process import GaussianProcessRegressor
            
            # Calculate residuals
            delta_homo = homo_dft - homo_tom
            delta_lumo = lumo_dft - lumo_tom
            
            # Convert SMILES to ECFP4 fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            fp_vec = _ecfp4_vector(mol).reshape(1, -1)
            
            # Online update using partial_fit (available in sklearn >= 1.0)
            # For older sklearn versions, we'll accumulate new data points
            if hasattr(self._homo_model, 'partial_fit'):
                # Use partial_fit for incremental updates
                self._homo_model.partial_fit(fp_vec, np.array([delta_homo]))
                self._lumo_model.partial_fit(fp_vec, np.array([delta_lumo]))
                self._X = np.vstack([self._X, fp_vec])
                self._y_homo = np.append(self._y_homo, delta_homo)
                self._y_lumo = np.append(self._y_lumo, delta_lumo)
            else:
                # For older sklearn, accumulate data and retrain on all accumulated data
                self._X = np.vstack([self._X, fp_vec])
                self._y_homo = np.append(self._y_homo, delta_homo)
                self._y_lumo = np.append(self._y_lumo, delta_lumo)
                
                # Retrain models with accumulated data
                self._homo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(self._X, self._y_homo)
                self._lumo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(self._X, self._y_lumo)
                
        except Exception as e:
            logger.debug(f"Online update failed for {smiles}: {e}")
            # Graceful degradation: if online update fails, continue without it

    def loo_mae(self) -> float:
        """Leave-one-out cross-validation MAE (mean of HOMO/LUMO prediction errors, eV).

        Physical justification: LOO gives an honest estimate of how the
        residual model generalizes to unseen molecules — each calibration
        molecule is scored by a model that never saw it, mirroring how the
        correction is applied to novel EA candidates.

        Uses the analytical LOO formula for GPR instead of brute-force
        refitting, reducing cost from O(n⁴) to O(n³).
        """
        errors = []
        n = len(self._X)
        jitter = 1e-2 * np.eye(n)
        loo_preds = []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for y, fit_model in [(self._y_homo, self._homo_model), (self._y_lumo, self._lumo_model)]:
                K = fit_model.kernel_(self._X) + jitter
                K_inv = np.linalg.solve(K, np.eye(n))
                centered_y = y - y.mean()
                alpha_vec = K_inv @ centered_y
                # LOO prediction: y_i - alpha_i / (K^{-1})_{ii}
                diag_K_inv = np.diag(K_inv)
                # Guard against division by zero for ill-conditioned diagonals
                diag_K_inv = np.where(np.isfinite(diag_K_inv) & (np.abs(diag_K_inv) > 1e-12), diag_K_inv, 1e-12)
                loo_pred = y - alpha_vec / diag_K_inv
                loo_preds.append(loo_pred)

        loo_pred_homo = loo_preds[0]
        loo_pred_lumo = loo_preds[1]

        for i in range(len(self._calib)):
            d_homo_loo = float(loo_pred_homo[i])
            d_lumo_loo = float(loo_pred_lumo[i])
            tom_homo, tom_lumo = predict_tom_orbitals(
                Chem.MolFromSmiles(self._calib[i]["smiles"])
            )
            pred_homo = tom_homo + d_homo_loo
            pred_lumo = tom_lumo + d_lumo_loo
            homo_err = abs(pred_homo - self._calib[i]["homo_eV"])
            lumo_err = abs(pred_lumo - self._calib[i]["lumo_eV"])
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


def compute_ood_spearman(
    ood_entries: list[dict[str, float]],
    model: DeltaCorrection | None = None,
) -> float:
    """Compute Spearman ρ between corrected HOMO predictions and DFT reference
    for out-of-distribution molecules.

    Physical justification: OOD Spearman ρ measures how well the Δ-correction
    model generalizes to novel chemical scaffolds not represented in the
    calibration set. A ρ > 0.30 indicates the correction captures meaningful
    electronic structure trends in OOD molecules, while ρ < 0.10 suggests
    the model is effectively reverting to raw TOM for novel scaffolds.

    Args:
        ood_entries: List of dicts with keys: smiles, homo_eV, lumo_eV.
        model: Optional DeltaCorrection instance. If None, uses the singleton.

    Returns:
        float: Spearman ρ between corrected HOMO and DFT HOMO for OOD molecules.
    """
    from scipy.stats import spearmanr

    if model is None:
        model = get_delta_correction()

    preds: list[float] = []
    refs: list[float] = []
    for entry in ood_entries:
        smiles = entry.get("smiles", "")
        ref_homo = entry.get("homo_eV")
        if ref_homo is None or not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        raw_h, raw_l = predict_tom_orbitals(mol)
        corr_h, corr_l = model.predict_corrected(mol, base=(raw_h, raw_l))
        preds.append(corr_h)
        refs.append(ref_homo)

    if len(preds) < 3:
        return 0.0
    rho, _ = spearmanr(preds, refs)
    return float(rho)
