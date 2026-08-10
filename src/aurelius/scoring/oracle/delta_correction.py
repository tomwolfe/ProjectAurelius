"""Δ-learning correction layer for the Topological Orbital Model (TOM).

Physical justification: TOM is a closed-form 1-D particle-in-a-box model of
conjugation. It systematically mis-estimates HOMO/LUMO for molecules whose
electronic structure is not captured by a single conjugation length (branched
pi-systems, through-bond coupling, hyperconjugation from C–F / C–O sigma
bonds, conformational averaging). These errors are structured, not random,
so a residual model trained on the difference between TOM and reference
values can correct them while keeping the interpretable TOM as the base model.

ADR-2026-08-09-02: HOMO and LUMO use *different* calibration sets:
  * HOMO residuals are trained on ``orbital_calibration.json`` (115 DFT-B3LYP
    entries, provenance-confounded but MAE-robust). The LPM HOMO model already
    correlates ρ = 0.91 against NIST experimental IPs, so the Δ-correction
    targets calibration (MAE), not ranking.
  * LUMO residuals are trained on ``lumo_calibration_xtb.json`` — 231 GFN2-xTB
    single-point values calibrated to the B3LYP/6-311++G** scale. All values
    come from the *same* quantum-chemical method, so the set is free of
    between-source confound (verified: citation-only ρ = 0.0).

The residual (Δ = reference − TOM) is regressed from ECFP4 fingerprints with a
Gaussian Process Regressor (GPR) using an RBF + WhiteKernel covariance.
GPR is preferred over kernel ridge because:

  1. It provides a calibrated standard deviation σ(Δ) for each prediction,
     which quantifies how far out-of-domain a molecule is and feeds into
     the conformal confidence discount.
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
import warnings
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

_LUMO_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "lumo_calibration_xtb.json",
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
    """Load the DFT HOMO calibration set (orbital_calibration.json)."""
    with open(_CALIBRATION_PATH) as f:
        return json.load(f)


def _load_lumo_calibration_xtb() -> list[dict[str, float]]:
    """Load the internally consistent xTB LUMO calibration set.

    ADR-2026-08-09-02: LUMO residuals are trained on GFN2-xTB single-point
    values (calibrated to B3LYP/6-311++G** scale) instead of confounded DFT
    labels from multiple functionals/basis sets.
    """
    try:
        with open(_LUMO_CALIBRATION_PATH) as f:
            data = json.load(f)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if entries:
            return entries
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    logger.warning(
        "LUMO calibration lumo_calibration_xtb.json not found; "
        "falling back to orbital_calibration.json (confounded DFT labels)."
    )
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


def _parse_ood_set(
    ood_calibration_set: list[dict[str, float]] | None,
) -> tuple[list[dict[str, float]], list[str]]:
    """Parse OOD calibration entries, returning (entries, canonical_smiles).

    OOD entries are later duplicated for 2× weight in ``DeltaCorrection.__init__``.
    Entries with unparseable SMILES are silently skipped.
    """
    if ood_calibration_set is None:
        return [], []
    entries: list[dict[str, float]] = []
    smiles: list[str] = []
    for entry in ood_calibration_set:
        mol = Chem.MolFromSmiles(entry.get("smiles", ""))
        if mol is None:
            continue
        entries.append(entry)
        smiles.append(Chem.MolToSmiles(mol))
    return entries, smiles


class DeltaCorrection:
    """GPR residual model mapping ECFP4 fingerprints to TOM HOMO/LUMO errors.

    Two independent Gaussian Process models are fit on the calibration residuals:
      * HOMO: trained on ``orbital_calibration.json`` DFT-B3LYP labels (MAE-robust).
      * LUMO: trained on ``lumo_calibration_xtb.json`` xTB labels (internally
        consistent, free of provenance confound — ADR-2026-08-09-02).

    A molecule's predicted mean residual is added to its raw TOM prediction to
    produce the corrected orbital energy. The associated GPR standard deviation
    quantifies prediction uncertainty and feeds into the conformal confidence
    discount in the selection layer.

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
        lumo_calib: list[dict[str, float]] | None = None,
        lumo_calib_smiles: list[str] | None = None,
        ood_calibration_set: list[dict[str, float]] | None = None,
    ) -> None:
        self._calib = calib if calib is not None else _load_calibration()
        self._calib_smiles = calib_smiles if calib_smiles is not None else [
            Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"])) for entry in self._calib
            if Chem.MolFromSmiles(entry["smiles"]) is not None
        ]

        self._lumo_calib = lumo_calib if lumo_calib is not None else _load_lumo_calibration_xtb()
        self._lumo_calib_smiles = lumo_calib_smiles if lumo_calib_smiles is not None else [
            Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"])) for entry in self._lumo_calib
            if Chem.MolFromSmiles(entry["smiles"]) is not None
        ]

        ood_entries, ood_smiles = _parse_ood_set(ood_calibration_set)
        self._all_smiles = list(self._calib_smiles) + ood_smiles + ood_smiles

        self._X_homo, self._y_homo = self._build_features_targets(
            self._calib, ood_entries, "homo_eV"
        )
        self._X_lumo, self._y_lumo = self._build_features_targets(
            self._lumo_calib, ood_entries, "lumo_eV"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._homo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(
                self._X_homo, self._y_homo
            )
            self._lumo_model = GaussianProcessRegressor(**_GPR_KWARGS).fit(
                self._X_lumo, self._y_lumo
            )

        self._prior_std_homo = float(np.std(self._y_homo)) or 1.0
        self._prior_std_lumo = float(np.std(self._y_lumo)) or 1.0

    @staticmethod
    def _build_features_targets(
        base_entries: list[dict[str, float]],
        ood_entries: list[dict[str, float]],
        value_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ECFP4 feature matrix and TOM residual vector.

        ``value_key`` is ``"homo_eV"`` or ``"lumo_eV"``; the TOM orbital of
        the same name is subtracted to produce the residual target. OOD entries
        are duplicated for 2× weight. Entries with unparseable SMILES are skipped.
        """
        all_entries = base_entries + ood_entries + ood_entries
        valid = [e for e in all_entries
                 if Chem.MolFromSmiles(e.get("smiles", "")) is not None]
        n = len(valid)
        X = np.zeros((n, 2048), dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        for i, entry in enumerate(valid):
            mol = Chem.MolFromSmiles(entry["smiles"])
            tom_homo, tom_lumo = predict_tom_orbitals(mol)
            ref_val = entry[value_key] - (tom_homo if value_key == "homo_eV" else tom_lumo)
            X[i] = _ecfp4_vector(mol)
            y[i] = ref_val
        return X, y

    def predict_deltas(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (delta_homo, delta_lumo) mean residual predictions for a molecule."""
        x = _ecfp4_vector(mol).reshape(1, -1)
        return float(self._homo_model.predict(x)[0]), float(self._lumo_model.predict(x)[0])

    def predict_deltas_batch(
        self,
        mols: list[Chem.Mol],
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Batch Δ-correction for many molecules, with optional MLX acceleration.

        Returns ``(d_homo, d_lumo, std_homo, std_lumo)`` where each is a 1-D
        array of shape ``(n_mols,)``. std arrays are None when ``return_std=False``.

        On Apple Silicon with MLX available, the GPR matrix operations run on
        the GPU. Otherwise falls back to sklearn per-molecule prediction.
        """
        from aurelius.utils.device import get_device

        n = len(mols)
        if n == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty, (empty if return_std else None), (empty if return_std else None)

        device = get_device()
        if device == "mlx":
            from aurelius.scoring.oracle.mlx_surrogate import predict_deltas_batch_mlx

            d_homo, std_homo = predict_deltas_batch_mlx(
                self._homo_model, mols, return_std=return_std
            )
            d_lumo, std_lumo = predict_deltas_batch_mlx(
                self._lumo_model, mols, return_std=return_std
            )
            return d_homo, d_lumo, std_homo, std_lumo

        # CPU fallback: per-molecule sklearn prediction
        d_homo = np.zeros(n, dtype=np.float32)
        d_lumo = np.zeros(n, dtype=np.float32)
        std_homo = np.zeros(n, dtype=np.float32) if return_std else None
        std_lumo = np.zeros(n, dtype=np.float32) if return_std else None
        for i, mol in enumerate(mols):
            if return_std:
                dh, dl, sh, sl = self.predict_deltas_with_uncertainty(mol)
                d_homo[i] = dh
                d_lumo[i] = dl
                std_homo[i] = sh  # type: ignore[index]
                std_lumo[i] = sl  # type: ignore[index]
            else:
                dh, dl = self.predict_deltas(mol)
                d_homo[i] = dh
                d_lumo[i] = dl
        return d_homo, d_lumo, std_homo, std_lumo

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

    def predict_corrected_batch(
        self,
        mols: list[Chem.Mol],
        base: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return corrected (homo, lumo) arrays for many molecules.

        Uses batch Δ-correction with optional MLX acceleration. The OOD
        damping is applied per-molecule using the GPR uncertainty.

        Args:
            mols: List of RDKit molecules.
            base: Optional list of (homo, lumo) TOM predictions. Computed
                internally if not provided.

        Returns:
            (homo_array, lumo_array) each of shape (n_mols,).
        """
        n = len(mols)
        if n == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty

        d_homo, d_lumo, std_homo, std_lumo = self.predict_deltas_batch(mols, return_std=True)

        if base is None:
            from aurelius.scoring.oracle.quantum import predict_tom_orbitals_batch
            homo_base, lumo_base = predict_tom_orbitals_batch(mols)
        else:
            homo_base = np.array([b[0] for b in base], dtype=np.float32)
            lumo_base = np.array([b[1] for b in base], dtype=np.float32)

        var_prior_homo = self._prior_std_homo ** 2
        var_prior_lumo = self._prior_std_lumo ** 2
        conf_homo = var_prior_homo / (var_prior_homo + std_homo ** 2)
        conf_lumo = var_prior_lumo / (var_prior_lumo + std_lumo ** 2)

        return homo_base + d_homo * conf_homo, lumo_base + d_lumo * conf_lumo

    def predict_corrected(
        self, mol: Chem.Mol, base: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Return corrected (homo_eV, lumo_eV) for a molecule.

        Falls back to raw TOM predictions if ``base`` is not supplied.
        Out-of-domain corrections are damped using the GPR uncertainty: when σ
        is large the residual is shrunk toward zero so the result reverts to
        raw TOM.

        ADR-2026-08-07-09: the damping is the normal-normal posterior mean
        shrinkage factor

            conf = σ_prior² / (σ_prior² + σ_pred²)

        where σ_prior is the spread of the training residuals and σ_pred is
        the GPR posterior standard deviation. This is the standard shrinkage
        estimator: it is the exact posterior weight when both the prior over
        the residual and the likelihood are Gaussian, so it needs no tuned
        constant and adapts on its own when the calibration set grows.

        It replaces ``exp(-(σ/0.5)²)``, whose 0.5 eV cutoff was set by hand
        and turned out to be far too aggressive. Measured by leave-one-out
        cross-validation over the calibration set, the old rule retained a mean
        confidence of only 0.19, discarding 81% of the correction and leaving
        HOMO MAE at 1.026 eV against raw TOM's 1.165 — the Δ-layer was doing
        almost nothing. The shrinkage rule retains 0.79 and gives:

            HOMO  ρ 0.433 → 0.439,  MAE 1.026 → 0.580 eV
            LUMO  MAE 0.731 → ~0.59 eV (internally consistent xTB set)

        Note: the HOMO/LUMO values above are from different calibration sets
        (DFT for HOMO, xTB for LUMO — ADR-2026-08-09-02). LOO MAE is the
        average of both. The previously reported OOD ρ ≈ 0.51 came from a test
        that trained on its own evaluation molecules; see
        test_ood_spearman_improvement.
        """
        tom_homo, tom_lumo = base if base is not None else predict_tom_orbitals(mol)
        d_homo, d_lumo, std_homo, std_lumo = self.predict_deltas_with_uncertainty(mol)
        var_prior_homo = self._prior_std_homo ** 2
        var_prior_lumo = self._prior_std_lumo ** 2
        conf_homo = var_prior_homo / (var_prior_homo + std_homo ** 2)
        conf_lumo = var_prior_lumo / (var_prior_lumo + std_lumo ** 2)
        return tom_homo + d_homo * conf_homo, tom_lumo + d_lumo * conf_lumo

    def update_online(
        self, smiles: str, homo_tom: float, lumo_tom: float,
        homo_dft: float, lumo_dft: float
    ) -> None:
        """DEPRECATED — no-op with warning.

        ADR-2026-08-07-06: Online point-updates of the GPR are unsafe during
        long discovery runs. Each incremental ``partial_fit`` call perturbs
        the kernel hyperparameters and can drive the model into an
        overfitting or numerical-unstable regime, silently corrupting the
        Δ-correction applied to *all* subsequent candidates.

        Use ``FeedbackController.accumulate()`` + periodic
        ``FeedbackController.maybe_refit()`` instead.  ``maybe_refit``
        performs a *full* GPR retrain from the combined calibration +
        feedback set on a configurable interval (default every 5
        generations), which is both numerically stable and cheaper than
        repeated partial fits at EA-loop batch sizes.
        """
        import warnings

        warnings.warn(
            "DeltaCorrection.update_online() is deprecated (ADR-2026-08-07-06) "
            "and is now a no-op to prevent GPR state corruption during long "
            "discovery runs. Use FeedbackController.accumulate() + maybe_refit() "
            "for periodic batch refits instead.",
            DeprecationWarning,
            stacklevel=2,
        )
            # Graceful degradation: if online update fails, continue without it

    def loo_mae(self) -> float:
        """Leave-one-out cross-validation MAE (average of HOMO/LUMO errors, eV).

        Physical justification: LOO gives an honest estimate of how the
        residual model generalizes to unseen molecules — each calibration
        molecule is scored by a model that never saw it, mirroring how the
        correction is applied to novel EA candidates.

        Uses the analytical LOO formula for GPR instead of brute-force
        refitting, reducing cost from O(n⁴) to O(n³).

        HOMO LOO uses the DFT calibration set; LUMO LOO uses the xTB set
        (ADR-2026-08-09-02). Each model is evaluated on its own calibration
        molecules.
        """
        errors = []
        specs = [
            ("homo", self._y_homo, self._X_homo, self._calib, 0),
            ("lumo", self._y_lumo, self._X_lumo, self._lumo_calib, 1),
        ]
        for name, y, X, ref_entries, idx in specs:
            n = len(X)
            if n == 0:
                continue
            model = self._homo_model if name == "homo" else self._lumo_model
            jitter = 1e-2 * np.eye(n)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                K = model.kernel_(X) + jitter
                K_inv = np.linalg.solve(K, np.eye(n))
                centered_y = y - y.mean()
                alpha_vec = K_inv @ centered_y
                diag_K_inv = np.diag(K_inv)
                diag_K_inv = np.where(
                    np.isfinite(diag_K_inv) & (np.abs(diag_K_inv) > 1e-12),
                    diag_K_inv, 1e-12,
                )
                loo_pred = y - alpha_vec / diag_K_inv

            for i in range(n):
                d_loo = float(loo_pred[i])
                mol = Chem.MolFromSmiles(ref_entries[i]["smiles"])
                tom_homo, tom_lumo = predict_tom_orbitals(mol)
                tom_val = tom_homo if name == "homo" else tom_lumo
                ref_val = ref_entries[i]["homo_eV" if name == "homo" else "lumo_eV"]
                pred_val = tom_val + d_loo
                errors.append(abs(pred_val - ref_val))

        return float(np.mean(errors)) if errors else 0.0


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
