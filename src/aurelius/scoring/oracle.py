"""PropertyOracle — Fragment-Additivity Group-Contribution Oracle.

Architecture
------------
A **single-tier fragment-additivity (group-contribution)** model for all
electrolyte-relevant properties:

  **Tier 1 — Fragment-Additivity GC (all properties)**
    Every property (HOMO, LUMO, Dielectric, Viscosity, Li+ Solvation) is
    modelled as a linear combination of substructure contributions from a
    curated SMARTS fragment table.  This is the simplest physically
    interpretable model: each functional group contributes a fixed additive
    shift to each property.  There is no machine-learning model, no
    p ≫ n problem, and no double-counting of F/P/S corrections.

    Why GC instead of a QM9-trained RF?
      - QM9 (~300 molecules after filtering) is too small for a 2053-dim
        feature space (p ≫ n).  An RF would memorise noise.
      - GC is interpretable, deterministic, and requires no training data.
      - The F/P/S inductive corrections previously applied as a separate
        layer are now embedded directly in the fragment table, eliminating
        the dual-heuristic overlap.

    Properties predicted:
      - HOMO energy (eV)          — fragment-additivity
      - LUMO energy (eV)          — fragment-additivity
      - Dielectric proxy          — fragment-additivity + TPSA-based cap
      - Viscosity proxy           — fragment-additivity + MW + rot. bonds
      - Li+ Solvation proxy       — fragment-additivity (new)

Usage:
    from aurelius.scoring.oracle import PropertyOracle
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = oracle.evaluate(ctx)
    print(result["homo_eV"])            # e.g. -7.6
    print(result["li_solvation_proxy"])  # e.g. 2.3
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.constants import MAX_DIELECTRIC_PER_TPSA
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_DATA_SOURCE: str = "fragment-additivity (Group Contribution)"


def get_data_source() -> str:
    """Return a human-readable string describing the oracle's data source."""
    return _DATA_SOURCE


# ---------------------------------------------------------------------------
# Fragment-Additivity (Group-Contribution) Models
# ---------------------------------------------------------------------------

# (smarts, name, homo_shift, lumo_shift, dielectric_contrib, viscosity_contrib, li_solvation_contrib)
# Li+ solvation contributions are based on donor-number and chelation ability:
#   - Carbonates bind moderately-strongly (high donor number ~16)
#   - Ethers bind moderately (glyme family chelates Li+)
#   - Nitriles bind moderately (acetonitrile donor number ~14)
#   - Fluorinated groups reduce binding (electron withdrawal lowers donor strength)
#   - Alcohols bind too strongly (high donor number, poor transference)
_GC_FRAGMENTS: list[tuple[str, str, float, float, float, float, float]] = [
    ("[CX3](=O)[OX2H0]",       "ester",              0.6, -0.4,  2.5,  0.6,  0.8),
    ("[CX3](=O)[OH]",          "carboxylic_acid",    0.3, -0.6,  4.0,  1.0,  1.8),
    ("[CX3](=O)[NX3]",         "amide",              0.8, -0.2,  5.0,  0.8,  1.2),
    ("[CX3](=O)[CX3]",         "ketone",             0.7, -0.8,  3.0,  0.5,  0.6),
    ("[CH](=O)",               "aldehyde",           0.5, -1.2,  2.5,  0.3,  0.3),
    ("O=C([OX2])[OX2]",        "carbonate",          1.0, -0.5,  5.0,  0.7,  1.5),
    ("[OD2]([CX4])[CX4]",      "ether",              0.4,  0.2,  1.5, -0.3,  0.5),
    ("[OH][CX4]",              "alcohol",            0.3,  0.1,  4.5,  1.2,  2.0),
    ("[NX3;H2][CX4]",          "primary_amine",      0.8,  0.5,  3.5,  0.5,  1.0),
    ("[NX3;H1]([CX4])[CX4]",   "secondary_amine",    0.9,  0.4,  2.5,  0.4,  0.8),
    ("[NX3;H0]([CX4])([CX4])[CX4]", "tertiary_amine", 0.5, 0.3,  1.5,  0.3,  0.5),
    ("[C]#[N]",                "nitrile",           -0.3, -0.6,  8.0,  0.4,  1.2),
    ("[CX3]=[CX3]",            "alkene",             0.8,  0.5,  0.5,  0.1,  0.1),
    ("[CX2]#[CX2]",            "alkyne",             0.6, -0.3,  1.0,  0.2,  0.2),
    ("[c]",                    "aromatic_carbon",    1.2,  1.0,  0.5,  0.5,  0.1),
    ("[F]",                    "fluorine",          -0.15, -0.08, 0.0,  0.1, -0.5),
    ("[Cl]",                   "chlorine",          -0.2, -0.15,  0.5,  0.2, -0.3),
    ("[Br]",                   "bromine",           -0.15, -0.1,   0.5,  0.3, -0.2),
    ("S(=O)(=O)[CX4]",         "sulfone",           -0.5, -1.2,  5.0,  0.5,  1.0),
    ("S(=O)(=O)[OX2]",         "sulfonate",         -0.6, -1.0,  5.5,  0.6,  1.2),
    ("S(=O)(=O)F",             "sulfonyl_fluoride", -0.7, -1.3,  4.0,  0.4,  0.5),
    ("[PX4](=O)([OX2])([OX2])[OX2]", "phosphate",   -0.5, -1.0,  4.0,  0.8,  1.5),
    ("[C](F)(F)F",             "trifluoromethyl",   -0.5, -0.3,  0.5,  0.2, -0.3),
    ("[C](F)(F)",              "difluoromethylene", -0.3, -0.2,  0.3,  0.1, -0.2),
    ("[BX3]([OX2])",           "boronate",          -0.4, -1.5,  2.0,  0.7,  1.0),
    ("[S]([CX4])[CX4]",        "thioether",          0.2, -0.3,  1.0,  0.2,  0.3),
]

_GC_BASE_HOMO: float = -9.2
_GC_BASE_LUMO: float = 2.8
_GC_BASE_DIELECTRIC: float = 1.9
_GC_BASE_VISCOSITY: float = 0.1
_GC_BASE_LI_SOLVATION: float = 1.0


def _count_fragments(mol: Chem.Mol) -> dict[str, int]:
    """Count occurrences of each fragment SMARTS pattern in a molecule."""
    counts: dict[str, int] = {}
    for _smarts, name, _dh, _dl, _dd, _dv, _ls in _GC_FRAGMENTS:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(_smarts))
        counts[name] = len(matches)
    return counts


def predict_fragment_additivity(mol: Chem.Mol) -> tuple[float, float]:
    """Predict HOMO/LUMO using fragment-additivity (group contribution) model."""
    counts = _count_fragments(mol)
    homo = _GC_BASE_HOMO
    lumo = _GC_BASE_LUMO
    for _smarts, _name, dh, dl, _dd, _dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        homo += n * dh
        lumo += n * dl
    return homo, lumo


def predict_dielectric_proxy(mol: Chem.Mol) -> float:
    """Predict a dielectric constant proxy via fragment-additivity + TPSA cap.

    Returns:
        Unitless dielectric proxy (typically 1–15 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_DIELECTRIC
    for _smarts, _name, _dh, _dl, dd, _dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dd

    from rdkit.Chem import rdMolDescriptors
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    value += tpsa * 0.02

    max_diel = _GC_BASE_DIELECTRIC + tpsa * MAX_DIELECTRIC_PER_TPSA
    value = min(value, max_diel)

    return max(1.0, value)


def predict_viscosity_proxy(mol: Chem.Mol) -> float:
    """Predict a viscosity proxy via fragment-additivity.

    Returns:
        Unitless viscosity proxy (typically 0.5–5.0 for electrolyte solvents).
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_VISCOSITY
    for _smarts, _name, _dh, _dl, _dd, dv, _ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * dv

    from rdkit.Chem import rdMolDescriptors
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    value += (mw - 30.0) * 0.005
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    value += n_rot * 0.15
    return max(0.1, value)


def predict_li_solvation_proxy(mol: Chem.Mol) -> float:
    """Predict a Li+ solvation energy proxy via fragment-additivity.

    Li+ solvation strength is modelled as a linear combination of
    functional-group donor abilities.  The proxy correlates with:
      - Donor number (DN): higher DN → stronger Li+ binding
      - Chelation: polydentate ethers (glymes) bind Li+ more strongly
        than monodentate analogues

    Returns:
        Unitless Li+ solvation proxy (typically 1.0–6.0).
            ~1.0–2.5 : weak binding (poor salt dissociation)
            ~2.5–4.5 : moderate binding (Goldilocks zone)
            ~4.5–6.0+ : strong binding (poor transference number)
    """
    counts = _count_fragments(mol)
    value = _GC_BASE_LI_SOLVATION
    for _smarts, _name, _dh, _dl, _dd, _dv, ls in _GC_FRAGMENTS:
        n = counts.get(_name, 0)
        value += n * ls

    from rdkit.Chem import rdMolDescriptors
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    value += max(0.0, (mw - 50.0)) * 0.002
    return max(0.5, value)


# ---------------------------------------------------------------------------
# RF Training (standalone utility — not used by PropertyOracle)
# ---------------------------------------------------------------------------
# Retained as a CLI-accessible utility for users who want to experiment
# with RF models once >10,000 training molecules are available. The main
# oracle pipeline uses pure GC to avoid the p ≫ n problem.


def train_oracle_rf(save_path: str = "models/oracle_rf.joblib") -> str:
    """Train a RandomForest regressor for HOMO/LUMO on the bundled QM9 data.

    .. warning::
       The bundled QM9 subset contains ~300 molecules with 2053 features.
       This is statistically unsound (p ≫ n). The trained RF will memorise
       noise. Only use this function when a dataset of >10,000 molecules
       is available.

    Uses ``MoleculeContext`` for featurization. Saves via joblib.
    """
    from pathlib import Path

    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from aurelius.data.loaders import load_qm9_homo_lumo_data

    logger.info("Loading QM9 HOMO/LUMO data...")
    data = load_qm9_homo_lumo_data()

    X_list: list[np.ndarray] = []
    y_homo: list[float] = []
    y_lumo: list[float] = []
    skipped = 0
    for smiles, homo, lumo in data:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            skipped += 1
            continue
        try:
            fp = ctx.get_feature_vector()
            X_list.append(fp)
            y_homo.append(homo)
            y_lumo.append(lumo)
        except Exception:
            skipped += 1
            continue

    if not X_list:
        raise RuntimeError("All QM9 molecules failed featurisation — cannot train RF model.")
    if skipped:
        logger.warning("Skipped %d/%d molecules that could not be featurised.", skipped, len(data))

    X = np.array(X_list, dtype=np.float32)
    y = np.column_stack([y_homo, y_lumo])

    logger.warning(
        "RF trained on only %d samples with %d features — this is statistically unsound. "
        "Provide a custom dataset with >= 10000 molecules for a meaningful QSPR model.",
        X.shape[0], X.shape[1],
    )
    logger.info("Training RandomForest on %d samples with %d features...", X.shape[0], X.shape[1])
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("RF model saved to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Tier3_QuantumOracle — xTB output parser (stub)
# ---------------------------------------------------------------------------


class Tier3QuantumOracle:
    """Stub interface for parsing xTB/DFT output to override GC proxies.

    When xTB single-point calculations are run on top-N candidates, this
    class parses the output files and extracts accurate HOMO/LUMO/dipole
    values, replacing the fragment-additivity estimates.

    Currently a stub.  Full implementation requires:
      - ``xtb.out`` parser for GFN2-xTB energy / HOMO / LUMO
      - ``xtbopt.log`` parser for optimised geometries
      - Dipole moment extraction from xTB output (``molecular dipole moment``)

    Usage (once implemented):
        >>> qc = Tier3_QuantumOracle()
        >>> result = qc.parse_xtb_out("xtb_input/001_CC.xyz/xtb.out")
        >>> result["homo_eV"]  # -7.23 (accurate, overriding GC's -7.8)
    """

    def __init__(self) -> None:
        self._parsed: dict[str, dict[str, float]] = {}

    def parse_xtb_out(self, path: str) -> dict[str, float] | None:
        """Parse a single xTB output file for HOMO/LUMO/dipole.

        Args:
            path: Path to ``xtb.out`` file.

        Returns:
            Dict with keys ``homo_eV``, ``lumo_eV``, ``dipole_D``
            or None if parsing fails.
        """
        try:
            with open(path) as f:
                lines = f.readlines()
        except (OSError, FileNotFoundError):
            logger.warning("Tier3_QuantumOracle: xTB output not found at %s", path)
            return None

        homo: float | None = None
        lumo: float | None = None
        dipole: float | None = None

        for line in lines:
            stripped = line.strip()
            if "HOMO" in stripped and "eV" in stripped:
                parts = stripped.split()
                for i, p in enumerate(parts):
                    if p == "HOMO" and i + 2 < len(parts):
                        with contextlib.suppress(ValueError, IndexError):
                            homo = float(parts[i + 2])
            if "LUMO" in stripped and "eV" in stripped:
                parts = stripped.split()
                for i, p in enumerate(parts):
                    if p == "LUMO" and i + 2 < len(parts):
                        with contextlib.suppress(ValueError, IndexError):
                            lumo = float(parts[i + 2])
            if "molecular dipole moment" in stripped.lower():
                parts = stripped.split()
                with contextlib.suppress(ValueError, IndexError):
                    dipole = float(parts[-1])

        if homo is not None and lumo is not None:
            result = {
                "homo_eV": homo,
                "lumo_eV": lumo,
                "dipole_D": dipole or 0.0,
            }
            self._parsed[path] = result
            logger.info("Tier3_QuantumOracle: parsed %s (HOMO=%.3f, LUMO=%.3f)", path, homo, lumo)
            return result

        logger.warning("Tier3_QuantumOracle: could not parse HOMO/LUMO from %s", path)
        return None

    def override(self, gc_result: dict[str, Any], qc_result: dict[str, float]) -> dict[str, Any]:
        """Override GC proxy values with quantum-chemical values.

        Args:
            gc_result: Result dict from PropertyOracle.evaluate().
            qc_result: Result dict from parse_xtb_out().

        Returns:
            Updated result dict with QC values replacing GC estimates.
        """
        override = dict(gc_result)
        override["homo_eV"] = round(qc_result["homo_eV"], 4)
        override["lumo_eV"] = round(qc_result["lumo_eV"], 4)
        override["gap_eV"] = round(qc_result["lumo_eV"] - qc_result["homo_eV"], 4)
        if qc_result.get("dipole_D", 0.0) > 0.0:
            override["dipole_D"] = round(qc_result["dipole_D"], 4)
        override["domain_reason"] = "quantum-chemical (xTB/DFT) override — replaces GC estimates"
        return override


# ---------------------------------------------------------------------------
# PropertyOracle — Pure Fragment-Additivity (Group-Contribution) Model
# ---------------------------------------------------------------------------


class PropertyOracle:
    """Multi-objective property oracle using pure fragment-additivity GC.

    Predicts five electrolyte-relevant properties:
      - HOMO energy (eV) — fragment-additivity
      - LUMO energy (eV) — fragment-additivity
      - Dielectric proxy — fragment-additivity + TPSA-based cap
      - Viscosity proxy — fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy — fragment-additivity

    All properties are predicted from a single curated fragment table
    (``_GC_FRAGMENTS``) with no machine-learning model.  This eliminates
    the p ≫ n statistical flaw of the previous hybrid RF + GC approach
    and removes the redundant F/P/S correction layer.
    """

    _CACHE: dict[str, dict[str, Any]] | None = None

    def __init__(self) -> None:
        pass

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Evaluate a molecule and return predicted properties.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dictionary with keys:
              - homo_eV (float)
              - lumo_eV (float)
              - gap_eV (float)
              - dielectric_proxy (float)
              - viscosity_proxy (float)
              - li_solvation_proxy (float)
              - domain_applicable (bool)
              - domain_reason (str)

        Raises:
            TypeError: If ctx is not a MoleculeContext.
        """
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )

        smiles = ctx.smiles
        if self._CACHE is not None and smiles in self._CACHE:
            return self._CACHE[smiles]

        homo, lumo = predict_fragment_additivity(ctx.mol)
        gap = lumo - homo
        dielectric = predict_dielectric_proxy(ctx.mol)
        viscosity = predict_viscosity_proxy(ctx.mol)
        li_solvation = predict_li_solvation_proxy(ctx.mol)

        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "dielectric_proxy": round(dielectric, 4),
            "viscosity_proxy": round(viscosity, 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "domain_applicable": True,
            "domain_reason": _DATA_SOURCE,
        }

        if self._CACHE is None:
            self._CACHE = {}
        self._CACHE[smiles] = result
        return result

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        """Convenience: parse SMILES then evaluate.

        Args:
            smiles: SMILES string.

        Returns:
            Result dict (same as evaluate()).
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        """Persist the cache to disk with joblib."""
        import joblib
        payload: dict[str, Any] = {
            "cache": self._CACHE,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: cache saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        """Load cached predictions from a joblib cache file.

        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        try:
            import joblib
            payload = joblib.load(path)
        except (FileNotFoundError, Exception) as exc:
            logger.debug("PropertyOracle: no cached oracle at %s (%s)", path, exc)
            return False

        self._CACHE = payload.get("cache")
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        """Clear the SMILES->properties cache."""
        if self._CACHE is not None:
            self._CACHE.clear()
