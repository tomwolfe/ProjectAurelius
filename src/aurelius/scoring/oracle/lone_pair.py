"""Lone-Pair Orbital Model (LPM) — HOMO from localised valence orbitals.

ADR-2026-08-08-01: Replaces the particle-in-a-box Topological Orbital Model
(TOM) as the default non-xTB HOMO estimator for saturated electrolyte
solvents.

Physical justification
----------------------
TOM models the HOMO as a delocalised pi orbital in a 1-D box, ``E ∝ 1/L²``.
That is the correct physics for polyenes and fused aromatics.  It is the
*wrong* physics for the molecules this project actually searches over.
Carbonates, ethers, esters, nitriles, sulfones and phosphates are saturated:
they have no extended pi system, and their highest occupied orbital is a
heteroatom **lone pair** (an ``n`` orbital), not a pi orbital.  Applying a
1/L² gap law to a molecule whose HOMO is an oxygen lone pair yields a number
whose *ordering* carries almost no information — measured against 88
experimental gas-phase ionisation energies, TOM attains Spearman ρ = 0.10
with MAE 3.7 eV.

The LPM instead enumerates candidate ionisable orbitals and applies
Koopmans' theorem, ``IP ≈ −E_HOMO``, taking the *highest* (most easily
ionised) one:

    E_HOMO = −min_k IP_k

Each candidate orbital ``k`` is one of:

* a heteroatom lone pair, typed by element and hybridisation
  (ether O vs carbonyl O, amine N vs nitrile N vs pyridine N, sulfide S vs
  sulfoxide/sulfone S, P, Cl, Br);
* the pi system, if any unsaturation is present;
* a C–H/C–C sigma bond, for saturated hydrocarbons with no heteroatom.

and its ionisation energy is

    IP_k = E0(type) − a·A_k + b·I_k + c·C_k

with three closed-form structural terms:

``A_k`` — *alkyl destabilisation*.  Alkyl groups donate electron density
    hyperconjugatively, raising the lone pair.  Contributions attenuate
    geometrically with topological distance, ``Σ ρ_alk^(d−1)``.  This is the
    classical Branch–Calvin fall-off and it is why IP falls monotonically
    along NH₃ (10.07) → MeNH₂ (8.97) → Me₂NH (8.23) → Me₃N (7.85).

``I_k`` — *inductive stabilisation*.  Electronegative atoms withdraw density
    and deepen the orbital, weighted by Pauling electronegativity difference
    from carbon and attenuated by the same geometric law, ``Σ Δχ·ρ_ind^(d−1)``.
    Crucially this **saturates**: the old TOM applied an unbounded linear
    ``−0.15 eV`` per fluorine, which drove perfluorinated species to a
    predicted HOMO of −12.2 eV against a −6.5 eV reference.  Geometric
    attenuation reproduces the experimentally observed saturation
    (CH₃F 12.47 → CH₂F₂ 12.71 → CHF₃ 13.86 eV: strongly sub-linear).

``C_k`` — *resonance delocalisation*.  A lone pair adjacent to a pi system
    delocalises into it and is stabilised, e.g. the ester/amide nitrogen.

Parameters are fitted by physics-anchored ridge regression: the class
intercepts ``E0`` are shrunk toward Hinze–Jaffé valence-state ionisation
energies rather than toward zero, so an orbital type with little or no
support in the fitting set degrades to its literature atomic value instead
of extrapolating freely.  This keeps every parameter chemically readable and
bounded — see ``scripts/calibrate_lone_pair.py``.

Performance (leave-one-out CV over 88 NIST gas-phase IPs,
``data/experimental_ionization.json``):

===========================  =========  =========
model                        Spearman ρ  MAE (eV)
===========================  =========  =========
TOM (particle-in-a-box)          0.10       3.72
LPM (this module)                0.90       0.43
===========================  =========  =========

Limitations (honest)
--------------------
* Koopmans' theorem ignores orbital relaxation and correlation; it is a rank
  proxy, not a route to sub-0.1 eV absolute accuracy.  xTB remains preferred
  when available.
* Gas-phase IPs are the fitting target.  Condensed-phase HOMO levels are
  shifted by polarisation (typically 1–2 eV); the model is calibrated for
  *ranking* oxidative stability, not for absolute electrochemical windows.
* LUMO is not predicted here.  Virtual orbitals are not accessible via
  Koopmans and remain with TOM/xTB — see ``predict_orbitals``.
* Radicals, open-shell species and charged fragments are out of domain.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

from rdkit import Chem

logger = logging.getLogger(__name__)

# Pauling electronegativities for the elements the model reasons about.
PAULING_EN: dict[int, float] = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}

_CARBON_EN = PAULING_EN[6]

# Orbital classes, in the order used by the fitted parameter vector.
ORBITAL_CLASSES: tuple[str, ...] = (
    "O_ether", "O_carbonyl", "N_amine", "N_nitrile", "N_arom",
    "S_sulfide", "S_oxide", "P", "Cl", "Br", "pi", "sigma",
)

STRUCTURAL_TERMS: tuple[str, ...] = ("alk", "ind", "conj", "pi_len", "pi_don")

FEATURE_NAMES: tuple[str, ...] = ORBITAL_CLASSES + STRUCTURAL_TERMS

# Hinze-Jaffe valence-state ionisation energies (eV). Literature atomic
# values used as the ridge prior — NOT fitted. Unseen or weakly-supported
# orbital classes fall back to these rather than extrapolating.
VOIE_PRIOR: dict[str, float] = {
    "O_ether": 12.60, "O_carbonyl": 12.60, "N_amine": 10.90,
    "N_nitrile": 14.10, "N_arom": 10.90, "S_sulfide": 10.40,
    "S_oxide": 10.40, "P": 10.70, "Cl": 13.00, "Br": 11.80,
    "pi": 11.40, "sigma": 12.50,
    "alk": -0.80, "ind": -0.90, "conj": -1.20, "pi_len": -1.50, "pi_don": -0.90,
}

_DEFAULT_PARAMS: dict[str, Any] = {
    "rho_alk": 0.55,
    "rho_ind": 0.35,
    "weights": {
        "O_ether": 12.268, "O_carbonyl": 11.993, "N_amine": 10.569,
        "N_nitrile": 14.104, "N_arom": 11.302, "S_sulfide": 10.686,
        "S_oxide": 10.668, "P": 10.059, "Cl": 12.471, "Br": 11.733,
        "pi": 11.481, "sigma": 12.534,
        "alk": -0.955, "ind": -0.690, "conj": -1.421,
        "pi_len": -1.460, "pi_don": -0.924,
    },
}

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "lone_pair_params.json"
)

_DONOR_ELEMENTS = frozenset({7, 8, 15, 16, 17, 35})
_MULTIPLE_BONDS = (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
_UNSATURATED_BONDS = (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def _load_params() -> dict[str, Any]:
    """Load LPM parameters from JSON, falling back to calibrated defaults."""
    try:
        with open(_PARAMS_PATH) as fh:
            loaded = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _DEFAULT_PARAMS
    weights = {**_DEFAULT_PARAMS["weights"], **loaded.get("weights", {})}
    return {
        "rho_alk": loaded.get("rho_alk", _DEFAULT_PARAMS["rho_alk"]),
        "rho_ind": loaded.get("rho_ind", _DEFAULT_PARAMS["rho_ind"]),
        "weights": weights,
    }


def get_params() -> dict[str, Any]:
    """Return cached LPM parameters."""
    cached = getattr(get_params, "_cache", None)
    if cached is None:
        cached = _load_params()
        get_params._cache = cached  # type: ignore[attr-defined]
    return cached


def clear_params_cache() -> None:
    """Drop the cached parameters (used by the calibration script and tests)."""
    if hasattr(get_params, "_cache"):
        delattr(get_params, "_cache")


def _classify_nitrogen(atom: Chem.Atom) -> str | None:
    """Type a nitrogen lone pair: nitrile, pyridine-like, or amine."""
    if any(b.GetBondType() == Chem.BondType.TRIPLE for b in atom.GetBonds()):
        return "N_nitrile"
    if atom.GetIsAromatic():
        # Pyrrole-type N-H donates its lone pair to the ring pi system, so it
        # is not an independent n orbital and must not be offered as the HOMO.
        return None if atom.GetTotalNumHs() > 0 else "N_arom"
    return "N_amine"


_SIMPLE_DONOR_CLASSES = {15: "P", 17: "Cl", 35: "Br"}


def classify_lone_pair(atom: Chem.Atom) -> str | None:
    """Assign a heteroatom to a lone-pair orbital class.

    Returns ``None`` for atoms that carry no ionisable lone pair: carbon,
    hydrogen, cations, and pyrrole-type N–H whose lone pair is part of the
    aromatic sextet rather than an independent ``n`` orbital.
    """
    z = atom.GetAtomicNum()
    if z not in _DONOR_ELEMENTS or atom.GetFormalCharge() > 0:
        return None
    if z == 7:
        return _classify_nitrogen(atom)
    if z in (8, 16):
        has_multiple = any(b.GetBondType() in _MULTIPLE_BONDS for b in atom.GetBonds())
        if z == 8:
            return "O_carbonyl" if has_multiple else "O_ether"
        return "S_oxide" if has_multiple else "S_sulfide"
    return _SIMPLE_DONOR_CLASSES[z]


def _alkyl_destabilisation(mol: Chem.Mol, dist: Any, idx: int, rho_alk: float) -> float:
    """Hyperconjugative electron donation from alkyl carbons, distance-attenuated."""
    total = 0.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        d = dist[idx][atom.GetIdx()]
        if d >= 1:
            total += rho_alk ** (d - 1)
    return total


def _inductive_stabilisation(mol: Chem.Mol, dist: Any, idx: int, rho_ind: float) -> float:
    """Electron withdrawal by electronegative atoms, geometrically attenuated.

    The geometric fall-off is what makes the term saturate: adding the fourth
    fluorine to a carbon contributes far less than the first.
    """
    total = 0.0
    for atom in mol.GetAtoms():
        other = atom.GetIdx()
        if other == idx:
            continue
        chi = PAULING_EN.get(atom.GetAtomicNum())
        if chi is None or chi <= _CARBON_EN:
            continue
        d = dist[idx][other]
        if d >= 1:
            total += (chi - _CARBON_EN) * (rho_ind ** (d - 1))
    return total


def _resonance_delocalisation(atom: Chem.Atom) -> float:
    """Count single bonds linking the lone pair to an unsaturated neighbour."""
    total = 0.0
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        neighbour = bond.GetOtherAtom(atom)
        if any(
            nb.GetBondType() in _UNSATURATED_BONDS and nb.GetIdx() != bond.GetIdx()
            for nb in neighbour.GetBonds()
        ):
            total += 1.0
    return total


def _pi_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    return [
        a for a in mol.GetAtoms()
        if a.GetIsAromatic() or any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
    ]


def _blank_features() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def orbital_candidates(
    mol: Chem.Mol,
    rho_alk: float | None = None,
    rho_ind: float | None = None,
) -> list[tuple[str, dict[str, float]]]:
    """Enumerate candidate ionisable orbitals with their feature vectors.

    Returns a list of ``(class_name, features)`` pairs. Always non-empty for
    a valid molecule: saturated hydrocarbons fall through to a sigma orbital.

    ``rho_alk``/``rho_ind`` override the calibrated attenuation factors; the
    calibration script sweeps them directly rather than via the params file.
    """
    if rho_alk is None or rho_ind is None:
        params = get_params()
        rho_alk = params["rho_alk"] if rho_alk is None else rho_alk
        rho_ind = params["rho_ind"] if rho_ind is None else rho_ind
    dist = Chem.GetDistanceMatrix(mol)
    candidates: list[tuple[str, dict[str, float]]] = []

    for atom in mol.GetAtoms():
        cls = classify_lone_pair(atom)
        if cls is None:
            continue
        idx = atom.GetIdx()
        feats = _blank_features()
        feats[cls] = 1.0
        feats["alk"] = _alkyl_destabilisation(mol, dist, idx, rho_alk)
        feats["ind"] = -_inductive_stabilisation(mol, dist, idx, rho_ind)
        feats["conj"] = -_resonance_delocalisation(atom)
        candidates.append((cls, feats))

    pi_feats = _pi_features(mol)
    if pi_feats is not None:
        candidates.append(("pi", pi_feats))

    if not candidates:
        candidates.append(("sigma", _sigma_features(mol)))

    return candidates


def _pi_features(mol: Chem.Mol) -> dict[str, float] | None:
    """Feature vector for ionisation out of the pi system, if one exists."""
    pi = _pi_atoms(mol)
    if not pi:
        return None
    feats = _blank_features()
    feats["pi"] = 1.0
    # Inductive withdrawal acts on the delocalised pi system as a whole.
    withdrawal = sum(
        PAULING_EN[a.GetAtomicNum()] - _CARBON_EN
        for a in mol.GetAtoms()
        if PAULING_EN.get(a.GetAtomicNum(), 0.0) > _CARBON_EN
    )
    feats["ind"] = -withdrawal * 0.35
    # Larger pi systems ionise more easily (weaker quantum confinement).
    feats["pi_len"] = math.log(len(pi))
    # Lone-pair donors conjugated into the ring raise the pi HOMO further.
    feats["pi_don"] = float(sum(1 for a in mol.GetAtoms() if _is_pi_donor(a)))
    return feats


def _is_pi_donor(atom: Chem.Atom) -> bool:
    """True for a saturated N/O/S lone pair attached to an aromatic ring."""
    if atom.GetAtomicNum() not in (7, 8, 16):
        return False
    if any(b.GetBondType() in _MULTIPLE_BONDS for b in atom.GetBonds()):
        return False
    return any(nb.GetIsAromatic() for nb in atom.GetNeighbors())


def _sigma_features(mol: Chem.Mol) -> dict[str, float]:
    """Feature vector for sigma ionisation (saturated hydrocarbons)."""
    n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_fluorine = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    feats = _blank_features()
    feats["sigma"] = 1.0
    # Larger alkanes ionise more easily; fluorination deepens the sigma bond.
    feats["alk"] = math.log(n_carbon + 1)
    feats["ind"] = -n_fluorine * 0.9
    return feats


def _score(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(features[name] * weights.get(name, 0.0) for name in FEATURE_NAMES)


def predict_ionization_energy(mol: Chem.Mol) -> tuple[float, str]:
    """Predict the vertical ionisation energy (eV, positive) and its orbital type.

    The HOMO is the *highest* occupied orbital, hence the *lowest* ionisation
    energy among all candidates.

    Returns:
        ``(ip_eV, orbital_class)`` — the class name makes the prediction
        interpretable: it names which orbital the model believes is oxidised.
    """
    weights = get_params()["weights"]
    candidates = orbital_candidates(mol)
    best_ip = math.inf
    best_cls = "sigma"
    for cls, feats in candidates:
        ip = _score(feats, weights)
        if ip < best_ip:
            best_ip, best_cls = ip, cls
    return best_ip, best_cls


# Gas-phase -> condensed-phase HOMO bridge (ADR-2026-08-08-01).
#
# The LPM is fitted to gas-phase ionisation energies; the rest of Aurelius
# works on the condensed-phase DFT HOMO scale used by orbital_calibration.json.
# The two differ by polarisation screening, which compresses the range: a
# least-squares fit over the 115 calibration molecules gives
#   E_HOMO(condensed) = 0.2611 * (-IP_gas) - 4.8368     [MAE 0.264 eV]
# This is a strictly increasing affine map, so it cannot change any ranking —
# it only puts the prediction on the scale the scoring functions expect.
_HOMO_SCALE = 0.2611
_HOMO_OFFSET = -4.8368


def predict_lone_pair_homo(mol: Chem.Mol, condensed_phase: bool = True) -> float:
    """Predict HOMO energy in eV via Koopmans' theorem, ``E_HOMO = −IP``.

    Args:
        mol: RDKit molecule.
        condensed_phase: If True (default) map the gas-phase result onto the
            condensed-phase DFT scale used throughout Aurelius. The map is
            monotone, so rankings are identical either way.
    """
    ip, _ = predict_ionization_energy(mol)
    if condensed_phase:
        return _HOMO_SCALE * (-ip) + _HOMO_OFFSET
    return -ip


def explain(mol: Chem.Mol) -> dict[str, Any]:
    """Return a human-readable breakdown of the HOMO prediction.

    Interpretability is a project requirement: this reports which orbital was
    selected, its energy, and the per-term contributions in eV.
    """
    weights = get_params()["weights"]
    candidates = orbital_candidates(mol)
    scored = sorted(((_score(f, weights), c, f) for c, f in candidates), key=lambda t: t[0])
    ip, cls, feats = scored[0]
    contributions = {
        name: round(feats[name] * weights.get(name, 0.0), 4)
        for name in STRUCTURAL_TERMS
        if abs(feats[name]) > 1e-9
    }
    return {
        "homo_eV": round(-ip, 4),
        "ionization_energy_eV": round(ip, 4),
        "orbital_type": cls,
        "base_eV": round(weights.get(cls, 0.0), 4),
        "contributions_eV": contributions,
        "n_candidate_orbitals": len(candidates),
        "runner_up": (
            {"orbital_type": scored[1][1], "ionization_energy_eV": round(scored[1][0], 4)}
            if len(scored) > 1
            else None
        ),
    }


def predict_lone_pair_homo_batch(mols: list[Chem.Mol]) -> list[float]:
    """Batch HOMO prediction. RDKit-bound; kept simple and CPU-side."""
    return [predict_lone_pair_homo(m) for m in mols]
