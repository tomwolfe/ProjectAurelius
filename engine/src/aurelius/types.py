"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.

MoleculeContext is the absolute single source of truth for molecular
parsing. No module should call ``Chem.MolFromSmiles`` outside of this
class or the mutation engine's fragment pool.

Binary mixtures are represented as compound SMILES: ``SMILES_A|SMILES_B|frac_A``
where ``frac_A`` is the volume fraction of the first component in [0.1, 0.9].

Ternary mixtures are represented as: ``SMILES_A|SMILES_B|SMILES_C|frac_A|frac_B``
where ``frac_A`` and ``frac_B`` are the volume fractions of the first two
components, and the third fraction is ``1.0 - frac_A - frac_B``. All fractions
must be in (0.0, 1.0) and sum to 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from aurelius.cache.lru import LRUCache

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

# Return type for parsed mixtures: binary -> (a, b, frac_a) or ternary -> (a, b, c, frac_a, frac_b)
ParsedMixture = tuple[str, str, float] | tuple[str, str, str, float, float]


class MoleculeParseError(ValueError):
    """Raised when a SMILES string cannot be parsed into a valid molecule."""


class SanitizationError(MoleculeParseError):
    """Raised when RDKit sanitization fails (e.g. valence error, kekulization)."""


class UnmatchedParenthesesError(MoleculeParseError):
    """Raised when SMILES contains unmatched parentheses."""


class UnmatchedBracketsError(MoleculeParseError):
    """Raised when SMILES contains unmatched square brackets."""


class UnmatchedRingDigitsError(MoleculeParseError):
    """Raised when SMILES contains unmatched ring digits."""


class EmpiricalFeedbackEntry(TypedDict):
    """Empirical wet-lab feedback entry for GC UQ ensemble retraining.

    Each entry maps a SMILES string to its experimentally measured
    dielectric constant and viscosity values. Used by
    ``GcUqEnsemble.append_empirical_data()`` to retrain the UQ ensemble
    with real-world data, reducing prediction uncertainty for fed-back
    molecules.
    """

    smiles: str
    dielectric_constant: float
    viscosity_cP: float


@dataclass(frozen=True)
class ScreeningResult:
    """Result from a single molecule screening."""

    smiles: str
    total_score: float
    is_viable: bool
    rejection_reasons: list[str]
    fingerprint: Any = None
    novelty_to_seed: float | None = None
    homo_eV: float | None = None
    lumo_eV: float | None = None
    dielectric_proxy: float | None = None
    viscosity_proxy: float | None = None
    li_solvation_proxy: float | None = None
    sa_score: float | None = None
    li_dissociation_proxy: float | None = None
    sub_scores: dict[str, float] | None = None
    estimated_cost_score: float | None = None
    uncertainty_score: float | None = None


# ---------------------------------------------------------------------------
# Typed oracle and scoring result structures for compile-time verification
# ---------------------------------------------------------------------------


class QuantumResult(NamedTuple):
    """Typed internal result from quantum/surrogate evaluation."""

    homo_eV: float
    lumo_eV: float
    gap_eV: float
    method: str
    confidence: str
    li_binding_energy_kcal: float = 0.0
    cluster_homo_eV: float = 0.0
    cluster_lumo_eV: float = 0.0


class GcResult(NamedTuple):
    """Typed internal result from GC property evaluation."""

    gc_props: dict[str, float]
    uq_penalty: float
    diel_std: float
    visc_std: float


class DomainResult(NamedTuple):
    """Typed result from domain penalty computation."""

    domain_penalty: float
    domain_reason_str: str
    domain_applicable: bool


class SeiResult(NamedTuple):
    """Typed result from SEI motif penalty computation.

    Includes both the SEI-specific penalty/reason and the domain applicability
    flag, ensuring callers can update state atomically.
    """

    sei_penalty: float
    sei_reason: str
    domain_applicable: bool


class OodResult(NamedTuple):
    """Typed result from OOD (out-of-domain) penalty computation."""

    ood_penalty: float
    ood_reason: str


class QuantumEvaluation(NamedTuple):
    """Fully typed quantum evaluation result.

    Fields match the dict keys returned by ``_evaluate_quantum()``
    for compile-time verification.
    """

    homo_eV: float
    lumo_eV: float
    gap_eV: float
    surrogate_penalty: float
    s_homo: float
    skip_quantum: bool
    quantum_method: str
    quantum_confidence_val: str
    li_binding_energy_kcal: float = 0.0
    cluster_homo_eV: float = 0.0
    cluster_lumo_eV: float = 0.0


class GcEvaluation(NamedTuple):
    """Fully typed GC evaluation result.

    Fields match the dict keys returned by ``_evaluate_gc()``
    for compile-time verification.
    """

    gc_props: dict[str, float]
    uq_penalty: float
    diel_std: float
    visc_std: float


class EvaluationResult(NamedTuple):
    """Fully typed oracle evaluation result.

    Fields match the dict keys returned by ``_assemble_result()``
    for compile-time verification. Convert to dict via ``._asdict()`` for
    JSON serialization.
    """

    homo_eV: float
    lumo_eV: float
    gap_eV: float
    domain_applicable: bool
    domain_drift_risk: bool
    domain_reason: str
    domain_penalty: float
    quantum_method: str
    quantum_confidence: str
    uncertainty_flag: bool
    uncertainty_score: float
    dielectric_proxy: float
    viscosity_proxy: float
    li_solvation_proxy: float
    ced_proxy: float
    li_dissociation_proxy: float
    hydrolysis_risk_proxy: float
    surrogate_skipped: bool = False
    sei_fracture_proxy: float = 0.0
    gas_evolution_proxy: float = 0.0
    li_binding_energy_kcal: float = 0.0
    cluster_homo_eV: float = 0.0
    cluster_lumo_eV: float = 0.0


class OracleEvaluation(NamedTuple):
    """Fully typed oracle evaluation result.

    Fields match the dict keys returned by ``PropertyOracle.evaluate()``
    for compile-time verification. Convert to dict via ``._asdict()`` for
    JSON serialization.
    """

    homo_eV: float
    lumo_eV: float
    gap_eV: float
    domain_applicable: bool
    domain_drift_risk: bool
    domain_reason: str
    domain_penalty: float
    quantum_method: str
    quantum_confidence: str
    uncertainty_flag: bool
    uncertainty_score: float
    dielectric_proxy: float
    viscosity_proxy: float
    li_solvation_proxy: float
    ced_proxy: float
    li_dissociation_proxy: float
    hydrolysis_risk_proxy: float
    surrogate_skipped: bool = False
    sei_fracture_toughness_proxy: float = 0.0
    gas_evolution_proxy: float = 0.0


class ScoreResult(NamedTuple):
    """Typed score result from multi-objective composite scoring."""

    total_score: float
    is_viable: bool
    sub_scores: dict[str, float]
    sa_score: float
    rejection_reasons: list[str]


# ---------------------------------------------------------------------------
# Module-level LRU caches for MoleculeContext lazy computations
# These replace the mutable dataclass fields, preserving lazy evaluation
# while keeping MoleculeContext fully immutable.
# ---------------------------------------------------------------------------

_molecule_ctx_cache: LRUCache[MoleculeContext | None] = LRUCache(maxsize=4096)
"""Thread-safe LRU cache for MoleculeContext.from_smiles()."""

_ecfp4_cache: LRUCache[Any] = LRUCache(maxsize=4096)
"""Thread-safe LRU cache for computed ECFP4 fingerprints."""

_feature_vector_cache: LRUCache[Any] = LRUCache(maxsize=4096)
"""Thread-safe LRU cache for computed 2053-dim feature vectors."""

_gc_feature_vector_cache: LRUCache[Any] = LRUCache(maxsize=4096)
"""Thread-safe LRU cache for computed GC fragment feature vectors."""


@dataclass(frozen=True)
class MoleculeContext:
    """Unified molecular context — parsed exactly once per screening step.

    Holds the SMILES string and its pre-parsed RDKit Mol object.
    All derived properties are lazily computed via ``cached_property``
    (``mw``, ``logp``, ``tpsa``, ``ring_count``, ``rotatable_bonds``,
    ``hbd``, ``hba``) or via module-level LRU caches (fingerprints,
    feature vectors).

    Fully immutable after creation — no ``object.__setattr__`` mutations.

    Usage:
        ctx = MoleculeContext.from_smiles("CCO")
        Pipeline.screen_molecule(ctx)
    """

    smiles: str
    mol: Chem.Mol

    @classmethod
    def from_smiles(cls, smiles: str) -> MoleculeContext | None:
        """Parse a SMILES string into a ``MoleculeContext``.

        Uses a thread-safe LRU cache (maxsize=4096) to avoid redundant
        RDKit parsing. Returns ``None`` (with a logged error message)
        on failure.  For a version that raises specific exceptions, use
        ``from_smiles_strict()``.
        """
        cached = _molecule_ctx_cache.get(smiles)
        if cached is not None:
            return cached
        result, error = cls._from_smiles_impl(smiles)
        if error:
            logger.error(error)
        _molecule_ctx_cache.put(smiles, result)
        return result

    @classmethod
    def cache_clear(cls) -> None:
        """Clear the global LRU cache for ``from_smiles``."""
        _molecule_ctx_cache.clear()
        _ecfp4_cache.clear()
        _feature_vector_cache.clear()
        _gc_feature_vector_cache.clear()

    @classmethod
    def from_smiles_strict(cls, smiles: str) -> MoleculeContext:
        """Parse a SMILES string, raising on failure.

        Raises:
            MoleculeParseError: If the SMILES cannot be parsed.
            SanitizationError: If RDKit sanitization fails.
            UnmatchedParenthesesError: If parentheses are unmatched.
            UnmatchedBracketsError: If brackets are unmatched.
            UnmatchedRingDigitsError: If ring digits are unmatched.
        """
        result, error = cls._from_smiles_impl(smiles)
        if result is not None:
            return result
        if "Unmatched parentheses" in (error or ""):
            raise UnmatchedParenthesesError(error)
        if "Unmatched brackets" in (error or ""):
            raise UnmatchedBracketsError(error)
        if "Unmatched ring" in (error or ""):
            raise UnmatchedRingDigitsError(error)
        if "Sanitization failed" in (error or ""):
            msg = (error or "Sanitization failed").replace("Sanitization failed for ", "").replace("':", ":")
            raise SanitizationError(msg)
        raise MoleculeParseError(error or f"Unknown error parsing SMILES: '{smiles}'")

    @classmethod
    def _from_smiles_impl(cls, smiles: str) -> tuple[MoleculeContext | None, str | None]:
        """Internal: parse SMILES, return (context, error_string)."""
        # Fast pure-Python pre-check: catch hypervalent species before RDKit.
        # (Lazy import to avoid circular dependency via screening/tier1/filter.py)
        from aurelius.screening.structural import is_structurally_viable as _viable
        if not _viable(smiles):
            return None, f"Structural pre-check failed for SMILES: '{smiles}'"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if smiles.count("(") != smiles.count(")"):
                return None, f"Unmatched parentheses in SMILES: '{smiles}'"
            if smiles.count("[") != smiles.count("]"):
                return None, f"Unmatched brackets in SMILES: '{smiles}'"
            if any(d in smiles for d in "0123456789"):
                return None, f"Unmatched ring digit in SMILES: '{smiles}'"
            return None, f"RDKit could not parse SMILES: '{smiles}'"
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            return None, f"Sanitization failed for '{smiles}': {exc}"
        return cls(smiles=smiles, mol=mol), None

    @classmethod
    def from_brics_fragment(cls, smiles: str) -> MoleculeContext | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return cls(smiles=smiles, mol=mol)

    def get_ecfp4(self) -> Any:
        cached = _ecfp4_cache.get(self.smiles)
        if cached is not None:
            return cached
        fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, radius=2, nBits=2048)
        _ecfp4_cache.put(self.smiles, fp)
        return fp

    @cached_property
    def mw(self) -> float:
        return float(Descriptors.ExactMolWt(self.mol))

    @cached_property
    def logp(self) -> float:
        return float(Descriptors.MolLogP(self.mol))

    @cached_property
    def tpsa(self) -> float:
        return float(Descriptors.TPSA(self.mol))

    @cached_property
    def ring_count(self) -> int:
        return Descriptors.RingCount(self.mol)

    @cached_property
    def rotatable_bonds(self) -> int:
        return Descriptors.NumRotatableBonds(self.mol)

    @cached_property
    def hbd(self) -> int:
        return Descriptors.NumHDonors(self.mol)

    @cached_property
    def hba(self) -> int:
        return Descriptors.NumHAcceptors(self.mol)

    def get_feature_vector(self) -> Any:
        """Get or compute 2053-dim feature vector (lazy).

        Layout:
          - [0:2048]  ECFP4 binary fingerprint (Morgan radius=2, 2048 bits)
          - [2048]    Exact molecular weight
          - [2049]    MolLogP
          - [2050]    TPSA
          - [2051]    Ring count
          - [2052]    NumRotatableBonds

        Results are cached in a module-level LRU cache keyed by SMILES.
        """
        cached = _feature_vector_cache.get(self.smiles)
        if cached is not None:
            return cached
        import numpy as np
        fp = self.get_ecfp4()
        arr = np.zeros(2053, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        arr[2048] = self.mw
        arr[2049] = self.logp
        arr[2050] = self.tpsa
        arr[2051] = self.ring_count
        arr[2052] = self.rotatable_bonds
        _feature_vector_cache.put(self.smiles, arr)
        return arr

    def is_valid_electrolyte_mol(self) -> bool:
        if self.mw < 30.0 or self.mw > 1000.0:
            return False
        return self.hba >= 1

    def count_heteroatoms(self) -> dict[int, int]:
        counts: dict[int, int] = {8: 0, 9: 0, 15: 0, 16: 0}
        for atom in self.mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z in counts:
                counts[z] += 1
        return counts


# ---------------------------------------------------------------------------
# Binary Mixture Support
# ---------------------------------------------------------------------------

_MIXTURE_SEP: str = "|"


def is_mixture_smiles(smiles: str) -> bool:
    """Check if a SMILES string represents a mixture (binary or ternary, contains '|')."""
    return _MIXTURE_SEP in smiles


def parse_mixture_smiles(smiles: str) -> ParsedMixture | None:
    """Parse a mixture SMILES string.

    Binary: ``SMILES_A|SMILES_B|frac_A`` -> ``(smiles_a, smiles_b, frac_a)``
    Ternary: ``SMILES_A|SMILES_B|SMILES_C|frac_A|frac_B`` -> ``(smiles_a, smiles_b, smiles_c, frac_a, frac_b)``

    Returns None if parsing fails.
    """
    try:
        parts = smiles.split(_MIXTURE_SEP)
        if len(parts) == 3:
            smi_a, smi_b, frac_str = parts
            frac = float(frac_str)
            if not (0.0 <= frac <= 1.0):
                return None
            return smi_a, smi_b, frac
        elif len(parts) == 5:
            smi_a, smi_b, smi_c, frac_a_str, frac_b_str = parts
            frac_a = float(frac_a_str)
            frac_b = float(frac_b_str)
            if not (0.0 <= frac_a <= 1.0 and 0.0 <= frac_b <= 1.0):
                return None
            frac_c = 1.0 - frac_a - frac_b
            if frac_c < 0.0 or frac_c > 1.0:
                return None
            return smi_a, smi_b, smi_c, frac_a, frac_b
        return None
    except (ValueError, TypeError):
        return None


def format_mixture_smiles(
    smi_a: str,
    smi_b: str,
    frac_a: float,
    smi_c: str | None = None,
    frac_b: float | None = None,
) -> str:
    """Format a mixture SMILES string.

    Binary: ``SMILES_A|SMILES_B|frac_A``
    Ternary: ``SMILES_A|SMILES_B|SMILES_C|frac_A|frac_B``
    """
    if smi_c is not None and frac_b is not None:
        return f"{smi_a}{_MIXTURE_SEP}{smi_b}{_MIXTURE_SEP}{smi_c}{_MIXTURE_SEP}{frac_a:.4f}{_MIXTURE_SEP}{frac_b:.4f}"
    return f"{smi_a}{_MIXTURE_SEP}{smi_b}{_MIXTURE_SEP}{frac_a:.4f}"
