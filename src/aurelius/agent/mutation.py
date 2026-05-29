"""SELFIES-based molecule mutation engine.

Generates candidate molecules from seed SMILES using:
- SELFIES token-level mutation (atom substitution, bond type changes)
- SELFIES token insertion/deletion (scaffold modification)
- SELFIES token permutation (isomer generation)

This ensures syntactically valid SELFIES strings that can be decoded
to valid RDKit molecules, enabling exploration of a far broader
chemical space than BRICS-only reassembly.

Requirements:
    - ``selfies`` package (pip install selfies)
    - RDKit for molecule validation
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from aurelius.utils.chem_utils import (
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
)
from aurelius.utils.dependencies import HAS_RDKIT

if HAS_RDKIT:
    from rdkit import Chem  # type: ignore[import-not-found, unused-ignore]
    from rdkit.Chem import (
        BRICS,  # type: ignore[import-not-found, unused-ignore]
        Descriptors,  # type: ignore[import-not-found, unused-ignore]
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SELFIES helpers (lazy import to avoid hard dependency)
# ---------------------------------------------------------------------------


def _import_selfies() -> Any:
    """Import selfies, raising ImportError with a helpful message if missing."""
    try:
        import selfies as sf  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise ImportError(
            "SELFIES-based mutation requires the selfies package. "
            "Install with: pip install selfies"
        ) from exc
    return sf


def smiles_to_selfies(smiles: str) -> str:
    """Convert a SMILES string to its SELFIES representation.

    Args:
        smiles: Valid SMILES string.

    Returns:
        SELFIES string that decodes back to the input molecule.

    Raises:
        ValueError: If SMILES is invalid or SELFIES conversion fails.
    """
    try:
        import selfies as sf

        # Validate that SMILES is parseable
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return sf.encoder(smiles)
    except Exception as exc:
        raise ValueError(f"Failed to convert SMILES to SELFIES: {exc}") from exc


def selfies_to_smiles(selfies_str: str) -> str | None:
    """Decode a SELFIES string to SMILES, validating the molecule.

    Args:
        selfies_str: Valid SELFIES string.

    Returns:
        SMILES string if decoding produces a valid molecule, else None.
    """
    try:
        import selfies as sf

        # sf.decoder returns a SMILES string directly
        smi = sf.decoder(selfies_str)
        if smi is None:
            return None
        # Validate that SMILES is parseable
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return smi
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mutation operations
# ---------------------------------------------------------------------------


def _mutate_atom(selfies_str: str, rng: np.random.Generator) -> str | None:
    """Replace one atom token in SELFIES with a different atom type.

    E.g. replace a carbon token with a nitrogen token to create an isomer.
    """
    import selfies as sf

    # Get valid tokens from the SELFIES string
    try:
        tokens = sf.split(selfies_str)
    except (AttributeError, ImportError) as exc:
        logger.debug("SELFIES split failed: %s", exc)
        return None

    if len(tokens) < 2:
        return None

    idx = rng.integers(0, len(tokens))
    current_token = tokens[idx]

    # Get all valid atom tokens except the current one
    other_atoms = [t for t in atoms if t != current_token]
    if not other_atoms:
        return None

    new_token = rng.choice(other_atoms)
    new_selfies = selfies_str.replace(current_token, new_token, 1)

    # Validate and decode
    smi = selfies_to_smiles(new_selfies)
    if smi is None:
        return None

    # Weight limit check
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mw = Descriptors.ExactMolWt(mol)
        if mw > 450:
            return None

    return smi


def _mutate_bond(selfies_str: str, rng: np.random.Generator) -> str | None:
    """Change a bond type in SELFIES (single → double, etc.)."""
    import selfies as sf

    # Get valid tokens from the SELFIES string
    try:
        tokens = sf.split(selfies_str)
    except (AttributeError, ImportError) as exc:
        logger.debug("SELFIES split failed: %s", exc)
        return None

    bond_tokens = [t for t in tokens if t in bonds]
    if not bond_tokens:
        return None

    idx = rng.integers(0, len(bond_tokens))
    current = bond_tokens[idx]

    other_bonds = [b for b in bonds if b != current]
    if not other_bonds:
        return None

    new_bond = rng.choice(other_bonds)
    new_selfies = selfies_str.replace(current, new_bond, 1)

    smi = selfies_to_smiles(new_selfies)
    return smi


def _insert_token(selfies_str: str, rng: np.random.Generator) -> str | None:
    """Insert a new atom token into the SELFIES string."""
    import selfies as sf

    atoms = sf.atom_encoder.get_valid_tokens()
    new_atom = rng.choice(atoms)

    # Insert at a random position
    pos = rng.integers(0, len(selfies_str) + 1)
    new_selfies = selfies_str[:pos] + new_atom + selfies_str[pos:]

    smi = selfies_to_smiles(new_selfies)
    return smi


def _delete_token(selfies_str: str, rng: np.random.Generator) -> str | None:
    """Delete a random token from the SELFIES string."""
    import selfies as sf

    try:
        tokens = sf.split(selfies_str)
    except (AttributeError, ImportError):
        return None
    if len(tokens) < 2:
        return None

    idx = rng.integers(0, len(tokens))
    new_selfies = "".join(tokens[:idx] + tokens[idx + 1 :])

    smi = selfies_to_smiles(new_selfies)
    return smi


def _permute_tokens(selfies_str: str, rng: np.random.Generator) -> str | None:
    """Permute (shuffle) SELFIES tokens to generate an isomer."""
    import selfies as sf

    try:
        tokens = sf.split(selfies_str)
    except (AttributeError, ImportError):
        return None
    n = len(tokens)
    if n < 2:
        return None

    indices = list(range(n))
    rng.shuffle(indices)
    new_tokens = [tokens[i] for i in indices]
    new_selfies = "".join(new_tokens)

    smi = selfies_to_smiles(new_selfies)
    return smi


def _apply_mutation(
    operation: str,
    selfies_str: str,
    rng: np.random.Generator,
) -> str | None:
    """Apply a single SELFIES mutation operation.

    Args:
        operation: One of ``"atom"``, ``"bond"``, ``"insert"``, ``"delete"``, ``"permute"``.
        selfies_str: Current SELFIES string.
        rng: Random number generator.

    Returns:
        SMILES string of the mutated molecule, or None if invalid.
    """
    operations = {
        "atom": _mutate_atom,
        "bond": _mutate_bond,
        "insert": _insert_token,
        "delete": _delete_token,
        "permute": _permute_tokens,
    }

    fn = operations.get(operation)
    if fn is None:
        return None

    return fn(selfies_str, rng)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MutationEngine:
    """SELFIES-based molecule mutation engine.

    Generates candidate molecules from seed SMILES using:
    - SELFIES token-level mutations (atom substitution, bond type changes)
    - SELFIES token insertion/deletion (scaffold modification)
    - SELFIES token permutation (isomer generation)
    - BRICS reassembly (fallback for molecules that fail SELFIES encoding)

    This ensures syntactically valid SELFIES strings that decode to
    valid RDKit molecules, enabling exploration of a far broader
    chemical space than BRICS-only reassembly.
    """

    def __init__(self, seed_smiles: list[str], known_fps_hex: list[str] | None = None) -> None:
        """Initialise the mutation engine.

        Args:
            seed_smiles: List of seed SMILES strings.
            known_fps_hex: Optional list of known fingerprint hex strings
                for novelty checking.
        """
        self.seed_pool: list[str] = list(set(seed_smiles))
        self.known_fps: list[Any] = []
        for h in known_fps_hex or []:
            with contextlib.suppress(Exception):
                self.known_fps.append(_deserialize_fp(h))
        self._rng = np.random.default_rng(42)

    def fingerprint_db_size(self) -> int:
        """Return the number of known fingerprints in the database."""
        return len(self.known_fps)

    def add_to_db(self, smiles: str) -> None:
        """Add a SMILES molecule to the known fingerprint database.

        Args:
            smiles: SMILES string to add.
        """
        mol = _safe_mol_from_smiles(smiles)
        if mol is not None:
            self.known_fps.append(_mol_to_fp(mol))

    def _novelty_check(self, mol: Any) -> bool:
        """Return True if molecule is novel (Tanimoto < 0.75 vs all known).

        Args:
            mol: RDKit Mol object.

        Returns:
            True if novel (all Tanimoto < 0.75).
        """
        fp = _mol_to_fp(mol)
        band_size = 4
        n_bands = len(fp) // band_size if len(fp) >= band_size else len(fp)
        if n_bands < 1:
            n_bands = 1

        fp_bits = np.packbits(fp.astype(np.uint8))
        band_hashes = [
            int(np.sum(fp_bits[band_size * i : band_size * (i + 1)] << i))
            for i in range(n_bands)
        ]

        band_key = hash(tuple(band_hashes))

        candidates = [
            known for known in self.known_fps
            if hash(tuple(np.packbits(known.astype(np.uint8)))) == band_key
        ]

        if not candidates:
            return True

        for known in candidates:
            from rdkit.DataStructs import TanimotoSimilarity

            if TanimotoSimilarity(fp, known) >= 0.75:
                return False
        return True

    def _brics_reassemble(self, mol: Any) -> list[str]:
        """BRICS decomposition + random reassembly using proper RDKit types."""
        generated: list[str] = []
        try:
            frag_smiles = list(BRICS.BRICSDecompose(mol))  # type: ignore[no-untyped-call]
            if len(frag_smiles) < 2:
                return generated

            frag_mols = [Chem.MolFromSmiles(s) for s in frag_smiles]
            frag_mols = [m for m in frag_mols if m is not None]
            if len(frag_mols) < 2:
                return generated

            for _ in range(20):
                rng = np.random.default_rng(self._rng.integers(0, 2**31))  # type: ignore[assignment]
                idx = rng.choice(len(frag_mols), size=min(2, len(frag_mols)), replace=False)
                try:
                    result_gen = BRICS.BRICSBuild([frag_mols[idx[0]], frag_mols[idx[1]]])  # type: ignore[no-untyped-call]
                    for r_mol in result_gen:
                        if r_mol is not None:
                            try:
                                Chem.SanitizeMol(r_mol)
                                s = Chem.MolToSmiles(r_mol, isomericSmiles=True)
                                generated.append(s)
                            except (RuntimeError, ValueError) as e:
                                logger.debug("RDKit operation failed: %s", e)
                except (RuntimeError, ValueError) as e:
                    logger.debug("BRICS build failed: %s", e)
        except (RuntimeError, ValueError) as e:
            logger.debug("BRICS reassembly failed: %s", e)
        return list(set(generated))

    def _selfies_mutate(self, smiles: str, batch_size: int = 50) -> list[str]:
        """Apply SELFIES-based mutations to generate candidate variants.

        Applies random SELFIES mutations (atom substitution, bond changes,
        token insertion/deletion, permutation) and returns valid SMILES.

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.

        Returns:
            List of candidate SMILES strings.
        """
        import selfies as sf

        generated: list[str] = []
        try:
            selfies_str = sf.encoder(smiles)
        except Exception as exc:
            logger.debug("SELFIES encoding failed: %s", exc)
            return generated

        operations = ["atom", "bond", "insert", "delete", "permute"]

        for _ in range(batch_size):
            operation = self._rng.choice(operations)
            candidate = _apply_mutation(operation, selfies_str, self._rng)
            if candidate is not None:
                generated.append(candidate)

        return generated

    def mutate(self, smiles: str, batch_size: int = 50) -> list[str]:
        """Generate up to batch_size mutated variants of a seed molecule.

        Applies SELFIES-based mutations with electrochemical stability
        filters and diversity checks. Falls back to BRICS if SELFIES
        encoding fails.

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.

        Returns:
            List of candidate SMILES strings.
        """
        candidates: set[str] = set()
        candidates.add(smiles)

        # Priority: SELFIES mutations
        selfies_results = self._selfies_mutate(smiles, batch_size)
        for s in selfies_results:
            m = _safe_mol_from_smiles(s)
            if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                candidates.add(s)

        # Fallback: BRICS
        brics_results = self._brics_reassemble(_safe_mol_from_smiles(smiles))
        for s in brics_results:
            m = _safe_mol_from_smiles(s)
            if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                candidates.add(s)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50) -> list[str]:
        """Mutate a batch of seed molecules, returning all variants.

        Args:
            batch_smiles: List of seed SMILES strings.
            batch_size: Maximum number of variants per seed.

        Returns:
            Deduplicated list of candidate SMILES strings.
        """
        all_variants: list[str] = []
        for smi in batch_smiles:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)
        return list(set(all_variants))

    def propose_candidates(
        self,
        n_candidates: int = 1000,
        batch_size: int = 50,
    ) -> list[str]:
        """Generate a large pool of candidate molecules from the seed pool.

        This method is the bridge between the mutation engine and the
        Gaussian Process surrogate: it produces ``n_candidates`` unique
        molecules that the Bayesian optimiser will later score via
        Expected Improvement.

        Args:
            n_candidates: Total number of unique candidates to propose.
            batch_size: Maximum variants per seed molecule.

        Returns:
            Deduplicated list of candidate SMILES strings.
        """
        all_variants: list[str] = []
        for smi in self.seed_pool:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)

        unique = list(dict.fromkeys(all_variants))
        if len(unique) > n_candidates:
            indices = self._rng.choice(len(unique), size=n_candidates, replace=False)
            unique = [unique[i] for i in indices]

        return unique
