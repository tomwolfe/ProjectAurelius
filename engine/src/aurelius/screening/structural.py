"""Pure-Python structural viability pre-check for SMILES strings.

Provides ``is_structurally_viable(smiles)`` — a fast, zero-dependency gate
that catches obviously invalid molecules (hypervalent species, unmatched
syntax) *before* they reach RDKit's C++ graph constructors.

Every function in this module uses only the Python standard library.
"""

from __future__ import annotations

from typing import Any

# Maximum permissible explicit valence for neutral organic atoms.
# Lowercase entries cover aromatic SMILES notation (c, n, o, s, p).
_MAX_VALENCE: dict[str, int] = {
    "B": 3,  "b": 3,
    "C": 4,  "c": 4,
    "N": 3,  "n": 3,
    "O": 2,  "o": 2,
    "F": 1,
    "Si": 4, "si": 4,
    "P": 6,  "p": 6,
    "S": 6,  "s": 6,
    "Cl": 1, "cl": 1,
    "Br": 1, "br": 1,
    "I": 1,  "i": 1,
}

# Elements whose valence limit rises by 1 under positive formal charge.
_CHARGE_BOOST: set[str] = {"N", "n", "P", "p", "S", "s"}

_RING_DIGITS: set[str] = set("0123456789")


def is_structurally_viable(smiles: str) -> bool:
    """Pure-Python structural pre-check for a SMILES string.

    Returns ``False`` when the molecule is *definitely* invalid (unbalanced
    syntax, unmatched ring digits, impossible explicit valence).  Returns
    ``True`` when it passes these basic checks (it may still fail RDKit's
    full ``SanitizeMol``).

    The function uses **zero** third-party dependencies — only the Python
    standard library.  This makes it safe to call in performance-critical
    mutation hot loops before handing the SMILES to RDKit.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    s = smiles.strip()

    # ------------------------------------------------------------------
    # 1.  Basic syntax pre-checks
    # ------------------------------------------------------------------
    if s.count("(") != s.count(")"):
        return False
    if s.count("[") != s.count("]"):
        return False

    # Ring-digit matching: every digit must appear exactly twice.
    seen: set[str] = set()
    for ch in s:
        if ch in _RING_DIGITS:
            if ch == "0":
                continue
            if ch in seen:
                seen.discard(ch)
            else:
                seen.add(ch)
    if seen:
        return False

    # ------------------------------------------------------------------
    # 2.  Lightweight valence pre-check
    # ------------------------------------------------------------------
    return _check_smiles_valence(s)


# ---------------------------------------------------------------------------
# SMILES valence checker
# ---------------------------------------------------------------------------

def _check_smiles_valence(smiles: str) -> bool:
    """Walk the SMILES string and verify no atom exceeds its valence limit.

    Uses a forward-scanning approach that counts explicit connections
    (branches, ring closures, and adjacent atoms) for each main-group
    element token.  Catches the most common hypervalent patterns that
    would cause RDKit ``SanitizeMol`` to emit C++ stderr warnings.
    """
    atoms: list[dict[str, Any]] = []
    _build_atom_list(smiles, atoms)

    for atom in atoms:
        elem: str = atom["elem"]
        if elem not in _MAX_VALENCE:
            continue

        allowed: int = _MAX_VALENCE[elem]
        charge: int = atom["charge"]
        if charge > 0 and elem in _CHARGE_BOOST:
            allowed += 1

        if atom["connections"] > allowed:
            return False

    return True


def _build_atom_list(smiles: str, atoms: list[dict[str, Any]]) -> None:
    """Tokenise the SMILES string and populate *atoms* with connection counts.

    Uses a scoped context stack so that atoms inside a branch chain to the
    previous atom *within that branch* rather than all connecting back to
    the branch parent.

    Each atom dict contains: ``elem``, ``charge``, ``pos``, ``connections``.
    """
    # ---- Phase 1: collect raw tokens -----------------------------------
    tokens: list[tuple[str, Any]] = []
    i = 0
    n = len(smiles)

    while i < n:
        ch = smiles[i]
        if ch in "()":
            tokens.append(("paren", ch))
            i += 1
        elif ch == "[":
            close = smiles.find("]", i)
            if close == -1:
                return
            inner = smiles[i + 1 : close]
            elem = ""
            for c in inner:
                if c.isalpha():
                    elem += c
                else:
                    break
            charge = inner.count("+") - inner.count("-")
            tokens.append(("atom", (elem, charge)))
            i = close + 1
        elif ch.isalpha():
            j = i + 1
            if j < n and smiles[j].islower():
                j += 1
            elem = smiles[i:j]
            tokens.append(("atom", (elem, 0)))
            i = j
        elif ch in "=-#:$":
            tokens.append(("bond", ch))
            i += 1
        elif ch.isdigit() and ch != "0":
            tokens.append(("ring", ch))
            i += 1
        elif ch == "%":
            tokens.append(("ring", smiles[i : i + 3]))
            i += 3
        elif ch in "./\\":
            tokens.append(("sep", ch))
            i += 1
        else:
            i += 1

    # ---- Phase 2: walk tokens with a scoped context stack --------------
    # Each context: {"parent": int, "last": int, "pending_bond": str|None}
    # Parent is the atom this scope hangs from; last is the most recent atom
    # at this scope level (for intra-scope chaining).
    ctx: list[dict[str, Any]] = [
        {"parent": -1, "last": -1, "pending_bond": None}
    ]

    for tok_type, tok_val in tokens:
        if tok_type == "paren":
            if tok_val == "(":
                # Parent = the last atom in the CURRENT scope (before push).
                parent = ctx[-1]["last"]
                # Carry forward any pending bond (e.g. '=' in C(=O)).
                bond = ctx[-1]["pending_bond"]
                ctx[-1]["pending_bond"] = None
                ctx.append({"parent": parent, "last": -1, "pending_bond": bond})
            else:  # ')'
                if len(ctx) > 1:
                    ctx.pop()
                ctx[-1]["pending_bond"] = None

        elif tok_type == "atom":
            elem, charge = tok_val
            idx = len(atoms)
            atoms.append({"elem": elem, "charge": charge, "pos": idx, "connections": 0})

            scope = ctx[-1]
            bt = _bond_order(scope["pending_bond"])

            if scope["last"] >= 0:
                # Bond between previous atom at this scope and the new one
                atoms[scope["last"]]["connections"] += bt
                atoms[idx]["connections"] += bt
            elif scope["parent"] >= 0:
                # First atom in this scope – bond to the parent
                atoms[scope["parent"]]["connections"] += bt
                atoms[idx]["connections"] += bt

            scope["last"] = idx
            scope["pending_bond"] = None

        elif tok_type == "bond":
            ctx[-1]["pending_bond"] = tok_val

        elif tok_type == "ring":
            if atoms:
                atoms[-1]["connections"] += 1
            ctx[-1]["pending_bond"] = None

        elif tok_type == "sep":
            if tok_val == ".":
                # Disconnection – reset top-level scope (fragment boundary)
                ctx[0]["last"] = -1
            ctx[-1]["pending_bond"] = None


def _bond_order(bond_sym: str | None) -> int:
    """Map a SMILES bond symbol to its bond-order contribution to valence."""
    if bond_sym == "=":
        return 2
    if bond_sym == "#":
        return 3
    if bond_sym == "$":
        return 2  # quadruple / aromatic – treat conservatively
    # '-', ':', '/', '\', None → single bond
    return 1
