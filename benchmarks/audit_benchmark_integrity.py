"""Structural-integrity audit for the external property benchmark.

ADR-2026-08-07-05: Added after the v11.0 review found that roughly half of
``external_property_benchmark.json`` was not literature-traceable, including
11 SMILES that do not parse and nitrile dielectric constants off by an order
of magnitude. Those entries dominated the reported oracle MAE, so accuracy
claims made against the unaudited file were not meaningful.

This module checks *internal consistency* — does the structure match the
name, does the SMILES parse, does a value contradict a verified source — and
never checks agreement with the oracle. A benchmark entry must be able to
fail while the model is right; that is the whole point of a reference set.

Run directly to re-audit::

    python -m benchmarks.audit_benchmark_integrity
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_PATH = _REPO_ROOT / "src" / "aurelius" / "data" / "external_property_benchmark.json"
_VERIFIED_PATH = _REPO_ROOT / "benchmarks" / "data" / "dielectric_verified.json"
_QUARANTINE_PATH = _REPO_ROOT / "benchmarks" / "data" / "quarantined_benchmark_entries.json"

_DIELECTRIC_TOLERANCE = 2.0
"""Permitted disagreement with a verified source before quarantine.

Generous: legitimate inter-source spread arises from measurement temperature
(EC is quoted at 40 C because it melts at 36 C) and frequency. Only gross
contradictions are flagged."""

# (name substring, SMARTS that must be present)
_NAME_STRUCTURE_RULES: list[tuple[str, str, str]] = [
    ("nitrile", "[NX1]#[CX2]", "no C#N group"),
    ("carbonate", "O=C([OX2])[OX2]", "no O-C(=O)-O group"),
    ("sulfolane", "O=S1(=O)CCCC1", "not a 5-membered SO2 ring"),
    ("sulfone", "S(=O)(=O)", "no SO2 group"),
    ("phosph", "[PX4]", "no phosphorus atom"),
]

_FLUORINE_NAME_PATTERN = re.compile(r"fluoro|fluorinated|perfluoro")


def audit_entry(entry: dict[str, Any], verified: dict[str, dict[str, Any]]) -> list[str]:
    """Return a list of integrity failures for one benchmark entry.

    An empty list means the entry passed.
    """
    reasons: list[str] = []
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        return ["SMILES does not parse"]

    name = entry["name"].lower()

    canonical = Chem.MolToSmiles(mol)
    reference = verified.get(canonical)
    epsilon = entry.get("dielectric_constant")
    if reference is not None and epsilon is not None:
        delta = abs(reference["dielectric_constant"] - epsilon)
        if delta > _DIELECTRIC_TOLERANCE:
            reasons.append(
                f"dielectric {epsilon} contradicts verified "
                f"{reference['dielectric_constant']} ({reference['name']})"
            )

    for keyword, smarts, message in _NAME_STRUCTURE_RULES:
        if keyword in name and not mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            reasons.append(f"name says '{keyword}' but structure has {message}")

    if _FLUORINE_NAME_PATTERN.search(name) and not any(
        atom.GetAtomicNum() == 9 for atom in mol.GetAtoms()
    ):
        reasons.append("name says fluorinated but structure has no F")

    return reasons


def load_verified() -> dict[str, dict[str, Any]]:
    """Load the verified reference set, keyed by canonical SMILES."""
    entries = json.loads(_VERIFIED_PATH.read_text())["entries"]
    return {
        Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])): e for e in entries
    }


def audit(write: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit the benchmark file.

    Args:
        write: If True, rewrite the benchmark with only passing entries and
            record the failures in the quarantine file.

    Returns:
        (kept, quarantined)
    """
    data = json.loads(_BENCHMARK_PATH.read_text())
    verified = load_verified()

    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for entry in data:
        reasons = audit_entry(entry, verified)
        if reasons:
            quarantined.append({**entry, "quarantine_reasons": reasons})
        else:
            kept.append(entry)

    if write:
        _BENCHMARK_PATH.write_text(json.dumps(kept, indent=2))
        _QUARANTINE_PATH.write_text(json.dumps(quarantined, indent=2))

    return kept, quarantined


def main() -> None:
    kept, quarantined = audit(write=False)
    total = len(kept) + len(quarantined)
    print(f"Benchmark integrity audit: {total} entries, "
          f"{len(kept)} pass, {len(quarantined)} fail")
    for entry in quarantined:
        print(f"  {entry['name'][:40]:40s} {entry['quarantine_reasons'][0]}")
    if not quarantined:
        print("All entries pass structural integrity checks.")


if __name__ == "__main__":
    main()
