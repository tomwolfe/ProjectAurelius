#!/usr/bin/env python3
"""Validate new entries for external_property_benchmark.json.

Usage:

    python scripts/validate_benchmark_entry.py path/to/new_entries.json

Checks that each entry has the required fields, valid SMILES, and
plausible value ranges. Exits with code 0 if all entries pass,
non-zero if any validation issues are found.

Add new entries to external_property_benchmark.json via:

    python scripts/merge_benchmark_entry.py new_entries.json
"""

import json
import sys
from pathlib import Path

from rdkit import Chem

REQUIRED_FIELDS = {"smiles", "name"}
OPTIONAL_FIELDS = {
    "reference",
    "dielectric_constant",
    "viscosity_cP",
    "donor_number",
    "homo_eV",
    "lumo_eV",
    "homo_source",
    "lumo_source",
}

PLAUSIBLE_RANGES = {
    "dielectric_constant": (1.0, 200.0),
    "viscosity_cP": (0.1, 100.0),
    "donor_number": (0.0, 60.0),
    "homo_eV": (-15.0, -4.0),
    "lumo_eV": (-5.0, 5.0),
}


def validate_entry(entry: dict, index: int) -> list[str]:
    """Validate a single benchmark entry, returning a list of issues."""
    issues: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in entry or not entry[field]:
            issues.append(f"Entry {index}: missing required field '{field}'")

    extra = set(entry.keys()) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if extra:
        issues.append(f"Entry {index}: unknown field(s): {', '.join(sorted(extra))}")

    smiles = entry.get("smiles", "")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            issues.append(f"Entry {index}: invalid SMILES: {smiles}")
        else:
            canon = Chem.MolToSmiles(mol)
            if canon != smiles:
                issues.append(f"Entry {index}: SMILES not canonicalised: {smiles} -> {canon}")

    for field, (lo, hi) in PLAUSIBLE_RANGES.items():
        val = entry.get(field)
        if val is not None:
            if not isinstance(val, (int, float)):
                issues.append(f"Entry {index}: {field} must be numeric, got {type(val).__name__}")
            elif val < lo or val > hi:
                issues.append(f"Entry {index}: {field}={val} outside plausible range [{lo}, {hi}]")

    return issues


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_benchmark_entry.py <file.json> [file2.json ...]")
        sys.exit(1)

    all_issues: list[str] = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            all_issues.append(f"File not found: {path}")
            continue

        with open(path) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError as e:
                all_issues.append(f"{path}: invalid JSON — {e}")
                continue

        if not isinstance(entries, list):
            all_issues.append(f"{path}: expected a list of entries, got {type(entries).__name__}")
            continue

        for i, entry in enumerate(entries):
            all_issues.extend(validate_entry(entry, i))

    if all_issues:
        print("VALIDATION FAILED:")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: all entries are valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
