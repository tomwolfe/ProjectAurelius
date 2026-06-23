#!/usr/bin/env python3
"""Validate and merge new entries into external_property_benchmark.json.

Usage:

    python scripts/merge_benchmark_entry.py path/to/new_entries.json

Runs validation via validate_benchmark_entry.py first, then merges
new entries (by SMILES dedup) into the benchmark JSON file.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "src" / "aurelius" / "data" / "external_property_benchmark.json"
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_benchmark_entry.py"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_benchmark_entry.py <file.json>")
        sys.exit(1)

    new_path = Path(sys.argv[1])
    if not new_path.exists():
        print(f"Error: file not found: {new_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(new_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Validation failed — merge aborted:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    with open(new_path) as f:
        new_entries = json.load(f)

    if not BENCHMARK_PATH.exists():
        print(f"Error: benchmark file not found: {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH) as f:
        existing = json.load(f)

    existing_smiles = {e["smiles"] for e in existing}
    added = 0
    skipped = 0
    for entry in new_entries:
        if entry["smiles"] in existing_smiles:
            skipped += 1
            print(f"  SKIP (duplicate): {entry['smiles']} ({entry.get('name', 'unnamed')})")
        else:
            existing.append(entry)
            existing_smiles.add(entry["smiles"])
            added += 1
            print(f"  ADDED: {entry['smiles']} ({entry.get('name', 'unnamed')})")

    with open(BENCHMARK_PATH, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")

    print(f"\nMerge complete: {added} added, {skipped} skipped")
    print(f"Total entries: {len(existing)}")


if __name__ == "__main__":
    main()
