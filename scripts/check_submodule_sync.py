#!/usr/bin/env python3
"""Verify submodule commit hash matches known-good hash from submodule_pins.json.

Usage:
    python scripts/check_submodule_sync.py

Reads the pinned hashes from ``engine/data/submodule_pins.json`` and compares
each submodule's current HEAD against the expected commit. Exits with code 1
if any submodule is out of sync.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    pins_path = root / "engine" / "data" / "submodule_pins.json"
    if not pins_path.exists():
        print(f"Pins file not found: {pins_path}", file=sys.stderr)
        sys.exit(1)

    with open(pins_path) as f:
        pins = json.load(f)

    all_ok = True
    for submodule_name, pin in pins.items():
        submodule_path = root / submodule_name
        if not submodule_path.exists():
            print(f"Submodule path not found: {submodule_path}", file=sys.stderr)
            all_ok = False
            continue

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
        )
        current_hash = result.stdout.strip()
        known_good = pin.get("known_good_hash", "")

        if not known_good:
            print(f"Warning: No pinned hash for '{submodule_name}' in {pins_path}")
            continue

        if current_hash != known_good:
            print(
                f"Submodule '{submodule_name}' is not at pinned commit.\n"
                f"  Current: {current_hash}\n"
                f"  Pinned:  {known_good}"
            )
            all_ok = False
        else:
            print(
                f"Submodule '{submodule_name}' is at pinned commit "
                f"{current_hash[:12]}..."
            )

    if all_ok:
        print("All submodules are at pinned commits.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
