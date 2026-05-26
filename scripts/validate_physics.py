#!/usr/bin/env python3
"""Physics validation for a single molecule.

This is a lightweight wrapper that imports and calls the module-level
function from ``src/aurelius/cli_scripts/``.
"""

from __future__ import annotations

from aurelius.cli_scripts.validate_physics import main

if __name__ == "__main__":
    main()
