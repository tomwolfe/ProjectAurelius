"""CLI scripts for data preparation and validation.

This package exposes entry points that are bundled with the aurelius
package for pip installation.
"""

from aurelius.cli_scripts.validate_physics import main as validate_physics_main

__all__ = [
    "validate_physics_main",
]
