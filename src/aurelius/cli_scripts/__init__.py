"""CLI scripts for data preparation and evaluation.

This package exposes entry points that are bundled with the aurelius
package for pip installation.
"""

from aurelius.cli_scripts.evaluate import main as evaluate_main

__all__ = [
    "evaluate_main",
]
