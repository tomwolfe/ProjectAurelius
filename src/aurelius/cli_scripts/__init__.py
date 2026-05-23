"""CLI scripts for data preparation and model training.

This package exposes training and data preparation entry points
that are bundled with the aurelius package for pip installation.
"""

from aurelius.cli_scripts.download_data import main as download_data_main
from aurelius.cli_scripts.prep_discovery import prep_discovery
from aurelius.cli_scripts.train_tier0 import train_main as train_tier0_main
from aurelius.cli_scripts.train_tier1 import train_main
from aurelius.cli_scripts.validate_physics import main as validate_physics_main

__all__ = [
    "download_data_main",
    "prep_discovery",
    "train_main",
    "train_tier0_main",
    "validate_physics_main",
]
