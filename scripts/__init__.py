"""Scripts for data preparation and model training.

Backward-compatible re-exports from cli_scripts.
"""

from aurelius.cli_scripts import (
    train_main,
    train_tier0_main,
    download_data_main,
    prep_discovery,
    train_tier0_main as train_tier0_main,
)

__all__ = [
    "train_main",
    "train_tier0_main",
    "download_data_main",
    "prep_discovery",
]
