"""OOD validation data loader for Project Aurelius.

Loads and provides access to out-of-distribution validation molecules
for testing oracle robustness.
"""

import json
from pathlib import Path

# DATA_DIR not defined in constants.py, use Path directly
OOD_VALIDATION_PATH = Path("src/aurelius/data/ood_validation.json")


def get_ood_molecules():
    """Load and return the OOD validation molecules.

    Returns:
        list: List of dictionaries containing OOD molecule data with keys:
            - smiles: SMILES string
            - name: Common name
            - class: Chemical class
            - expected_dielectric_rank: "high|medium|low"
            - expected_viscosity_rank: "high|medium|low"
    """
    if not OOD_VALIDATION_PATH.exists():
        raise FileNotFoundError(f"OOD validation file not found: {OOD_VALIDATION_PATH}")

    with open(OOD_VALIDATION_PATH, "r") as f:
        return json.load(f)
