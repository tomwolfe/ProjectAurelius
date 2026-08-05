"""OOD validation data loader for Project Aurelius.

Loads and provides access to out-of-distribution validation molecules
for testing oracle robustness.
"""

import json
from importlib.resources import files
from pathlib import Path

_OOD_VALIDATION_PATH = Path(files("aurelius.data")) / "ood_validation.json"


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
    with open(_OOD_VALIDATION_PATH, "r") as f:
        return json.load(f)
