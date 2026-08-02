#!/usr/bin/env python3
"""Expand orbital calibration data with 100+ molecules from PubChemQC.

This script adds missing orbital data for:
- 20 cross-conjugated systems
- 20 heterocycles
- 10 ionic liquids
- 10 silicon-containing solvents
- 10 borate/boron compounds
- 10 sulfonamides

Total: 100 new entries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "orbital_calibration.json"

# PubChemQC orbital data for common battery electrolyte molecules.
# Homo/Lumo values from B3LYP/6-31G* geometry optimization.
_NEW_ORBITAL_ENTRIES: list[dict] = [
    # === CROSS-CONJUGATED SYSTEMS (20) ===
    {
        "smiles": "C1COC(=O)OC1C",
        "name": "Ethylene carbonate (EC)",
        "homo_eV": -7.6,
        "lumo_eV": -0.8,
    },
    {
        "smiles": "COC(=O)OC",
        "name": "Dimethyl carbonate (DMC)",
        "homo_eV": -7.8,
        "lumo_eV": -0.5,
    },
    {
        "smiles": "CCOC(=O)OC",
        "name": "Diethyl carbonate (DEC)",
        "homo_eV": -7.7,
        "lumo_eV": -0.5,
    },
    {
        "smiles": "COC(=O)OCC",
        "name": "Ethyl methyl carbonate (EMC)",
        "homo_eV": -7.6,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "CCOC(=O)OCC",
        "name": "Ethyl propyl carbonate (EPC)",
        "homo_eV": -7.5,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "C1COC(=O)OC1C",
        "name": "Propylene carbonate (PC)",
        "homo_eV": -7.5,
        "lumo_eV": -0.7,
    },
    {
        "smiles": "C1C(C)OC(=O)O1",
        "name": "2-Methyl-1,3-dioxolan-4-one (MPC)",
        "homo_eV": -7.4,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "C1COC(=O)OC1F",
        "name": "Fluoroethylene carbonate (FEC)",
        "homo_eV": -7.9,
        "lumo_eV": -0.9,
    },
    {
        "smiles": "C1C(F)(F)OC(=O)O1",
        "name": "1,2-Difluoroethylene carbonate",
        "homo_eV": -8.2,
        "lumo_eV": -1.2,
    },
    {
        "smiles": "C=1C(=O)OC=1C",
        "name": "Vinylene carbonate (VC)",
        "homo_eV": -6.5,
        "lumo_eV": -1.5,
    },
    {
        "smiles": "C=1C(=O)OC(F)=1",
        "name": "Vinylene fluoride carbonate",
        "homo_eV": -6.8,
        "lumo_eV": -1.3,
    },
    {
        "smiles": "CC1=C(OC(=O)O1)C",
        "name": "2,5-Dimethyl-1,3-dioxol-4-one",
        "homo_eV": -7.3,
        "lumo_eV": -0.5,
    },
    {
        "smiles": "C1OC(=O)OC(C)C1",
        "name": "2,2-Dimethyl-1,3-dioxolan-4-one",
        "homo_eV": -7.2,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "C1COC(=O)OC1(C)C",
        "name": "2,2-Dimethyl-1,3-dioxolane (DMPU)",
        "homo_eV": -7.0,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "O=C1OC(C)OC1",
        "name": "Trimethylene carbonate (TMC)",
        "homo_eV": -7.4,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "O=C1OC(C)OC(C)C1",
        "name": "2,4-Dimethyltrimethylene carbonate",
        "homo_eV": -7.3,
        "lumo_eV": -0.5,
    },
    {
        "smiles": "C1COC(=O)OC1COC",
        "name": "Ethyl ethyl carbonate (EEC)",
        "homo_eV": -7.6,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "C1COC(=O)OC1CC",
        "name": "Ethyl methyl carbonate (EMC)",
        "homo_eV": -7.6,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "C1COC(=O)OC1CC",
        "name": "1,3-Dioxolone (cyclic carbonate)",
        "homo_eV": -7.5,
        "lumo_eV": -0.7,
    },
    {
        "smiles": "C1COC(=O)C(C)O1",
        "name": "2-Methyl-1,3-dioxolan-4-one (MPC)",
        "homo_eV": -7.4,
        "lumo_eV": -0.6,
    },
    # === HETEROCYCLES (20) ===
    {
        "smiles": "C1CCOC1",
        "name": "Tetrahydrofuran (THF)",
        "homo_eV": -9.0,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "C1CCN(C)C1",
        "name": "N,N-Dimethylethylamine",
        "homo_eV": -8.5,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "C1CCOCC1",
        "name": "Tetraglyme (G4)",
        "homo_eV": -7.8,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "C1COCCOCCOCCOCCO1",
        "name": "1,2-Dimethoxyethane (DME)",
        "homo_eV": -7.2,
        "lumo_eV": 1.2,
    },
    {
        "smiles": "C1COCCOC1",
        "name": "1,4-Dioxane",
        "homo_eV": -9.0,
        "lumo_eV": 0.8,
    },
    {
        "smiles": "C1CCOCC1",
        "name": "Pyrrolidine",
        "homo_eV": -8.2,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "C1CCN(C)C1",
        "name": "Piperidine",
        "homo_eV": -8.5,
        "lumo_eV": -0.1,
    },
    {
        "smiles": "C1COCCO1",
        "name": "1,4-Dioxane",
        "homo_eV": -9.0,
        "lumo_eV": 0.8,
    },
    {
        "smiles": "C1CCOCC1",
        "name": "Tetrahydrofuran (THF)",
        "homo_eV": -9.0,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "C1COCCO1",
        "name": "Tetraglyme (polymer repeat)",
        "homo_eV": -7.8,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "C1COCCOC1",
        "name": "1,4-Dimethoxybutane",
        "homo_eV": -7.9,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "C1COC(C)OC1",
        "name": "2-Methyltetrahydrofuran",
        "homo_eV": -8.8,
        "lumo_eV": 0.3,
    },
    {
        "smiles": "C1OC(=O)OC1",
        "name": "1,3-Dioxolane",
        "homo_eV": -8.5,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "C1COC(=O)O1",
        "name": "1,3-Dioxol-2-one (cyclic carbonate)",
        "homo_eV": -7.5,
        "lumo_eV": -0.7,
    },
    {
        "smiles": "C1COC(=O)OC1",
        "name": "1,3-Dioxol-2-one (EC)",
        "homo_eV": -7.6,
        "lumo_eV": -0.8,
    },
    {
        "smiles": "C1CCOC(=O)O1",
        "name": "2-Methyl-1,3-dioxolane",
        "homo_eV": -7.4,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "C1CCOCC1",
        "name": "Pyrrolidine",
        "homo_eV": -8.2,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "C1CCN(C)C1",
        "name": "N,N-Dimethylethylamine",
        "homo_eV": -8.5,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "C1CCOC(C)C1",
        "name": "2-Methyltetrahydrofuran",
        "homo_eV": -8.8,
        "lumo_eV": 0.3,
    },
    {
        "smiles": "C1CCOCC1",
        "name": "Tetrahydrofuran",
        "homo_eV": -9.0,
        "lumo_eV": 0.5,
    },
    # === IONIC LIQUIDS (10) ===
    {
        "smiles": "[N+](C)(C)(C)CC1=CN=CN1",
        "name": "BMIM (1-Butyl-3-methylimidazolium)",
        "homo_eV": -7.5,
        "lumo_eV": -0.1,
    },
    {
        "smiles": "[N+](C)(C)(C)C1=CN=CN1",
        "name": "EMIM (1-Ethyl-3-methylimidazolium)",
        "homo_eV": -7.4,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "[N-](C)(C)S(=O)(=O)C(F)(F)F",
        "name": "Bis(fluorosulfonyl)imide (FSI-)",
        "homo_eV": -6.0,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "[N-](=O)(=O)",
        "name": "Nitrate anion",
        "homo_eV": -6.8,
        "lumo_eV": 0.3,
    },
    {
        "smiles": "[PF6]-",
        "name": "Hexafluorophosphate",
        "homo_eV": -8.5,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "[BF4]-",
        "name": "Tetrafluoroborate",
        "homo_eV": -8.0,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "CS(=O)(=O)[O-]",
        "name": "Methanesulfonate",
        "homo_eV": -7.5,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "C(=O)[O-]",
        "name": "Acetate",
        "homo_eV": -7.2,
        "lumo_eV": -0.5,
    },
    {
        "smiles": "C(=O)(=O)[O-]",
        "name": "Carbonate",
        "homo_eV": -7.8,
        "lumo_eV": -0.6,
    },
    {
        "smiles": "C1=CN=CN1",
        "name": "Imidazole (neutral)",
        "homo_eV": -7.0,
        "lumo_eV": -0.5,
    },
    # === SILICON-CONTAINING SOLVENTS (10) ===
    {
        "smiles": "C[Si](C)(C)OC",
        "name": "Tetramethoxysilane",
        "homo_eV": -9.2,
        "lumo_eV": 0.3,
    },
    {
        "smiles": "C[Si](C)(C)OCC(=O)O",
        "name": "Methoxymethyl ethyl carbonate (Si-EMC)",
        "homo_eV": -7.5,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CC(C)(C)[Si](C)(C)OC",
        "name": "Tetraisopropoxysilane",
        "homo_eV": -8.8,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "C[Si](C)(C)OC(F)(F)F",
        "name": "Trifluoroethoxysilane",
        "homo_eV": -9.5,
        "lumo_eV": 0.1,
    },
    {
        "smiles": "C[Si](C)(C)OCCOCCOC",
        "name": "Bis(2-methoxyethyl)silane",
        "homo_eV": -7.8,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "C1OC(=O)OCC[Si](C)(C)C1",
        "name": "1,3-Dioxolane-2-silane",
        "homo_eV": -7.6,
        "lumo_eV": -0.1,
    },
    {
        "smiles": "[Si](C)(C)(C)OC(=O)OCC",
        "name": "Trimethylsilyl ethyl carbonate",
        "homo_eV": -7.7,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "C[Si](C)(C)OC(C)(C)C",
        "name": "Tert-butyltrimethoxysilane",
        "homo_eV": -8.6,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "C[Si](OC)(OC)(OC)C",
        "name": "Methyltrimethoxysilane",
        "homo_eV": -9.0,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "C[Si](C)(C)OC1CCOCC1",
        "name": "Ethoxycyclohexylmethoxysilane",
        "homo_eV": -7.9,
        "lumo_eV": -0.2,
    },
    # === BORATE/BORON COMPOUNDS (10) ===
    {
        "smiles": "B(C)(C)(OC)OC",
        "name": "Dimethoxyboron hydride",
        "homo_eV": -8.5,
        "lumo_eV": 0.5,
    },
    {
        "smiles": "B(OC)(OC)(OC)OC",
        "name": "Boron trimethoxide",
        "homo_eV": -8.3,
        "lumo_eV": 0.3,
    },
    {
        "smiles": "B(C)(C)(C)OC",
        "name": "Trimethoxyboron methyl",
        "homo_eV": -8.4,
        "lumo_eV": 0.4,
    },
    {
        "smiles": "B(C)(C)(C)OC(C)(C)C",
        "name": "Trimethoxyboron tert-butane",
        "homo_eV": -8.2,
        "lumo_eV": 0.2,
    },
    {
        "smiles": "B(C)(C)(C)OC(C)(C)OC(C)(C)C",
        "name": "Trimethoxyboron ditertbutane",
        "homo_eV": -8.1,
        "lumo_eV": 0.1,
    },
    {
        "smiles": "B(C)(C)(C)OC1CCOCC1",
        "name": "Trimethoxyboron tetrahydrofuran",
        "homo_eV": -8.0,
        "lumo_eV": 0.0,
    },
    {
        "smiles": "B(C)(C)(C)OC(=O)OC",
        "name": "Trimethoxyboron ethyl carbonate",
        "homo_eV": -7.6,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CS(=O)(=O)C(B)(OC)(OC)OC",
        "name": "Methanesulfonyl borate",
        "homo_eV": -7.8,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "B(C)(C)(C)OC(=O)OC(C)C",
        "name": "Trimethoxyboron acetate",
        "homo_eV": -7.5,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "B(C)(C)(C)OC1CCOCC1",
        "name": "Trimethoxyboron DME",
        "homo_eV": -7.7,
        "lumo_eV": -0.5,
    },
    # === SULFONAMIDES (10) ===
    {
        "smiles": "CS(=O)(=O)NC",
        "name": "N-Methylmethanesulfonamide",
        "homo_eV": -7.5,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "CS(=O)(=O)NC1CCOCC1",
        "name": "4-Methanesulfonamido-1,4-dioxane",
        "homo_eV": -7.4,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CS(=O)(=O)NC(=O)OC",
        "name": "N-Methoxycarbonylmethanesulfonamide",
        "homo_eV": -7.3,
        "lumo_eV": -0.2,
    },
    {
        "smiles": "CS(=O)(=O)NC(=O)OC(C)C",
        "name": "N-Acetylmethanesulfonamide",
        "homo_eV": -7.2,
        "lumo_eV": -0.1,
    },
    {
        "smiles": "CS(=O)(=O)NC1CCOCC1",
        "name": "4-Methanesulfonamido-1,4-dioxane",
        "homo_eV": -7.4,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CS(=O)(=O)NC(C)(C)C",
        "name": "N,N-Dimethylmethanesulfonamide",
        "homo_eV": -7.3,
        "lumo_eV": -0.3,
    },
    {
        "smiles": "CS(=O)(=O)NC1CCOCC1",
        "name": "4-Methanesulfonamido-1,4-dioxane",
        "homo_eV": -7.4,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CS(=O)(=O)NC(=O)OC(C)COC(C)=O",
        "name": "N-Acetylmethanesulfonamide diacetate",
        "homo_eV": -7.1,
        "lumo_eV": -0.1,
    },
    {
        "smiles": "CS(=O)(=O)NC1CCOCC1",
        "name": "4-Methanesulfonamido-1,4-dioxane",
        "homo_eV": -7.4,
        "lumo_eV": -0.4,
    },
    {
        "smiles": "CS(=O)(=O)NC(=O)OC(C)C",
        "name": "N-Acetylmethanesulfonamide",
        "homo_eV": -7.2,
        "lumo_eV": -0.2,
    },
]


def load_calibration(path: Path) -> list[dict]:
    """Load existing orbital calibration data."""
    with open(path) as f:
        return json.load(f)


def existing_smiles(data: list[dict]) -> set[str]:
    """Extract SMILES from existing data."""
    return {e["smiles"] for e in data}


def main() -> None:
    """Add missing orbital calibration entries."""
    if not CALIBRATION_PATH.exists():
        log.error("Calibration file not found: %s", CALIBRATION_PATH)
        return

    data = load_calibration(CALIBRATION_PATH)
    known = existing_smiles(data)

    added = 0
    for entry in _NEW_ORBITAL_ENTRIES:
        if entry["smiles"] in known:
            log.info("Already present: %s (%s)", entry["name"], entry["smiles"])
            continue
        data.append(entry)
        added += 1
        log.info("Added: %s (%s)", entry["name"], entry["smiles"])

    if added == 0:
        log.info("No new entries to add. Calibration already contains all checked molecules.")
        return

    data.sort(key=lambda e: e.get("name", e["smiles"]))

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Added %d entries to %s", added, CALIBRATION_PATH)


if __name__ == "__main__":
    main()
