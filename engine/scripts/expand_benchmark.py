#!/usr/bin/env python3
"""Expand the external property benchmark with missing common electrolyte solvents.

Reads ``external_property_benchmark.json``, identifies common battery electrolyte
solvents that are absent, and appends placeholder entries with
``homo_source: "literature"`` and a published reference.

Only entries whose experimental values are publicly known and cited are added.
No fabricated data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "src" / "aurelius" / "data" / "external_property_benchmark.json"

# Known common electrolyte solvents missing from the current benchmark.
# Each entry: (smiles, name, reference, dielectric, viscosity, donor_number)
# Values are from peer-reviewed literature as cited.
_MISSING_SOLVENTS: list[dict] = [
    {
        "smiles": "CC1CCCO1",
        "name": "2-Methyltetrahydrofuran (2-MeTHF)",
        "reference": "Xu 2004, Chem. Rev. 104, 4303; CRC Handbook 97th Ed.",
        "dielectric_constant": 6.97,
        "viscosity_cP": 0.47,
        "donor_number": 18.0,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
    {
        "smiles": "CCOC(=O)OCCC",
        "name": "Ethyl propyl carbonate (EPC)",
        "reference": "Xu 2004, Chem. Rev. 104, 4303",
        "dielectric_constant": 2.8,
        "viscosity_cP": 0.69,
        "donor_number": None,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
    {
        "smiles": "N#CCCCC#N",
        "name": "Glutaronitrile",
        "reference": "CRC Handbook; J. Electrochem. Soc. 2015, 162, A7037",
        "dielectric_constant": 37.0,
        "viscosity_cP": 4.9,
        "donor_number": 10.0,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
    {
        "smiles": "C1COCCO1",
        "name": "1,4-Dioxane",
        "reference": "CRC Handbook 97th Ed.; Izutsu 2002",
        "dielectric_constant": 2.21,
        "viscosity_cP": 1.18,
        "donor_number": 14.8,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
    {
        "smiles": "CC(=O)OC(C)C",
        "name": "Isopropyl acetate",
        "reference": "CRC Handbook 97th Ed.",
        "dielectric_constant": 5.6,
        "viscosity_cP": 0.54,
        "donor_number": 16.8,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
    {
        "smiles": "CC(=O)OCCOC(C)=O",
        "name": "Ethylene glycol diacetate (EGDA)",
        "reference": "CRC Handbook; J. Power Sources 2018, 402, 310",
        "dielectric_constant": 7.9,
        "viscosity_cP": 2.8,
        "donor_number": None,
        "homo_eV": None,
        "lumo_eV": None,
        "homo_source": "literature",
        "lumo_source": "literature",
    },
]


def load_benchmark(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def existing_smiles(data: list[dict]) -> set[str]:
    return {e["smiles"] for e in data}


def main() -> None:
    if not BENCHMARK_PATH.exists():
        log.error("Benchmark file not found: %s", BENCHMARK_PATH)
        return

    data = load_benchmark(BENCHMARK_PATH)
    known = existing_smiles(data)

    added = 0
    for entry in _MISSING_SOLVENTS:
        if entry["smiles"] in known:
            log.info("Already present: %s (%s)", entry["name"], entry["smiles"])
            continue
        data.append(entry)
        added += 1
        log.info("Added: %s (%s)", entry["name"], entry["smiles"])

    if added == 0:
        log.info("No new entries to add. Benchmark already contains all checked solvents.")
        return

    data.sort(key=lambda e: e.get("name", e["smiles"]))

    with open(BENCHMARK_PATH, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Added %d entries to %s", added, BENCHMARK_PATH)


if __name__ == "__main__":
    main()
