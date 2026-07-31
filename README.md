# Project Aurelius — Physics-Grounded Molecular Discovery Engine

**Autonomous discovery of novel electrolyte molecules for battery applications.**

Project Aurelius is a self-contained evolutionary algorithm pipeline that discovers
novel electrolyte molecules using a **hybrid oracle** combining quantum chemistry
(xTB/GFN2-xTB with Topological Orbital Model fallback) with interpretable
group-contribution fragment-additivity models for bulk properties.

## Features

- **Quantum backend**: GFN2-xTB via subprocess (preferred) with Topological Orbital
  Model (TOM) fallback — frontier orbital energies (HOMO/LUMO) predicted from
  first principles, not fragment-additivity.
- **Bulk properties**: Dielectric, viscosity, Li+ solvation, CED, ionic conductivity
  via transparent fragment-additivity with Michaelis-Menten saturation and non-linear
  cross-terms.
- **Uncertainty quantification**: 5-model Random Forest ensemble provides prediction
  intervals and out-of-distribution detection.
- **Autonomous agent**: BRICS mutation + SMARTS reactions, tournament selection with
  Tanimoto diversity penalty, active learning queue.
- **Pareto optimization**: Multi-objective NSGA-II selection across LUMO, dielectric,
  and viscosity objectives.
- **Retrosynthetic verification**: Synthetic feasibility checking via BRICS
  decomposition against commercial building block precursors.

## Quick Start

**Dependency note:** The engine requires **RDKit** for all chemical graph processing.
Install via:
```bash
conda install -c conda-forge rdkit
# or: pip install rdkit-pypi
```

```bash
pip install -e "./engine[ml]"
aurelius init
aurelius screen "C1COC(=O)O1"
```

For detailed explanations of proxy values (dielectric, viscosity, Li+ solvation)
with curated examples and troubleshooting, see
[`docs/quickstart_chemist.md`](docs/quickstart_chemist.md).

For the full user guide (installation, CLI reference, property system, agent usage),
see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

For the engine architecture overview, see [`engine/README.md`](engine/README.md).

## Quantum Backend: xTB

xTB (GFN2-xTB) is the **mandatory** preferred quantum backend. Install it from
[xtb releases](https://github.com/grimme-lab/xtb/releases) and add to `PATH`.

When xTB is unavailable, the engine falls back to the **Topological Orbital Model
(TOM)** — a conjugation-aware particle-in-a-box model with ~1.07 eV MAE on the
calibration set.

## Repository Structure

```
project-aurelius/
├── LICENSE                   # MIT License
├── README.md                 # This file
├── CHANGELOG.md
├── .gitignore
├── engine/                   # MIT-licensed Discovery Engine
│   ├── src/aurelius/         # Core Python package
│   ├── tests/
│   ├── benchmarks/
│   ├── scripts/
│   ├── docs/
│   ├── pyproject.toml
│   └── README.md
├── docs/                     # User documentation
│   ├── quickstart_chemist.md
│   ├── architecture.md
│   ├── USER_GUIDE.md
│   ├── contributing_fragments.md
│   └── case_studies/
```

## License

Project Aurelius is MIT-licensed. See [`engine/LICENSE`](engine/LICENSE).
