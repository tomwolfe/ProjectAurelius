# Project Aurelius v10.0

**Novel molecule discovery for battery electrolytes.**

A physically-grounded Evolutionary Algorithm pipeline with a **hybrid quantum + fragment-additivity oracle**. Frontier orbitals (HOMO/LUMO) are predicted via quantum chemistry (xTB/GFN2-xTB preferred, Topological Orbital Model fallback) — bulk properties (dielectric, viscosity, Li+ solvation) via interpretable group-contribution fragment-additivity.

## Why Hybrid?

| Property | Method | Rationale |
|----------|--------|-----------|
| HOMO / LUMO | QuantumOracle (xTB or TOM) | Orbitals are delocalised quantum phenomena — NOT additive. GC would be "gamed" by stacking fragments. |
| Dielectric ε | GC fragment-additivity + TPSA | Bulk polarity is reasonably additive. |
| Viscosity | GC fragment-additivity + MW + RotB | Transport properties correlate with group contributions. |
| Li+ Solvation | GC fragment-additivity | Donor-number additivity is physically valid. |

The hybrid oracle (non-linear quantum HOMO/LUMO + additive GC bulk properties) keeps the pipeline physically grounded while maintaining interpretability.

## Architecture

```mermaid
flowchart TD
    A["Seed Pool"] --> B["BRICS Mutation"]
    B --> C["Anti-Gaming Gate"]
    C --> D["Novelty Gate"]
    D --> E["Tier 1 Filter"]
    E --> F{"Hybrid Oracle"}

    subgraph Oracle["Oracle Internals"]
        G["xTB / TOM"] --> H["HOMO / LUMO"]
        I["GC Fragment-Additivity"] --> J["Bulk: eps, eta, Li+"]
        H --> K["DoA Penalty"]
        J --> K
        K --> L["Composite Score"]
    end

    F --> L
    L --> M["Tournament Selection"]
    M --> N{"Converged?"}
    N -->|No| B
    N -->|Yes| O["Top Discoveries"]
```

## Overview

| Component | Framework | Purpose |
|-----------|-----------|---------|
| Filter | RDKit | Electrolyte viability (MW, HBD, RotB, SA score) |
| Oracle | Quantum + GC | HOMO/LUMO from xTB/TOM; bulk from fragment-additivity |
| Mutation | SMARTS + BRICS | Targeted electrolyte edits + scaffold hopping |
 | Selection | Tournament Selection | Tanimoto-guided evolutionary diversity pressure |

The composite Aurelius Score is computed via Gaussian LUMO reward (SEI formation window), sigmoid HOMO penalty (oxidative stability threshold), sigmoid dielectric/viscosity/Li-solvation rewards, and SA score penalty. Tournament selection with a Tanimoto diversity penalty steers each generation away from chemical saturation.

## Installation

Since Aurelius relies on RDKit's C++ bindings, we recommend a conda-first installation.

```bash
# 1. Create and activate a conda environment with Python 3.11 and RDKit
conda create -n aurelius python=3.11 rdkit -c conda-forge
conda activate aurelius

# 2. Install Aurelius directly from GitHub
pip install git+https://github.com/tomwolfe/ProjectAurelius.git
```

## Quick Start

```bash
aurelius init                         # Initialize pipeline
aurelius doctor                       # Validate dependencies
aurelius doctor-xtb                   # Check xTB quantum backend
aurelius screen "CC(=O)OC1=CC=CC=C1" # Screen a molecule
aurelius batch examples/molecules.smi --output results.json  # Batch screen
aurelius agent --max-generations 50   # Run autonomous discovery loop
```

## CLI Reference

```
aurelius init                    Initialize pipeline
aurelius doctor                  Validate dependencies and hardware
aurelius doctor-xtb              Check xTB quantum backend availability
aurelius screen <smiles>         Screen a single molecule
aurelius batch <file>            Screen molecules from SMILES file
aurelius score <smiles>          Compute Aurelius score only
aurelius validate <smiles>       Run physics validation
aurelius agent                   Run the autonomous screening agent
```

## Quantum Backend

### Preferred: xTB (GFN2-xTB)
Install the xTB binary from https://xtb-docs.readthedocs.io and ensure it's on your PATH.
The oracle will automatically detect and use it.

### Fallback: Topological Orbital Model (TOM)
When xTB is unavailable, the oracle falls back to a **Topological Orbital Model** based on
particle-in-a-box and Hückel theory. TOM estimates HOMO/LUMO from:
- Longest conjugation path length (non-linear 1/L² gap scaling)
- Heteroatom perturbation analysis
- Inductive effects from fluorine, sulfone, CF₃ groups

TOM is non-linear in molecular topology and cannot be "gamed" by fragment stacking.

## Anti-Gaming Constraints

The mutation engine includes topological safeguards:
- **Max conjugation path**: 16 atoms (prevents infinitely conjugated "Frankenstein" molecules)
- **Min sp³ carbon fraction**: 20% (ensures 3D structural complexity)
- **Max rings**: 3 (electrolytes are small molecules, not drug-like macrocycles)
- **Strained ring rejection**: 3-4 membered rings (electrochemically unstable)

## Project Structure

```
src/aurelius/
├── agent/
│   ├── loop.py             # DiscoveryLoop (Evolutionary Algorithm)
│   ├── mutation.py         # SMARTS+BRICS mutation engine
│   ├── reporting.py        # SDF + JSON report generation
│   ├── state.py            # Checkpoint, convergence, feedback
│   └── selection.py        # Tournament selection + diversity penalty
├── scoring/
│   └── oracle.py           # Hybrid Quantum + GC oracle
├── screening/
│   └── tier1/filter.py     # Electrolyte viability filter
├── pipeline.py             # Pipeline orchestrator
├── config.py               # Configuration
└── types.py                # Shared type definitions
```

## Scientific References

- Bannwarth, C. et al. "GFN2-xTB — An Accurate and Broadly Parametrized Self-Consistent Tight-Binding QM Method." *J. Chem. Theory Comput.* 2019.
- Morgan, H. L. "The Generation of a Unique Machine Description for Chemical Structures." *J. Chem. Doc.* 1965.
- Heilbronner, E. & Bock, H. "The HMO Model and its Application." *Wiley-VCH* 1976.
- Degen, J. et al. "SMARTS — A Language for Describing Molecular Patterns." *J. Chem. Inf. Model.* 2008.
- Delphi, L. et al. "BRICS: Decomposition and Reassembly of Molecules." *J. Chem. Inf. Model.* 2008.

## License

MIT License
