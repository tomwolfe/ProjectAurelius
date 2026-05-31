# Project Aurelius v9.0

**Novel molecule discovery for battery electrolytes.**

A focused Bayesian active-learning pipeline: molecules are mutated via SMARTS chemistry and BRICS scaffold hopping, scored by a QSPR Random Forest oracle predicting HOMO/LUMO frontier orbitals, and intelligently selected via Novelty-Weighted Expected Improvement from an RF surrogate.

## Overview

| Component | Framework | Purpose |
|-----------|-----------|---------|
| Filter | RDKit | Electrolyte viability (MW, HBD, RotB, SA score) |
| Oracle | Random Forest + QM9 | HOMO/LUMO frontier orbital energy prediction |
| Mutation | SMARTS + BRICS | Targeted electrolyte edits + scaffold hopping |
| Surrogate | Random Forest | NWEI acquisition for Bayesian candidate selection |

The composite Aurelius Score is computed via Gaussian LUMO reward (SEI formation window), sigmoid HOMO penalty (oxidative stability threshold), and SA score penalty. No aqueous solubility — that is physically meaningless for organic battery electrolytes.

## Quick Start

```bash
aurelius init                    # Initialize pipeline
aurelius doctor                  # Validate dependencies
aurelius screen "CC(=O)OC1=CC=CC=C1"  # Screen a molecule
aurelius batch examples/molecules.smi --output results.json  # Batch screen
aurelius train                  # Retrain Oracle RF model
aurelius agent --max-generations 50  # Run autonomous discovery loop
```

## CLI Reference

```
aurelius init                    Initialize pipeline
aurelius doctor                  Validate dependencies and hardware
aurelius screen <smiles>         Screen a single molecule
aurelius batch <file>            Screen molecules from SMILES file
aurelius score <smiles>          Compute Aurelius score only
aurelius train                   Train QM9 surrogate model
aurelius validate <smiles>       Run physics validation
aurelius agent                   Run the autonomous screening agent
```

## Project Structure

```
src/aurelius/
├── agent/
│   ├── loop.py             # DiscoveryLoop (Bayesian active learning)
│   ├── mutation.py         # SMARTS+BRICS mutation engine
│   ├── reporting.py        # SDF + JSON report generation
│   ├── state.py            # Checkpoint, convergence, feedback
│   └── surrogate.py        # Random Forest NWEI surrogate
├── cli_scripts/
│   ├── agent.py            # Autonomous screening agent
│   └── validate_physics.py # Physics validation
├── scoring/
│   └── oracle.py           # RF-based HOMO/LUMO oracle
├── screening/
│   └── tier1/filter.py     # Electrolyte viability filter
├── pipeline.py             # Pipeline orchestrator
├── config.py               # Configuration
└── types.py                # Shared type definitions
```

## Scientific References

- Morgan, H. L. "The Generation of a Unique Machine Description for Chemical Structures." *J. Chem. Doc.* 1965.
- Ramakrishnan, R. et al. "Quantum Chemistry Structures and Properties of 134 Kilo Molecules." *Sci. Data* 2014.
- Degen, J. et al. "SMARTS — A Language for Describing Molecular Patterns." *J. Chem. Inf. Model.* 2008.
- Delphi, L. et al. "BRICS: Decomposition and Reassembly of Molecules." *J. Chem. Inf. Model.* 2008.

## License

MIT License
