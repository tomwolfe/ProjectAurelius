# Project Aurelius — Architecture

## Overview

Project Aurelius is an open-core evolutionary algorithm for battery electrolyte discovery. The system uses a **hybrid oracle** combining quantum chemistry (xTB / Tight-binding Orbital Model) with a **group-contribution (GC) fragment-additivity model** to predict electrolyte properties.

The architecture follows a **pipeline** pattern:

```
SMILES → Tier 1 (Physical Filters) → Oracle (Quantum + GC) → Scoring → Decision
```

## Repository Structure

```
project-aurelius/
├── engine/                          # [PUBLIC] MIT-licensed Discovery Engine
│   ├── src/aurelius/
│   │   ├── __main__.py              # CLI entry point (click-based)
│   │   ├── pipeline.py              # AureliusPipeline orchestrator
│   │   ├── constants.py             # Physical thresholds, SMARTS patterns
│   │   ├── types.py                 # MoleculeContext, ScreeningResult
│   │   ├── scoring/
│   │   │   └── oracle/
│   │   │       ├── oracle.py        # PropertyOracle (multi-objective)
│   │   │       ├── gc.py            # GC fragment-additivity + BasePropertyModel
│   │   │       ├── quantum.py       # QuantumOracle (xTB / TOM fallback)
│   │   │       ├── surrogate.py     # SurrogateQuantumOracle (ML surrogate)
│   │   │       └── packs/           # Property pack modules
│   │   │           ├── organic_electronics.py
│   │   │           └── __init__.py
│   │   ├── agent/                   # Evolutionary algorithm loop
│   │   │   ├── loop.py              # DiscoveryLoop
│   │   │   ├── selection.py         # Tournament selection, UCB
│   │   │   ├── mutation/            # BRICS, SMILES mutation
│   │   │   └── state.py             # LoopState
│   │   ├── screening/
│   │   │   └── tier1/               # Physical pre-filters
│   │   └── utils/                   # Dependency checks, chem utils
│   ├── tests/
│   └── pyproject.toml
│
├── certification-lab/               # [PRIVATE] Proprietary Certification Tools
│   ├── src/certifier/
│   │   ├── optimizer.py             # KernelOptimizer (Nelder-Mead)
│   │   ├── signer.py                # Ed25519 KernelSigner
│   │   ├── validator.py             # UncertaintyAuditor
│   │   └── report_generator.py      # PDF/text validation reports
│   └── tests/
│
└── docs/
    ├── quickstart_chemist.md        # Quick-start guide for chemists
    ├── architecture.md              # This file
    ├── certification_protocol.md    # Certification workflow
    ├── contributing_fragments.md    # Guide for adding GC fragments
    └── kernel_schema.json           # Certified Kernel JSON Schema
```

## Pipeline Flow

### 1. Tier 1 — Physical Filters

Filters molecules that violate basic physical constraints before running the expensive Oracle:
- MW > 500 → reject (too heavy for electrolyte use)
- Rotatable bonds > 20 → reject (too flexible)
- Extreme fluorination without polar groups → domain penalty

### 2. Oracle — Hybrid Property Prediction

The `PropertyOracle` combines three models:

| Component | Technology | Outputs |
|-----------|-----------|---------|
| Quantum Oracle | xTB (preferred) or TOM fallback | HOMO, LUMO, gap (eV) |
| GC Model (ElectrolytePack) | Fragment-additivity + cross-terms | dielectric, viscosity, Li⁺ solvation, CED proxies |
| GC UQ Ensemble | Random Forest (5 models) | Uncertainty estimates for GC predictions |

### 3. Scoring — Multi-Objective

A weighted sum of six objectives:
- **HOMO stability** (low HOMO → oxidative stability)
- **LUMO stability** (high LUMO → reductive stability)
- **Dielectric proxy** (high → good salt dissociation)
- **Viscosity proxy** (low → fast ion transport)
- **Li⁺ solvation proxy** (high → salt dissolution)
- **CED proxy** (cohesive energy density → SEI formation)

### 4. Agent — Evolutionary Loop

The `DiscoveryLoop` runs iteratively:
1. **Mutate** seed molecules via BRICS / SMILES transformations
2. **Filter** invalid or duplicate candidates
3. **Evaluate** via pipeline
4. **Select** top candidates (tournament selection + diversity penalty)
5. **Feed back** discoveries into seed pool

## Property Pack System

Property packs allow the engine to predict properties for different chemical domains without changing the core code. Each pack defines:

- **Fragments**: SMARTS patterns with per-property contribution values
- **Cross-terms**: Non-linear corrections for co-occurring fragment pairs
- **Prediction methods**: `predict_all()` returning proxy values

### Available Packs

| Pack | Domain | Proxies |
|------|--------|---------|
| `electrolyte` (default) | Battery electrolytes | dielectric, viscosity, li_solvation, CED, conductivity, li_dissociation |
| `organic_electronics` | OLED/OPV materials | hole_mobility, electron_affinity |

## Certified Kernel System

A **Certified Kernel** is a signed JSON artifact that adjusts the engine's TOM/GC parameters for optimal accuracy within a specific chemical domain. The certification workflow:

1. **Data Preparation**: Collect experimental benchmark pairs (SMILES, property values)
2. **Kernel Optimizer**: Nelder-Mead tuning of TOM offsets + GC scales to minimize MAE
3. **Uncertainty Auditor**: Validate predictions on hold-out set, check domain coverage
4. **Kernel Signer**: Ed25519 digital signature for tamper-proof distribution

See `docs/certification_protocol.md` for full details.
