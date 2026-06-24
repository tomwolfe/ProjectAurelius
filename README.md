# Project Aurelius — Open Core

**Novel molecule discovery for battery electrolytes.**

This repository follows an **Open Core** model:

- [`engine/`](engine/) — **MIT-licensed** Discovery Engine. Fully functional,
  self-verifying evolutionary algorithm pipeline with a hybrid quantum + GC
  oracle. Installable as `pip install aurelius-engine`.
- [`certification-lab/`](certification-lab/) — **Proprietary** Certification
  Tools for tuning the engine to specific chemical domains. Produces signed
  **Certified Kernels** (`aurelius_kernel.json`).
- [`docs/`](docs/) — Certification protocol documentation and JSON Schema
  definitions.

## Quick Start (Open-Source Engine)

```bash
pip install aurelius-engine
aurelius screen "CC(=O)OC1=CC=CC=C1"
```

For detailed explanations of proxy values (dielectric, viscosity, Li+ solvation)
with curated examples and troubleshooting, see
[`docs/quickstart_chemist.md`](docs/quickstart_chemist.md).

For full CLI reference, see [`engine/README.md`](engine/README.md).

## Certified Kernels

A **Certified Kernel** is a signed JSON artifact that adjusts the engine's
TOM/GC parameters for optimal accuracy within a specific chemical domain
(e.g., fluorinated carbonates, ether electrolytes). Contact the Aurelius team
for custom certification campaigns.

See [`docs/certification_protocol.md`](docs/certification_protocol.md) for the
conceptual overview and [`docs/kernel_schema.json`](docs/kernel_schema.json)
for the schema definition.

## License

- **Engine**: MIT License — see [`engine/LICENSE`](engine/LICENSE) (or root `LICENSE`).
- **Certification Lab**: Proprietary — see `certification-lab/NOTICE`.

## Architecture

For a detailed breakdown of the pipeline, property pack system, agent loop,
and certification workflow, see [`docs/architecture.md`](docs/architecture.md).

## Repository Structure

```
project-aurelius/
├── LICENSE                   # MIT License (engine)
├── README.md                 # This file
├── .gitignore
├── engine/                   # [PUBLIC] MIT-licensed
│   ├── src/aurelius/         # Core Python package
│   ├── tests/
│   ├── pyproject.toml
│   └── examples/
├── certification-lab/        # [PRIVATE] Proprietary
│   ├── src/certifier/
│   ├── scripts/
│   └── pyproject.toml
└── docs/
    ├── quickstart_chemist.md
    ├── architecture.md
    ├── certification_protocol.md
    ├── contributing_fragments.md
    └── kernel_schema.json
```
