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

**Dependency note:** The engine requires **RDKit** for all chemical graph
processing (molecular fingerprints, SMILES parsing, conformer generation,
substructure matching). Install via:
```bash
conda install -c conda-forge rdkit
# or: pip install rdkit-pypi
```

```bash
pip install aurelius-engine
aurelius screen "CC(=O)OC1=CC=CC=C1"
```

For detailed explanations of proxy values (dielectric, viscosity, Li+ solvation)
with curated examples and troubleshooting, see
[`docs/quickstart_chemist.md`](docs/quickstart_chemist.md).

For full user guide (installation, CLI reference, property system, agent usage),
see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

For the engine-specific README (architecture, quantum backend, project structure),
see [`engine/README.md`](engine/README.md).

For a detailed architecture breakdown, see [`docs/ARCHITECTURE.md`](docs/architecture.md).

### Pre-Certified Kernels (Quick Start)

Jump-start your workflow with domain-tuned, signed kernels generated from
curated public benchmark data. Load any of these into the engine via the
Certification Lab or use them as reference baselines:

| Kernel | Domain | Description |
|--------|--------|-------------|
| [`carbonate_high_voltage.json`](docs/examples/kernels/carbonate_high_voltage.json) | Carbonate | High-voltage organic carbonate electrolytes (EC, PC, DMC, DEC, EMC, FEC) |
| [`ether_low_temperature.json`](docs/examples/kernels/ether_low_temperature.json) | Ether | Low-temperature ether-based solvents (DME, THF, Diglyme, DEE, DOL) |
| [`sulfone_stable.json`](docs/examples/kernels/sulfone_stable.json) | Sulfone | High-stability sulfone electrolytes (DMSO, Sulfolane, PS, dimethyl sulfone) |

These kernels were produced by the [generate_pre_certified_kernels.py](certification-lab/scripts/generate_pre_certified_kernels.py)
script and are ready for evaluation, comparison, or as a starting point for
further domain tuning.

## Certification Lab (SaaS)

The Certification Lab provides a B2B SaaS platform for chemists to upload experimental data,
generate certified kernels, and manage subscriptions. It features:

- **JWT authentication** — register/login with email & password
- **Kernel CRUD** — upload CSV/JSON data, view/download kernels
- **Stripe billing** — subscription management with checkout sessions
- **HTMX frontend** — single-page application with tabbed interface (kernels, upload, subscription)

### Quick Start

```bash
# Install dependencies
pip install -e .[web]

# Run database migrations
cd certification-lab
python -m alembic upgrade head

# Start the server
uvicorn server.api_server:app --host 0.0.0.0 --port 8001
```

### Setup Environment Variables

| Variable | Description |
|---|---|
| `AURELIUS_JWT_SECRET` | Secret key for JWT token signing |
| `AURELIUS_API_KEY` | Legacy API key for unauthenticated access |
| `AURELIUS_SECRET` | Ed25519 signing seed |
| `AURELIUS_DATABASE_URL` | Database connection string (default: SQLite) |
| `STRIPE_SECRET_KEY` | Stripe secret API key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook verification secret |
| `FRONTEND_URL` | Base URL for redirect URLs |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/register` | Create new user account |
| `POST` | `/auth/login` | Login and receive JWT |
| `GET` | `/kernels` | List user's kernels |
| `GET` | `/kernels/{id}` | Get single kernel |
| `POST` | `/kernels` | Create a new kernel |
| `DELETE` | `/kernels/{id}` | Delete a kernel |
| `POST` | `/kernels/certify` | Full certification pipeline |
| `POST` | `/kernels/{id}/diff/{old_id}` | Compare two kernels |
| `POST` | `/subscribe` | Create Stripe checkout session |
| `POST` | `/webhook` | Stripe webhook handler |
| `GET` | `/health` | Health check |

### Frontend Pages

| Path | Description |
|------|-------------|
| `/` | Main app shell |
| `/login` | Login page |
| `/register` | Registration page |
| `/dashboard` | User dashboard with tabs |
| `/upload` | Upload & certify data |

### Testing

```bash
cd certification-lab
python -m pytest tests/ -v
```

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
│   ├── server/               # FastAPI server, auth, templates
│   ├── db/                   # SQLAlchemy models & engine setup
│   ├── alembic/              # Database migrations
│   ├── alembic.ini           # Alembic configuration
│   ├── tests/
│   └── pyproject.toml
├── docs/
│   ├── quickstart_chemist.md
│   ├── architecture.md
│   ├── certification_protocol.md
│   ├── contributing_fragments.md
│   └── kernel_schema.json
└── data/                     # SQLite database (auto-created)
```
