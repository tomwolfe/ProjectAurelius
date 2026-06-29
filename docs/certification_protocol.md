# Aurelius Certification Protocol

## Overview

The Aurelius Certification Protocol is the process by which a generic
open-source Discovery Engine is tuned and validated for a **specific chemical
domain** (e.g., fluorinated carbonates, ether-based electrolytes, sulfone
co-solvents). The output is a **Certified Kernel** — a signed JSON artifact
that adjusts the engine's predictive parameters for optimal accuracy within
that domain.

## Why Certification?

The open-source Discovery Engine ships with universal default parameters that
perform reasonably well across broad chemical space. A Certified Kernel
narrows the focus:

- **TOM calibration**: corrects systematic bias in frontier-orbital predictions
  for a given domain.
- **GC fragment corrections**: adds or adjusts group-contribution terms for
  substructures that are underrepresented in the generic training data.
- **UQ calibration**: ensures uncertainty intervals have calibrated coverage
  (e.g., 95% of error bars contain the experimental value).

## Certification Workflow

```
Experimental Data (SMILES + values)
        │
        ▼
  ┌─────────────────┐
  │ Kernel Optimizer │  Multi-objective tuning of TOM/GC params
  └────────┬────────┘
           │ kernel draft
           ▼
  ┌────────────────────┐
  │ Uncertainty Auditor │  Hold-out validation + coverage check
  └────────┬───────────┘
           │ validated kernel
           ▼
  ┌──────────────┐
  │ Kernel Signer │  HMAC-SHA256 signature
  └──────┬───────┘
         │ signed kernel
         ▼
  ┌──────────────────────┐
  │ aurelius_kernel.json │  Ready for distribution
  └──────────────────────┘
```

### 1. Data Preparation

The client provides a dataset of **(SMILES, experimental value)** pairs for
properties of interest (HOMO, LUMO, dielectric constant, viscosity, Li+
solvation energy). A minimum of 20 data points is recommended.

### 2. Optimisation

The Kernel Optimizer minimises the prediction error of the engine against the
experimental data by adjusting:

- **TOM offsets** (eV-scale corrections to HOMO/LUMO)
- **GC fragment corrections** (per-substructure additive terms)
- **UQ ensemble weights** (variance scaling for prediction intervals)

The objective function is a weighted combination of Spearman rank correlation,
mean absolute error, and interval sharpness.

### 3. Validation (Uncertainty Audit)

A held-out portion of the data (typically 20%) is used to measure **coverage
probability** — the fraction of experimental values that fall within the
engine's predicted uncertainty intervals. A passing kernel must achieve ≥90%
coverage at the 95% confidence level.

### 4. Signing

The validated kernel is serialised to canonical JSON and signed with
**HMAC-SHA256**. The signature ensures integrity and authenticates the kernel
as originating from an authorised certifier.

## Certified Kernel Schema

See [`kernel_schema.json`](kernel_schema.json) for the full JSON Schema
definition. Key fields:

| Field | Description |
|-------|-------------|
| `version` | Schema version (semver) |
| `domain_boundary` | Chemical space definition (MW, fragments, LogP limits) |
| `tom_parameters` | Tuned HOMO/LUMO offsets and GC/UQ scale factors |
| `gc_fragments` | Custom fragment corrections for the GC model |
| `uq_weights` | Uncertainty quantification ensemble weights |
| `validation_metrics` | Spearman ρ, MAE, and audit coverage |
| `signature` | HMAC-SHA256 hex digest |

## Using a Certified Kernel

Once you have an `aurelius_kernel.json` file, load it into the Discovery Engine:

```python
from aurelius.pipeline import AureliusPipeline

pipeline = AureliusPipeline()
pipeline.load_kernel("aurelius_kernel.json")
results = pipeline.screen("CC(=O)OC1=CC=CC=C1")
```

The engine automatically applies the kernel's TOM offsets, GC corrections, and
UQ weights during scoring.

## Security

- Kernels are signed with per-client secrets — never share your `AURELIUS_SECRET`.
- Verify kernel signatures before use with the `KernelSigner.verify()` method.
- The certification-lab tooling is proprietary; contact the Aurelius team for
  custom certification campaigns.

## Joint Venture Data & IP Protocol

In Joint Venture engagements, the Certification Lab operates under a strict **Data Clean Room** protocol:

1. **Data Ingestion:** Your experimental CSV data is ingested into an isolated, encrypted instance of the Certification Lab.
2. **Kernel Generation:** A proprietary kernel is tuned specifically to your chemical domain. This kernel is signed but **never leaves your VPC** unless explicitly authorized for verification.
3. **Candidate Handoff:** Top candidates are delivered as SMILES strings with predicted properties.
4. **Royalty Tracking:** Each candidate is tagged with a unique `Discovery ID` linked to the JV contract. Commercialization of molecules bearing this ID triggers the agreed-upon royalty (1–3%).
