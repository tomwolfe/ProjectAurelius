# Project Aurelius — Domain Tuning Protocol

## Overview

The Aurelius engine ships with universal default parameters that perform
reasonably well across broad chemical space. For optimal accuracy within a
**specific chemical domain** (e.g., fluorinated carbonates, ether-based
electrolytes, sulfone co-solvents), parameters can be tuned locally using the
built-in `KernelOptimizer`.

## Why Domain Tuning?

- **TOM calibration**: corrects systematic bias in frontier-orbital predictions
  for a given domain.
- **GC fragment corrections**: adds or adjusts group-contribution terms for
  substructures that are underrepresented in the generic training data.
- **UQ calibration**: ensures uncertainty intervals have calibrated coverage.

## Tuning Workflow

```
Experimental Data (SMILES + values)
         │
         ▼
┌─────────────────┐
│ Kernel Optimizer │  Nelder-Mead tuning of TOM offsets + GC scales
└────────┬────────┘
         │ kernel draft
         ▼
┌────────────────────┐
│ Validation           │  Hold-out validation metrics
└────────┬───────────┘
         │ tuned kernel
         ▼
┌──────────────────────┐
│ aurelius_kernel.json │  Ready for use
└──────────────────────┘
```

### 1. Data Preparation

Provide a dataset of **(SMILES, experimental value)** pairs for properties of
interest (HOMO, LUMO, dielectric constant, viscosity, Li+ solvation energy).
A minimum of 20 data points is recommended.

### 2. Optimisation

The `KernelOptimizer` minimises the prediction error of the engine against the
experimental data by adjusting:

- **TOM offsets** (eV-scale corrections to HOMO/LUMO)
- **GC fragment corrections** (per-substructure additive terms)
- **UQ ensemble weights** (variance scaling for prediction intervals)

The objective function is a weighted combination of Spearman rank correlation,
mean absolute error, and interval sharpness.

### 3. Validation

A held-out portion of the data (typically 20%) is used to measure validation
metrics:

- Spearman rank correlation (ρ)
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Coverage probability

### 4. Usage

```bash
aurelius tune experiments.csv --output my_kernel.json
```

```python
from aurelius.pipeline import AureliusPipeline

pipeline = AureliusPipeline()
pipeline.load_kernel("my_kernel.json")
results = pipeline.screen_smiles("C1COC(=O)O1")
```
