# Aurelius

Evolutionary Algorithm screening pipeline with a hybrid quantum / group-contribution
(QM + GC) orbital oracle and a Gaussian-penalty objective.

> v12.0.0 — Scoring oracle is the single source of truth for `homo_eV` / `lumo_eV`.
> The Δ-learning residual is trained and evaluated on the **same base model** (LPM for
> HOMO, TOM for LUMO) — see `ADR-2026-08-09-02` and
> `src/aurelius/scoring/oracle/delta_correction.py`.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
# optional, for Apple GPU acceleration
pip install -e ".[gpu]"
```

## Quick start

```bash
# Score a molecule from its SMILES
aurelius score "c1ccccc1" --property homo_eV

# Run an evolutionary discovery loop
aurelius run --population 50 --generations 20
```

## Scoring oracle

`aurelius.scoring.oracle` assembles the property predictions consumed by the
evolutionary loop:

| Property        | Source                                                                       |
|-----------------|------------------------------------------------------------------------------|
| `homo_ev`       | LPM lone-pair model, refined by the Δ-correction residual                    |
| `lumo_eV`       | Topological orbital model (TOM), refined by the Δ-correction residual        |
| `gap_eV`        | `lumo_eV - homo_eV`                                                          |
| `mlp_eV`        | Solvated-electron ML model                                                   |
| `dft_bridge`    | Conformal xTB bridge for low-confidence candidates                           |

The Δ-correction layer (`DeltaCorrection`) fits a kernel-ridge residual on
**ECFP4 fingerprints** so the interpretable physics-based base model is preserved
while systematic bias is removed. Out-of-domain corrections are damped by the GPR
predictive variance (`predict_corrected*`).

## Development

```bash
# tests (with coverage)
pytest

# lint
ruff check . && ruff format --check .

# types
mypy src/aurelius

# fast subset (no DFT subprocesses)
pytest tests/test_reporting.py tests/test_delta_correction.py -m "not slow"
```

## Project layout

```
src/aurelius/
  scoring/oracle/     # hybrid QM+GC property models (HOMO/LUMO, Δ-correction)
  agent/              # evolutionary algorithm, discovery loop, experiment suggester
  screening/          # tier-0 / tier-1 filters
tests/
benchmarks/           # throughput & closed-loop benchmarks
```
