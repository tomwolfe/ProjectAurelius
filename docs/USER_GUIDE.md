# Project Aurelius — User Guide

**Novel molecule discovery for battery electrolytes, organic electronics, and catalysis.**

This guide covers installation, configuration, and CLI usage for the open-source
Discovery Engine (`aurelius-engine`). For the architecture overview, see
[`architecture.md`](architecture.md). For the proprietary Certification Lab
and kernel signing workflow, see [`certification_protocol.md`](certification_protocol.md).

---

## Installation

Since Aurelius relies on RDKit's C++ bindings, a conda-first installation is recommended.

### Conda (Recommended)

```bash
conda create -n aurelius python=3.11 rdkit -c conda-forge
conda activate aurelius
pip install aurelius-engine
```

### pip (Linux-only, requires pre-installed RDKit)

```bash
pip install aurelius-engine
```

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[cli]` | `rich` | Colored terminal output |
| `[ml]` | `numpy`, `scikit-learn`, `joblib`, `scipy` | Surrogate oracle, kernel tuning |
| `[web]` | `fastapi`, `uvicorn`, `diskcache` | Certification Lab server |
| `[dev]` | `pytest`, `ruff`, `mypy`, `hypothesis` | Development |

```bash
pip install aurelius-engine[cli,ml]
```

---

## Quick Start

```bash
# Validate your installation
aurelius init
aurelius doctor

# Screen a single molecule
aurelius screen "C1COC(=O)O1"

# Run the full autonomous discovery loop
aurelius agent --max-generations 50 --batch-size 25
```

---

## CLI Reference

### `init`

Initialize the Aurelius pipeline (loads property models and caching).

```bash
aurelius init
aurelius init --pack organic_electronics
```

### `doctor`

Validate dependencies, hardware, and the xTB quantum backend.

```bash
aurelius doctor
aurelius doctor --verbose
```

### `doctor-xtb`

Check whether the xTB (GFN2-xTB) binary is available on `PATH`. xTB is the
preferred quantum backend; if absent, the Topological Orbital Model (TOM)
fallback is used.

```bash
aurelius doctor-xtb
```

### `screen`

Screen a single molecule through the full Filter → Oracle → Score pipeline.

```bash
aurelius screen "C1COC(=O)O1"
aurelius screen "COC(=O)OC" --pack organic_electronics
```

### `batch`

Screen multiple molecules from a SMILES file (one SMILES per line, `#` comments are ignored).

```bash
aurelius batch molecules.smi
aurelius batch molecules.smi --output results.json
aurelius batch my_set.smi --pack organic_electronics
```

### `score`

Quick score (alias for `screen` without the tier-1 filter breakdown).

```bash
aurelius score "C1COC(=O)O1"
aurelius score "CC#N"
```

### `evaluate`

Run the full ML Oracle evaluation on a molecule and print the Aurelius Score.

```bash
aurelius evaluate --smiles "C1COC(=O)O1"
```

### `validate`

Full pipeline with a detailed report card: per-objective sub-scores, weighted
contributions, fragment-level rejection insights (top 3 contributing GC fragments
per failing property), and predicted property values.

```bash
aurelius validate "C1COC(=O)O1"
aurelius validate "CC(=O)OC" --pretty
```

### `agent`

Run the autonomous evolutionary screening agent. Uses BRICS mutation,
multi-objective hybrid oracle, tournament selection, and active learning to
evolve electrolyte candidates over multiple generations.

```bash
aurelius agent
aurelius agent --max-generations 100 --batch-size 25
aurelius agent --pack organic_electronics
```

### `mixture`

Screen a binary or ternary electrolyte mixture. Uses thermodynamic mixing rules
with a Margules-inspired synergy bonus for complementary pairs
(high-dielectric + low-viscosity).

**Binary:**
```bash
aurelius mixture "C1COC(=O)O1" "COCCOC"
aurelius mixture "C1COC(=O)O1" "COCCOC" --frac 0.3
```

**Ternary:**
```bash
aurelius mixture "C1COC(=O)O1" "COCCOC" --smiles-c "CC#N" --frac-a 0.4 --frac-b 0.4
```

### `tune`

Tune kernel parameters from a CSV of experimental data (columns: `smiles`,
`property`, `value`). Runs a local Nelder-Mead optimizer (no Stripe/JWT/Postgres
required) and writes the tuned kernel to a JSON file.

```bash
aurelius tune experiments.csv
aurelius tune experiments.csv --output my_kernel.json --max-iter 500
```

### `verify-kernel`

Verify a kernel's Ed25519 signature using the public key compiled into the engine.

```bash
aurelius verify-kernel aurelius_kernel.json
```

---

## Property System

### Frontier Orbitals (HOMO/LUMO)

Predicted via a two-tier quantum oracle:

1. **xTB (GFN2-xTB)** — semi-empirical QM (sub-1.0 eV MAE). Preferred backend.
   Install from [xtb-docs.readthedocs.io](https://xtb-docs.readthedocs.io).
2. **Topological Orbital Model (TOM)** — closed-form fallback (~1.07 eV MAE).
   Based on particle-in-a-box pi-electron theory with heteroatom perturbations.

### Bulk Properties (GC Fragment-Additivity)

| Property | Method | Description |
|----------|--------|-------------|
| Dielectric ε | GC + TPSA cap + UQ Ensemble | Salt dissolution capability |
| Viscosity η | GC + MW + RotB + UQ Ensemble | Ion mobility |
| Li+ Solvation | GC fragment-additivity | Binding strength |
| CED | GC fragment-additivity | SEI mechanical robustness |
| SEI Fracture | Topological proxy | Cross-linking motifs + rigidity |
| Gas Evolution | SMARTS pattern penalty | Reductive CO₂/CO generation |
| Hydrolysis Risk | SMARTS pattern penalty | Water-reactive motif count |

### Aurelius Score

Multi-objective composite (0–100) with Gaussian and sigmoid rewards:

| Objective | Weight | Transform |
|-----------|--------|-----------|
| LUMO reward | 0.23 | Gaussian (target −1.0 eV) |
| HOMO penalty | 0.17 | Sigmoid (threshold −6.0 eV) |
| Dielectric reward | 0.17 | Sigmoid (target 5.0) |
| Viscosity penalty | 0.10 | Sigmoid (threshold 2.5) |
| Li solvation reward | 0.15 | Gaussian (target 3.5) |
| SEI fracture | 0.06 | Sigmoid (target 4.0) |
| Gas evolution penalty | 0.05 | Sigmoid (threshold 0.5) |
| Hydrolysis penalty | 0.05 | Sigmoid (threshold 1.0) |
| CED reward | 0.01 | Sigmoid (target 5.0) |
| SA penalty | 0.01 | Sigmoid (threshold 4.0) |

Additional multiplicative penalties:
- **Al corrosion** (0.7×) — high-LUMO fluorinated molecules
- **Hypofluorite** (0.5×) — O–F bonds, violently reactive
- **Building block grounding** (0.7–1.0×) — BRICS fragment coverage
- **Quantum confidence** (0.85×) — TOM low-confidence predictions

---

## Evolutionary Algorithm Agent

The autonomous screening agent (`aurelius agent`) runs a generational loop:

1. **Generate** — BRICS mutation + SMARTS reactions from seed pool
2. **Filter** — structural validity, duplicate removal, novelty gate
3. **Evaluate** — hybrid oracle (quantum + GC) with UQ ensemble
4. **Select** — tournament selection with Tanimoto diversity penalty
5. **Record** — scaffold tracking, checkpoint, convergence detection

Active learning queue: molecules with high surrogate uncertainty are forwarded
to the real QuantumOracle for accurate evaluation instead of the ML surrogate.

---

## Quantum Backend: xTB

1. Install the binary: https://github.com/grimme-lab/xtb/releases
2. Add `xtb` directory to your `PATH`
3. Verify: `aurelius doctor` should show `OK` for xtb

When xTB is unavailable, the engine falls back to the **Topological Orbital
Model (TOM)** automatically — no configuration needed.

---

## Beyond Batteries

The engine's modular hybrid oracle generalizes to other domains:

- **Organic Electronics**: Tune HOMO/LUMO targets, add charge-mobility fragments
- **Catalysis**: Reweight frontier-orbital descriptors, add metal-coordination terms
- **Small-Molecule Drugs**: Replace electrolyte viability filter with Lipinski/Ro5 rules

Domain tuning requires a **Certified Kernel** generated by the Certification Lab
(see [`certification_protocol.md`](certification_protocol.md)).

---

## License

The Discovery Engine (`aurelius-engine`) is MIT-licensed.
The Certification Lab is proprietary.
