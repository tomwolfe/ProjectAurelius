# Project Aurelius

**Discover novel battery electrolytes — without black-box ML.**

A physically-grounded Evolutionary Algorithm with a hybrid quantum chemistry / fragment-additivity oracle. HOMO/LUMO from xTB (or Topological Orbital Model fallback). Bulk properties — dielectric, viscosity, Li⁺ solvation, conductivity — from interpretable group-contribution additivity.

**Anti-Frankenstein gates.** BRICS recombination can stack ester groups or string 14-carbon chains. Aurelius rejects these explicitly: Michaelis-Menten saturation ceilings, DFS chain-length checks, valence sanity, ring strain limits. Eight topological gates — each a 5-line decorated function.

**Self-verifying architecture.** A `Net Progress` objective function penalizes code complexity (lines of code, cyclomatic complexity, dependencies, file count) while rewarding discovery value. The CI test asserts `Net Progress > 0` — every commit must add more science than bloat.

**Zero ML bloat.** No PyTorch. No TensorFlow. Just RDKit, xTB, and pure Python.

**External validation:** Live benchmarking confirms strong positive rank correlation (ρ > 0.50) for quantum properties and (ρ > 0.80) for bulk GC proxies against published experimental data. Run `python -m benchmarks.benchmark_external_validation` for current metrics.

```bash
aurelius init                          # Initialize pipeline
aurelius doctor                        # Validate dependencies
aurelius screen "CC(=O)OC1=CC=CC=C1"   # Screen a molecule
aurelius agent --max-generations 50    # Run autonomous discovery
aurelius mixture "C1COC(=O)O1" "COCCOC" --frac 0.5  # Screen EC/DME 50:50
```

MIT License — github.com/tomwolfe/ProjectAurelius
