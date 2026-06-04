# Show HN: Project Aurelius — A self-penalizing evolutionary algorithm for electrolyte discovery

I built an open-source evolutionary algorithm for battery electrolyte discovery that has a unique feature: a test that penalizes ME (the developer) for writing too much code.

## The self-penalizing simplicity metric

The repo includes a test (`tests/test_net_progress.py`) that defines:

- **Discovery Value** — measures what the pipeline actually achieves: rediscovery rate, scaffold novelty, top-k enrichment, holdout generalization, experimental trend recovery
- **Simplicity Cost** — measures my laziness: lines of code, cyclomatic complexity violations, third-party dependency count, architectural surface area (number of public classes/functions)

**Net Progress = Discovery Value - 0.35 × Simplicity Cost**

The test asserts `Net Progress > 0`. Every commit must add more scientific value than it adds code bloat. If I add a feature without improving discovery, or if I over-engineer a simple fix, the test fails. It's a lint-check for research productivity.

## Why this matters for computational chemistry

Most molecular generation pipelines are black boxes — you feed in SMILES and get scores out, with no idea whether the model is hallucinating impossible molecules. Aurelius uses:

1. **A hybrid quantum + fragment-additivity oracle** — xTB/GFN2-xTB for HOMO/LUMO (quantum phenomena aren't additive), group-contribution for bulk properties (dielectric, viscosity, Li+ solvation)
2. **Anti-gaming gates** — BRICS reassembly can produce "Frankenstein" molecules with 14-carbon aliphatic chains. The pipeline explicitly rejects them via DFS-based chain-length checks, valence sanity, ring strain limits, and sp³ fraction minimums
3. **Domain-of-applicability penalties** — if a molecule is too fluorinated or has excessive conjugation, the oracle discounts its own score rather than confidently giving wrong answers

## CLI-first, zero ML bloat

```bash
pip install aurelius  # or: conda install -c conda-forge aurelius
aurelius doctor        # validate dependencies
aurelius screen "CC(=O)OC1=CC=CC=C1"  # screen a molecule
aurelius agent --max-generations 50   # run autonomous discovery
```

No PyTorch, no TensorFlow, no JAX. The AST-level test (`test_no_ml_framework_imports_via_ast`) scans every .py file and fails if it finds ML framework imports. If a future contributor tries to add a neural network, the test catches it before review.

## Stack

- RDKit for molecular manipulation (BRICS, SMARTS, fingerprints)
- xTB/GFN2-xTB for quantum chemistry (or TOM fallback — a closed-form Hückel/particle-in-a-box model — when xTB isn't installed)
- Pure Python group-contribution fragment-additivity for bulk properties
- Click CLI, structlog for structured logging

## Comparative validation

External validation against published experimental data shows Spearman ρ = 0.76 for LUMO predictions and positive rank correlation across all five benchmarked properties (dielectric, viscosity, donor number, HOMO, LUMO). The mutation engine achieves >20% novel scaffold discovery rate in 3-generation loops.

## Why "self-verifying"?

Because the codebase verifies itself. The philosophy tests check that:
- The oracle is actually non-linear (cross-terms, fragment saturation)
- The anti-gaming gates actually reject bad molecules
- The architectural complexity stays bounded
- The discovery yield doesn't silently degrade with new features

**Repo:** https://github.com/tomwolfe/ProjectAurelius

**Preprint:** `paper/manuscript.md` in the repo
