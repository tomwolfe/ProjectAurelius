1/5 🧵

**Anti-Frankenstein molecular design: why your black-box ML is generating impossible battery electrolytes — and how to fix it**

Every computational chemistry lab has seen it: a generative model outputs "C(C)(C)(C)C" as a top candidate — a pentavalent carbon that violates freshman organic chemistry. The model doesn't know any better because it was trained on SMILES strings, not physics.

2/5

**The problem: fragment stacking**

Black-box ML models discover that ester fragments correlate with high dielectric constants. So they stack 5 ester groups onto a single molecule and the model says "excellent candidate!" — ignoring that real dielectrics saturate via the Onsager reaction field.

Aurelius rejects this explicitly. The oracle uses Michaelis-Menten saturation ceilings: `f(n) = min(ΔD_i · n_i, cap_i)`. Five esters do NOT give five times the dielectric boost of one ester. The test `test_fragment_saturation_prevents_stacking` enforces this.

The oracle also predicts ionic conductivity (Walden-product proxy combining dielectric, viscosity, and Li+ solvation) and handles solvent mixtures with Margules-inspired non-ideal mixing synergy.

`src/aurelius/scoring/oracle/gc.py` — saturation, cross-terms, conductivity, and mixture logic, line by line, no ML.

3/5

**The 14-carbon chain problem**

BRICS recombination (connecting molecular fragments) can accidentally produce molecules with continuous aliphatic chains of 14+ carbons. These are synthetically inaccessible, insoluble in electrolyte formulations, and have zero heteroatom density — useless for Li+ conduction.

Aurelius runs a DFS on every BRICS product:

```
def has_excessive_aliphatic_chain(mol, max_chain=12):
    # depth-first search counts consecutive sp3 carbons
    return longest[0] > max_chain
```

(`src/aurelius/agent/mutation/brics.py:90`)

No data required. No training. Just graph traversal.

4/5

**The full gate system**

Aurelius applies 8 explicit topological checks before any molecule sees the oracle:

| Gate | What it catches |
|------|-----------------|
| Aliphatic chain ≤ 12 | Frankenstein linkers |
| Rings ≤ 3 | Macrocyclic artifacts |
| No 3-4 membered rings | Electrochemical instability |
| sp³ fraction ≥ 20% | Flat, poorly solvating molecules |
| Conjugation ≤ 16 | "Infinite wire" polyenes |
| Aromatic rings ≤ 2 | Drug-like overreach |
| Valence sanity | Pentavalent carbons, hypervalent O |
| Heteroatom ratio ≥ 0.25 | Pure hydrocarbon nonsense |

Each is a 5-line Python function decorated with `@_register` in `smarts.py`. Data-driven, auditable, composable.

5/5

**Why this matters for battery research**

A generative model that can't distinguish between a realizable carbonate ester and an impossible 14-carbon chain isn't just wrong — it's wasting synthesis resources and eroding trust in computational discovery.

External validation against published experimental data confirms Spearman ρ = 0.76 for LUMO (N=26, p<0.0001) and positive correlation across all five benchmarked properties (dielectric, viscosity, donor number, HOMO, LUMO). The mutation engine achieves 93.5% novel scaffold discovery.

Aurelius is open source (MIT), CLI-first, zero ML frameworks. The entire anti-gating system is a few hundred lines of RDKit + Python.

`aurelius screen "CC(=O)OC1=CC=CC=C1"` — the gate rejects or accepts before the oracle even runs.

**No PyTorch. No TensorFlow. Just physics and graph theory.**

https://github.com/tomwolfe/ProjectAurelius
