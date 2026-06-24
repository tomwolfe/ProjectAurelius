# Quick Start Guide for Chemists

This guide explains what the proxy values mean, shows example molecules, and helps troubleshoot common issues.

## What Are Proxy Values?

Aurelius predicts physical properties using **fragment-additivity**: a molecule is decomposed into functional groups, and each group contributes additively to the final prediction. The results are _proxies_ — unitless scores calibrated against experimental data, not exact physical measurements.

### `dielectric_proxy`

- **What it measures:** How well a solvent can screen electrostatic charges (related to the dielectric constant ε).
- **Why it matters:** Higher dielectric → better Li-salt dissociation → higher ionic conductivity.
- **Typical range:** 1–25. Values above ~8 indicate a good electrolyte solvent.
- **Physical basis:** Polar groups (carbonyl, nitrile, sulfoxide) raise the proxy; non-polar groups (alkanes, aromatics) contribute little. Cyclic carbonates (EC/PC) score highest due to cooperative dipole alignment (Kirkwood g>1).

### `viscosity_proxy`

- **What it measures:** Resistance to molecular flow (related to dynamic viscosity in cP).
- **Why it matters:** Lower viscosity → faster ion transport → higher conductivity.
- **Typical range:** 0.1–10. Values below ~3 are desirable.
- **Physical basis:** Large rigid groups (rings, branched sp³ carbons) increase viscosity; small flexible groups keep it low.

### `li_solvation_proxy`

- **What it measures:** A solvent's ability to coordinate Li⁺ ions (related to donor number / Gutmann scale).
- **Why it matters:** Higher solvation → better salt dissolution → more charge carriers.
- **Typical range:** 0.5–8. Values above ~2 indicate meaningful Li⁺ binding.
- **Physical basis:** Lewis-basic heteroatoms (O, N, S) contribute strongly. Sulfoxide, amide, and aromatic nitrogen are the strongest donors. Fluorine atoms _reduce_ solvation (electron withdrawing).

---

## Example Molecules

### Ethylene Carbonate (EC)

```
SMILES: C1COC(=O)O1
```

| Proxy              | Expected Value | Interpretation                                |
|--------------------|---------------|-----------------------------------------------|
| `dielectric_proxy` | 15–20         | Excellent — cyclic carbonate (ε≈90 real)      |
| `viscosity_proxy`  | 3–5           | Moderate — ring rigidity raises viscosity     |
| `li_solvation_proxy` | 2–3         | Adequate — carbonyl O binds Li⁺               |

```bash
aurelius screen "C1COC(=O)O1"
```

EC is the benchmark high-dielectric solvent. Its five-membered ring forces the carbonate into a cis conformation with aligned dipoles (Kirkwood g>1), producing the highest dielectric of any common solvent. The ring rigidity comes at a viscosity cost.

### Dimethyl Carbonate (DMC)

```
SMILES: COC(=O)OC
```

| Proxy              | Expected Value | Interpretation                               |
|--------------------|---------------|----------------------------------------------|
| `dielectric_proxy` | 3–4           | Low — linear carbonate, dipoles cancel       |
| `viscosity_proxy`  | 1–2           | Low — small, flexible molecule               |
| `li_solvation_proxy` | 2–3         | Adequate — same carbonyl O as EC             |

```bash
aurelius screen "COC(=O)OC"
```

DMC is the archetypal low-viscosity co-solvent. Unlike EC, its carbonate adopts an anti-periplanar conformation where dipoles partially cancel (Kirkwood g<1). DMC is never used alone — it is paired with EC or another high-dielectric solvent.

### Acetonitrile (ACN)

```
SMILES: CC#N
```

| Proxy              | Expected Value | Interpretation                                |
|--------------------|---------------|-----------------------------------------------|
| `dielectric_proxy` | 8–12          | High — strong C≡N dipole (μ≈3.9 D)           |
| `viscosity_proxy`  | < 1           | Very low — tiny, flexible molecule            |
| `li_solvation_proxy` | 2–3         | Adequate — nitrile N coordinates Li⁺          |

```bash
aurelius screen "CC#N"
```

ACN demonstrates how a single polar group (nitrile) can deliver high dielectric proxy without a large viscosity penalty. Its low viscosity makes it an excellent benchmark for the "high-dielectric, low-viscosity" ideal, though its electrochemical stability limits real battery use.

---

## Running the Examples

### Screen a single molecule

```bash
aurelius screen "C1COC(=O)O1"
```

### Screen with a non-default property pack

```bash
aurelius screen "c1ccccc1c1ccccc1" --pack organic_electronics
```

### View per-objective scorecard

```bash
aurelius validate "C1COC(=O)O1"
```

---

## Troubleshooting

### `Error: Invalid SMILES provided.`

- Check your SMILES string is valid. Test it with RDKit directly:
  ```python
  from rdkit import Chem
  mol = Chem.MolFromSmiles("C1COC(=O)O1")  # returns None if invalid
  ```
- Common mistakes: unbalanced parentheses, unconjugated radical dots, invalid valence.
- Use the `doctor` command to verify RDKit is installed:
  ```bash
  aurelius doctor
  ```

### `xTB binary not found on PATH — TOM fallback active.`

- This is a **warning, not an error**. The engine falls back to the Tight-Binding Orbital Model (TOM), a heuristic that estimates HOMO/LUMO from fragment-additivity.
- For production use, install xTB:
  1. Download from https://github.com/grimme-lab/xtb/releases
  2. Extract and add the binary to your `PATH`
  3. Verify: `aurelius doctor-xtb`
- Without xTB, orbital predictions use TOM (reduced accuracy for novel scaffolds).

### `RDKit is not installed.`

- Aurelius requires RDKit. Install via conda:
  ```bash
  conda install -c conda-forge rdkit
  ```
- Or via pip:
  ```bash
  pip install rdkit
  ```

### `Aurelius Score: 0.0/100 REJECTED`

A score of 0.0 usually means the molecule failed a physical filter (Tier 1):
- **Too heavy:** MW > 500
- **Too flexible:** > 20 rotatable bonds
- **Too many fluorine atoms without polar groups:** Outside GC calibration domain
- **Invalid valence or strained ring:** RDKit parsing failure

Run `aurelius validate "<SMILES>"` to see which objective caused rejection.

### I changed `_GC_FRAGMENTS` but nothing happens

- The fragment list is read at import time. Restart the Python process.
- Verify your SMARTS pattern matches by testing in RDKit:
  ```python
  from rdkit import Chem
  patt = Chem.MolFromSmarts("[CX3](=O)[OX2]")
  mol = Chem.MolFromSmiles("CCOC(=O)C")
  print(mol.HasSubstructMatch(patt))
  ```

### Parity plot looks bad after kernel tuning

A poor parity plot (low Spearman ρ after tuning) indicates the kernel parameters are not well-suited to your test set. Try:
1. Expand your test set (≥10 molecules minimum).
2. Run the `KernelOptimizer` with more iterations.
3. Check the `uncertainty_weights` in the kernel — high-UQ molecules will not improve.
