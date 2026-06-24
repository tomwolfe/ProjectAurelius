# Contributing Guide: Adding GC Fragments

This guide explains how to extend the group-contribution (GC) fragment-additivity model with new SMARTS patterns and property contributions.

## Where Fragments Live

All electrolyte fragments are defined in `engine/src/aurelius/scoring/oracle/gc.py` in the `_GC_FRAGMENTS` list:

```python
# (pattern, name, dielectric_contrib, viscosity_contrib, li_solvation_contrib, ced_contrib)
_GC_FRAGMENTS: list[tuple[Chem.Mol, str, float, float, float, float]] = [
    (Chem.MolFromSmarts("[CX3](=O)([#6])[OX2H0]"), "ester", 2.5, 0.6, 0.8, 2.0),
    ...
]
```

For property packs in other domains (e.g., `organic_electronics`), fragments live in their respective pack file under `engine/src/aurelius/scoring/oracle/packs/`.

## Fragment Template

Each fragment is a tuple with the following fields:

```python
(Chem.MolFromSmarts("[SMARTS]"), "name", dielectric_contrib, viscosity_contrib, li_solvation_contrib, ced_contrib)
```

| Field              | Type  | Description                                          |
|--------------------|-------|------------------------------------------------------|
| `Chem.MolFromSmarts(...)` | `Chem.Mol` | RDKit SMARTS pattern for substructure matching |
| `"name"`           | `str` | Human-readable fragment name (used in cross-terms)   |
| `dielectric_contrib` | `float` | Contribution to dielectric proxy (0.0–8.0)         |
| `viscosity_contrib`  | `float` | Contribution to viscosity proxy (−1.0–3.0)         |
| `li_solvation_contrib` | `float` | Contribution to Li+ solvation proxy (0.0–4.0)     |
| `ced_contrib`       | `float` | Contribution to cohesive energy density (0.0–6.0)    |

For non-electrolyte packs (e.g., organic electronics), the tuple may have fewer fields. Follow the existing pattern in the target pack file.

## How to Add a New Fragment

### Step 1: Design the SMARTS Pattern

- Test your SMARTS in RDKit before submitting:
  ```python
  from rdkit import Chem
  patt = Chem.MolFromSmarts("your_smarts_here")
  test_mol = Chem.MolFromSmiles("your_test_molecule")
  print(test_mol.HasSubstructMatch(patt))
  if patt is None:
      print("Invalid SMARTS — check syntax")
  ```
- Be specific enough to avoid false positives. For example, `[CX3](=O)` matches a carbonyl carbon, while `C(=O)` would also match carboxylic acids.
- Use SMARTS features like atom numbering, ring membership, and hybridization to control specificity.

### Step 2: Determine Contribution Values

**Crucial: every contribution must have experimental justification.**

Good sources of experimental data:
- Published dielectric constants (ε) at 25 °C
- Dynamic viscosity (cP) at 25 °C
- Gutmann donor numbers (DN) for Li+ solvation
- Cohesive energy density from vaporisation enthalpies

**Principles for assigning values:**

1. **Start small.** A new functional group should contribute less than existing well-characterised groups (e.g., a new polar group should contribute less to dielectric than carbonate's 2.0).
2. **Think about saturation.** The model applies Michaelis-Menten saturation internally — a fragment with `dielectric_contrib=2.0` will contribute at most 2.0 after saturation, even if present many times.
3. **Negative contributions are allowed** for viscosity (e.g., ether: −0.4) and Li+ solvation (e.g., fluorine: −0.5), but must be justified.

### Step 3: Add Cross-Terms (If Needed)

If your fragment interacts non-linearly with an existing fragment, add a cross-term to `_ELECTROLYTE_CROSS_TERMS`:

```python
("new_fragment", "existing_fragment", 0.3, "description of the interaction"),
```

Cross-terms should be in the range [−1.0, 1.0] and justified by literature.

### Step 4: Submit with Justification

Your pull request should include:

1. The new fragment tuple(s)
2. A comment block above the fragment citing the experimental source
3. At least one test molecule demonstrating the fragment is matched correctly
4. Expected proxy values for the test molecule

## Example Submission

```python
# Sulfolane (tetramethylene sulfone, ε≈44, η≈10 cP at 30 °C).
# Dielectric contribution: sulfolane ε=44, acyclic sulfones ε=30-35.
# The 1.5 boost over the general "sulfone" (5.0) reflects the
# ring-locked S=O dipole orientation with higher effective polarisation.
# Source: J. Chem. Eng. Data 1967, 12, 2, 244-248.
(Chem.MolFromSmarts("[SX4](=O)(=O)1[CX4][CX4][CX4][CX4]1"), "sulfolane", 1.5, 2.0, 0.0, 1.5),
```

## Testing Your Fragment

After adding, verify with the existing test suite:

```bash
cd engine
pytest tests/test_property_packs.py -v
```

Add a test case in `test_property_packs.py` that validates your fragment's contribution:

```python
def test_my_new_fragment_increases_dielectric(self) -> None:
    pack = ElectrolytePack()
    ctx = MoleculeContext.from_smiles("SMILES_with_fragment")
    assert ctx is not None
    ctx_plain = MoleculeContext.from_smiles("SMILES_without_fragment")
    assert ctx_plain is not None
    assert pack.predict_dielectric(ctx) > pack.predict_dielectric(ctx_plain)
```

## Common Pitfalls

- **Overlapping patterns:** If your SMARTS is a subset of an existing pattern, both will match and double-count. Use SMARTS specificity (e.g., atom numbering, ring requirements) to distinguish.
- **Valence errors:** RDKit `MolFromSmarts` returns `None` for invalid SMARTS. Always check.
- **Unjustified values:** A fragment with `dielectric_contrib=5.0` needs a strong physical justification and literature citation.
- **Missing cross-terms:** If your fragment commonly co-occurs with another (e.g., carbonate + ether in glymes), add a cross-term or the model will undervalue the synergy.
