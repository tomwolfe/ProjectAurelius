# Refactor Notes

## Removed

### `src/aurelius/cli_scripts/train_tier0.py`
Placeholder file that returned a static dict — no actual training logic. Dead code since Tier 0 was never implemented. All references removed from `__main__.py` (tier0 CLI branch) and `cli_scripts/__init__.py`.

### Tier0 references in `hub/uploader.py`
Removed `tier0` from task-description mapping and dataset label in generated README.

### Lipinski Rule-of-5 filter (`screening/tier1/filter.py`)
Replaced with Electrolyte Viability Filter (MW < 300, HBD == 0, RotB <= 6, HBA >= 1). Lipinski was designed for oral drugs, not battery electrolytes. The new filter blocks protic molecules (HBD > 0) that cause parasitic SEI reactions with alkali metals.

### Markdown report generators (`agent/reporting.py`)
Deleted `generate_screening_statistics`, `generate_chemical_insights`, `generate_manifest`, `generate_discovery_results`, `write_top_discoveries`, and `generate_discoveries_csv`. Replaced with single `generate_run_summary()` that writes `run_summary.json`. Output is now exactly `discoveries.sdf` + `run_summary.json`.

## Fixed

### Mutation tautology (`agent/mutation.py:_mutate_bond`)
`bond_tokens = [t for t in tokens if t in tokens]` was a no-op. Now samples from `sf.get_semantic_robust_alphabet()`.

### Sampling scope (`agent/mutation.py:_mutate_atom`)
`other_atoms` sampled only from the current molecule's tokens, limiting exploration. Now samples from the full SELFIES alphabet.

### SanitizeMol guardrails (`agent/mutation.py:_selfies_mutate`)
Added explicit RDKit `SanitizeMol` check before accepting mutation candidates, preventing invalid structures from reaching the Oracle.

### Naming drift — "Gaussian Process" → "Random Forest"
All docstring references to "Gaussian Process surrogate" updated to "Random Forest surrogate". The code always used `RandomForestSurrogate`; only documentation was out of date.

## Added

### Synthetic Accessibility penalty (`pipeline.py:_compute_score`)
Integration with RDKit's `sascorer` (with fallback heuristic) penalizes molecules that are novel but impossible to synthesise.

### QM9 domain applicability (`scoring/oracle.py`)
New `_domain_applicable()` check: molecules with atoms outside {C,H,N,O,F} or with Tanimoto similarity < 0.3 to the QM9 centroid receive a 50% score penalty (low confidence).

### Model persistence (`scoring/oracle.py`)
`PropertyOracle.save()` and `load()` using joblib. `AureliusPipeline.initialize()` checks for `oracle_cache.joblib` to skip retraining.

### Electrolyte Viability Filter test cases
- Ethanol (CCO, HBD=1) now correctly fails
- DMC (COC(=O)OC, HBD=0, O acceptors) now correctly passes
