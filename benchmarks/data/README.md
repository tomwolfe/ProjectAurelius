# Benchmark Data Provenance

## `dielectric_verified.json`

55 static dielectric constants at 25 °C (unless noted), used as the primary
accuracy gate for the Kirkwood-Fröhlich model in
`src/aurelius/scoring/oracle/gc.py`.

Every entry was checked on two axes before inclusion:

1. the SMILES parses and its molecular formula matches the named compound;
2. the ε value is traceable to CRC Handbook 97th ed. ("Permittivity of
   Liquids") or Xu, *Chem. Rev.* **104** (2004) 4303 for battery carbonates.

This file is the reference set for `benchmark_external_validation.py`.
Do not add an entry without a citation and a formula check.

## `quarantined_benchmark_entries.json`

16 entries removed from `src/aurelius/data/external_property_benchmark.json`
by the audit in `benchmarks/audit_benchmark_integrity.py`.

They fail structural integrity, not merely model disagreement. Categories:

- **11 unparseable SMILES** — e.g. `2-Fluoroethyl sulfolane`
  (`C1CS(=O)(=O)CC(F)F`, unclosed ring), `Decalin`
  (`CC1CC2OCCO2C1`, which is not decalin). These silently became RDKit
  `None` and were scored as prediction 0.0, inflating reported error.
- **2 values contradicting CRC** — `Butyronitrile` listed at ε = 3.4 and
  `Isobutyronitrile` at ε = 2.7. CRC gives 24.83 and 20.4. Nitriles are
  strongly polar (μ ≈ 3.9 D); ε ≈ 3 is not physically possible for them.
- **3 name/structure mismatches** — entries named "…carbonate" whose SMILES
  contain no `O-C(=O)-O` group.

Retained here rather than deleted so the exclusion is auditable and
reversible if primary sources are located.

### Why this mattered

Splitting the original 120-entry file by whether its citation was
literature-traceable gave, for the Kirkwood-Fröhlich model:

| subset | n | MAE | Spearman ρ |
| --- | --- | --- | --- |
| CRC / Xu / Izutsu cited | 52 | 4.70 | 0.675 |
| uncited | 53 | 11.77 | 0.144 |

Most of the apparent "model error" on the uncited half was error in the
reference data. Reporting a single MAE over the union would have understated
model quality while overstating benchmark rigour.
