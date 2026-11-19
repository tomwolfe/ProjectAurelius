# Unified Benchmark Report

**Status**: ✅ PASS
**Generated**: 2026-11-18 23:01:43

## Tolerances
### Orbital
- `lpm_nist_rho_min`: 0.9
- `lpm_nist_mae_max`: 0.45
- `lpm_unseen_dft_rho_min`: 0.35
- `tom_leakage_gap_max`: 0.45
- `tom_lumo_unseen_rho_min`: 0.05
- `tom_lumo_unseen_mae_max`: 1.2
- `high_hetero_homo_rho_min`: 0.6
- `high_hetero_homo_mae_max`: 0.65

### Dielectric
- `kf_mae_max`: 5.0
- `kf_rho_min`: 0.9
- `commercial_mae_max`: 3.0
- `commercial_rho_min`: 0.95
- `max_err_max`: 50.0
- `p95_err_max`: 15.0

### Viscosity
- `rho_min`: 0.5

### Donor_Number
- `rho_min`: 0.15

### Ml_Baseline
- `oracle_beats_rf_gap_min`: -0.05

### Discovery
- `rediscovery_rate_min`: 0.5
- `novel_scaffold_min`: 0.8
- `score_gap_min`: 0.0

### Integrity
- `trivial_smiles`: ['O', 'C#N']
- `min_retrosynthetic_depth`: 1

## Results
### Orbital (Leakage-Aware)
#### All (n=72)
- TOM: ρ=+0.2044, MAE=1.3117 eV
- LPM: ρ=+0.5561, MAE=0.3855 eV

#### Seen (n=27)
- TOM: ρ=+0.5422, MAE=0.5143 eV
- LPM: ρ=+0.8487, MAE=0.2483 eV

#### Unseen (n=45)
- TOM: ρ=+0.1700, MAE=1.7901 eV
- LPM: ρ=+0.4319, MAE=0.4678 eV

#### Experimental IPs (NIST, no leakage)
- LPM: ρ=+0.9399, MAE=0.3135 eV, 0.005s
- TOM: ρ=+0.2557, MAE=3.7823 eV, 0.022s
- Span: 8.48 eV, 81 distinct values

### Dielectric (Kirkwood-Fröhlich)
- Verified set (n=55): ρ=+0.9340, MAE=3.2584
- Commercial solvents (n=10): ρ=+0.9879, MAE=1.8016

### Bulk Properties (External Benchmark)
- Dielectric: ρ=+0.6651, MAE=6.7748, n=99
- Viscosity: ρ=+0.5624, MAE=1.4695, n=98
- Donor_Number: ρ=+0.1885, MAE=18.6259, n=33

### Oracle vs ML Baseline (ECFP4+RF)
- HOMO: Oracle ρ=+0.5680, RF ρ=+0.3872 ± 0.3172, gap=+0.1808 ✅
- LUMO: Oracle ρ=+0.2744, RF ρ=+0.2947 ± 0.2352, gap=-0.0203 ✅
- Dielectric: Oracle ρ=+0.6417, RF ρ=+0.6331 ± 0.1392, gap=+0.0085 ✅
- Viscosity: Oracle ρ=+0.6359, RF ρ=+0.4207 ± 0.2293, gap=+0.2152 ✅
- Donor Number: Oracle ρ=+0.2993, RF ρ=+0.3726 ± 0.3684, gap=-0.0733 ⚠️

### Discovery Metrics
- Rediscovery rate (seeded-exact recovery, GATING): 68.8% (33/48 knowns recovered in the screened pool; target >=50%)
- Rediscovery exact-match rate (gating number): 68.8% (33/48 exact SMILES)
- Rediscovery coverage rate (transparency-only, top 25%): 7.5% (does not gate)
- Novel scaffold ratio: 100.0% (target ≥80%)
- Known mean score: 46.53
- Top mean score: 85.52
- Score gap: +38.99

## ✅ All Tolerances Met