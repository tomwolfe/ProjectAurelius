# Unified Benchmark Report

**Status**: ❌ FAIL
**Generated**: 2026-08-14 15:58:10

## Tolerances
### Orbital
- `lpm_nist_rho_min`: 0.9
- `lpm_nist_mae_max`: 0.45
- `lpm_unseen_dft_rho_min`: 0.35
- `tom_leakage_gap_max`: 0.45

### Dielectric
- `kf_mae_max`: 5.0
- `kf_rho_min`: 0.9
- `commercial_mae_max`: 3.0
- `commercial_rho_min`: 0.95

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

## Results
### Orbital (Leakage-Aware)
#### All (n=72)
- TOM: ρ=+0.2044, MAE=1.3132 eV
- LPM: ρ=+0.5561, MAE=0.3855 eV

#### Seen (n=27)
- TOM: ρ=+0.5422, MAE=0.5163 eV
- LPM: ρ=+0.8487, MAE=0.2483 eV

#### Unseen (n=45)
- TOM: ρ=+0.1700, MAE=1.7913 eV
- LPM: ρ=+0.4319, MAE=0.4678 eV

#### Experimental IPs (NIST, no leakage)
- LPM: ρ=+0.9399, MAE=0.3135 eV, 0.007s
- TOM: ρ=+0.2556, MAE=3.7847 eV, 0.037s
- Span: 8.48 eV, 81 distinct values

### Dielectric (Kirkwood-Fröhlich)
- Verified set (n=55): ρ=+0.9340, MAE=3.2584
- Commercial solvents (n=10): ρ=+0.9879, MAE=1.8016

### Bulk Properties (External Benchmark)
- Dielectric: ρ=+0.6651, MAE=6.7748, n=99
- Viscosity: ρ=+0.5513, MAE=1.4139, n=98
- Donor_Number: ρ=+0.1885, MAE=18.6259, n=33

### Oracle vs ML Baseline (ECFP4+RF)
- HOMO: Oracle ρ=+0.1096, RF ρ=+0.3872 ± 0.3172, gap=-0.2776 ⚠️
- LUMO: Oracle ρ=+0.2911, RF ρ=+0.2947 ± 0.2352, gap=-0.0035 ✅
- Dielectric: Oracle ρ=+0.6417, RF ρ=+0.6331 ± 0.1392, gap=+0.0085 ✅
- Viscosity: Oracle ρ=+0.6330, RF ρ=+0.4207 ± 0.2293, gap=+0.2123 ✅
- Donor Number: Oracle ρ=+0.2993, RF ρ=+0.3726 ± 0.3684, gap=-0.0733 ⚠️

### Discovery Metrics
- Rediscovery rate (seeded-exact recovery): 68.8% (33/48 knowns recovered in the screened pool; target ≥50%)
- Rediscovery coverage rate (Gap 4 transparency, top 25%): 18.8%
- Novel scaffold ratio: 60.0% (target ≥80%)
- Known mean score: 72.27
- Top mean score: 100.00
- Score gap: +27.72

## ❌ CI Failures
- Novel scaffold ratio=0.600 < 0.8