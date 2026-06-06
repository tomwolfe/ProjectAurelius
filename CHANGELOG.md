# Changelog

## [10.0.6] - 2026-06-06

### Changed
- Enforced Single Source of Truth: Added `benchmark_mixture_synergy.py` to `scripts/update_benchmark_docs.py` auto-generation pipeline.
- CI Enforcement: Added atomic git-diff check in `.github/workflows/ci.yml` to prevent silent documentation drift of `docs/benchmarks.md`.
- Refactored Net Progress normalization constants into `src/aurelius/constants.py` with explicit architectural justification to prevent future YAGNI violations (CI gaming).

## [10.0.5] - 2026-06-06

### Fixed
- Corrected future-dated typo in v10.0.0 changelog entry (2026-06-11 → 2026-06-01) to maintain chronological audit integrity.

### Changed
- Condensed historical ADR inline comments in `gc.py` and `quantum.py` to enforce KISS; scientific justification retained, verbose tuning history moved to CHANGELOG.md.
- Enforced strict Single Source of Truth by replacing README.md "Validation Metrics" section with a direct reference to `scripts/update_benchmark_docs.py`.

## [10.0.4] - 2026-06-06

### Fixed
- Enforced Single Source of Truth: Removed residual hardcoded benchmark metrics from `README.md`, `paper/manuscript.md`, and `docs/marketing/pitch.md`.

### Added
- `scripts/update_benchmark_docs.py`: Auto-generates `docs/benchmarks.md` from live `benchmarks.benchmark_external_validation` and `benchmarks.benchmark_reality_check` executions, eliminating manual copy-pasting and preventing future documentation drift.

## [10.0.3] - 2026-06-06

### Changed
- Enforced Single Source of Truth: Completely removed hardcoded validation metrics table from `README.md` to fulfill v10.0.1 changelog promise.
- Simplified `DiscoveryLoop`: Removed redundant `screened_smiles` set to strictly enforce `LoopState` as the sole source of truth (YAGNI/KISS).

### Added
- `benchmarks/benchmark_mixture_synergy.py`: Lightweight, pure-function validation script for the Margules-inspired binary mixture synergy bonus.

## [10.0.2] - 2026-06-06

### Changed
- Enforced KISS: Removed redundant `all_results` and `discoveries` lists from `DiscoveryLoop`; `LoopState` is now the sole source of truth for screening state.
- Documentation sync: Updated `README.md` and `paper/manuscript.md` with live benchmark metrics, removing hardcoded historical baselines to resolve v10.0.1 changelog contradiction.
- Updated `tests/test_loop.py` to reference `state._all_results` instead of removed `loop.all_results`.

## [10.0.1] - 2026-06-06

### Changed
- Consolidated `docs/marketing/` into a single `pitch.md` to reduce architectural surface area (Net Progress simplicity cost).

### Added
- Included `examples/mixtures.smi` and updated README Quick Start to demonstrate v10.0 binary mixture screening capabilities.

### Fixed
- Removed hardcoded benchmark metrics from README.md to enforce a Single Source of Truth via executable benchmark scripts.

## [10.0.0] - 2026-06-01

### Added
- Mixture CLI command, deterministic net progress, benchmark table in README
- TOM parameter tuning — Wiener-index compactness, aromatic stabilization, nitrile C≡N π* correction; LUMO Spearman ρ +0.0246 (ADR-2026-06-11)
- σ* LUMO correction for S/P=O groups and phosphate HOMO correction for non-conjugated molecules (ADR-2026-06-10)
- Peierls distortion damping for long conjugation paths (ADR-2026-06-06)
- Cyclic sulfone/sultone GC fragments, instability filters (ADR-2026-06-05f)
- Ester SMARTS disambiguation — C(=O)-C vs C(=O)-O (ADR-2026-06-05d)
- Cyclic carbonate GC fragment (+6.0 dielectric), TPSA coefficient 0.02→0.025, nitrile dielectric 5.5→7.5 (ADR-2026-06-05b)
- Margules-inspired non-ideal mixing term for binary mixture synergy (ADR-2026-06-05)
- Aromatic ring stabilization term to TOM (ADR-2026-06-02)
- Expanded orbital calibration from 14→34 molecules, recalibrated EW coefficients (ADR-2026-06-01)
- [-2.0, 2.0] clip to dielectric cross-term corrections (ADR-2026-06-01)
- Ionic conductivity Walden-product proxy (dielectric × solvation / viscosity)

### Changed
- Wiener compactness factor tuned from 0.28 to 0.30 (ADR-2026-06-11)
- LUMO EW scaling increased from 0.30 to 0.35 (ADR-2026-06-11)
- HOMO aromatic stabilization strengthened from -0.20 to -0.25 per ring (ADR-2026-06-11)
- Cyclic carbonate dielectric 6.0→8.0, linear carbonate dielectric 5.0→2.0, TPSA 0.025→0.030 (ADR-2026-06-05c)
- Aromatic nitrogen Li+ solvation 3.5→4.0 (ADR-2026-06-05c)
- GC fragment rebalance: nitrile 8.0→5.5, amide 5.0→6.0, sulfoxide 6.0→7.5; Li+ solvation: amide 1.2→2.5, glyme_chelating 1.8→0.6, sulfoxide 2.5→3.5 (ADR-2026-06-05)
- Carbonate Li+ solvation 1.5→1.2, nitrile Li+ solvation 1.2→0.8
- EW coefficient -0.25→-0.32, LUMO EW scaling 0.7→0.3 (ADR-2026-06-02)

### Fixed
- Skip mypy follow_imports for rdkit.* to avoid rdkit-stubs syntax error
- Replace direct `Chem.MolFromSmiles` calls with `MoleculeContext` for single-point parsing
- MyPy type error in quantum.py: wrap compactness calculation in `int()`
