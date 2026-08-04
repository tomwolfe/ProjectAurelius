# Aurelius API Reference — v11.0

Reference for the **standalone Oracle API** introduced in v11.0. It exposes the
hybrid quantum/GC oracle and the N-component mixture predictor as a plain Python
library surface, independent of the evolutionary loop.

```python
from aurelius.oracle_api import (
    predict_properties,
    predict_mixture,
    get_domain_applicability,
    reset_pipeline,
)
```

---

## `predict_properties(smiles: str) -> dict[str, Any]`

Full hybrid property prediction for a single molecule, routed through the same
pipeline code path as `aurelius screen`. Each call is self-contained (the
oracle pipeline is constructed lazily and cached).

| Key | Type | Meaning |
|---|---|---|
| `homo_eV` | `float` | Predicted HOMO energy (eV), negative. |
| `lumo_eV` | `float` | Predicted LUMO energy (eV). |
| `band_gap_eV` | `float` | `lumo_eV - homo_eV`. |
| `dielectric_proxy` | `float` | GC fragment-additivity dielectric proxy. |
| `viscosity_proxy` | `float` | GC viscosity proxy (cP). |
| `li_solvation_proxy` | `float` | Li+ solvation donor-number proxy. |
| `ionic_conductivity_proxy` | `float` | Walden-product conductivity proxy. |
| `total_score` | `float` | Composite Aurelius score. |
| `domain_penalty` | `float` | DoA multiplier in `[0.70, 1.0]`. |
| `domain_reason` | `str` | Human-readable DoA explanation. |
| `valid` | `bool` | `False` if the SMILES is invalid or fails the electrolyte viability filter. |

Example:

```python
>>> predict_properties("C1COC(=O)O1")["dielectric_proxy"]
5.74
```

## `predict_mixture(components: list[str], fractions: list[float]) -> dict[str, Any]`

N-component mixture prediction (binary **or** ternary — both `["A", "B"]`
and `["A", "B", "C"]` inputs are supported).

**Anti-gaming guarantee:** every component is validated individually through
the full single-molecule pipeline (electrolyte viability, anti-gaming gates,
DoA). A mixture can never mask an invalid component.

Args:
- `components`: canonical SMILES strings, length ≥ 2.
- `fractions`: mole fractions summing to `1.0` (±1e-6), same length as
  `components`.

Raises `ValueError` on malformed input (fewer than 2 components, length
mismatch, or fractions that do not sum to 1.0).

Return keys extend `predict_properties` with:
- `mixture` — `True`
- `components` — `[{"smiles", "total_score", "dielectric_proxy", "viscosity_proxy", "li_solvation_proxy"}, ...]`
- `mixture_synergy_bonus` — `float` in `[0, 6.0]` from the Margules-inspired non-ideal term
- Mixing rules used per property: dielectric = mole-weighted mean;
  viscosity = log-linear (Grunberg-Nissan); Li+ solvation = additive; HOMO/LUMO
  reported as per-component values (frontier orbitals are non-additive and are
  never mixed).

```python
>>> predict_mixture(
...     ["C1COC(=O)O1", "CCOCCO", "CS(=O)(=O)C"],
...     [0.5, 0.3, 0.2],
... )["mixture_synergy_bonus"]
1.12
```

## `get_domain_applicability(smiles: str) -> tuple[float, str]`

Returns `(penalty_multiplier, reason)`. The multiplier is `1.0` when the
molecule is fully inside the calibrated domain and falls toward `0.70` as it
leaves it (conjugation length, π-electron count, fluorination without solvation
sites, molecular weight, flexibility).

## `reset_pipeline() -> None`

Drops the cached oracle pipeline. Use after re-calibration so the next
prediction rebuilds the pipeline from the updated calibration data.

---

## CLI: `aurelius predict <smiles>`

```bash
aurelius predict "C1COC(=O)O1"        # single molecule
aurelius predict "A|B|0.5"            # binary mixture (v10 format)
aurelius predict "A|B|C|0.5|0.3"      # ternary mixture (last fraction implied)
```

Accepts either a plain SMILES or a pipe-delimited mixture string in the same
format used by the mutation engine. Output is a JSON dictionary.

## Related modules

- `src/aurelius/oracle_api.py` — public API surface
- `src/aurelius/scoring/oracle/gc.py` — `predict_mixture_dielectric_n`,
  `predict_mixture_viscosity_n`, `predict_mixture_li_solvation_n`,
  `mixture_synergy_bonus_n`
- `src/aurelius/types.py` — `parse_mixture_smiles_n`, `format_mixture_smiles_n`
- `src/aurelius/scoring/oracle/dft_validator.py` — ORCA re-ranking gate
  (`DFTValidator`, `spearman_correlation`, `has_orca`)
