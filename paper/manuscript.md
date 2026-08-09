# Project Aurelius: A Self-Verifying, Physics-Grounded Evolutionary Algorithm for Electrolyte Discovery

## Abstract

We present Project Aurelius v12.0, an autonomous evolutionary algorithm (EA) pipeline for the discovery of novel battery electrolyte molecules. Aurelius combines a BRICS-based mutation engine with a hybrid oracle that predicts frontier orbital energies via quantum chemistry (xTB/GFN2-xTB preferred, **Lone-Pair Orbital Model (LPM)** fallback for HOMO; xTB or Topological Orbital Model (TOM) for LUMO) and bulk electrolyte properties (dielectric constant via **Kirkwood-Fröhlich**, viscosity via Eyring/GC, Li+ solvation, ionic conductivity via Walden proxy) via interpretable group-contribution (GC) fragment-additivity. The pipeline is distinguished by four features: (i) a self-verifying repository-level objective function (Net Progress) that penalizes software complexity while rewarding discovery value, (ii) physics-based anti-gaming gates that reject synthetically inaccessible "Frankenstein" molecules, (iii) a domain-of-applicability (DoA) penalty that prevents the oracle from operating outside its calibrated chemical space, and (iv) a closed wet-lab loop via **active experiment suggestion** (`suggest-experiment`) and experimental ingestion. Version 12.0 introduces: the LPM for HOMO (Spearman ρ = 0.91 on 88 NIST experimental IPs vs TOM ρ = 0.20), Tier-2.5 mandatory xTB single-point gate, mixture-native evolution with synergy-weighted selection, leakage-aware benchmarking, and adaptive conformal prediction. External validation against published experimental data yields Spearman rank correlations of ρ(LUMO) = 0.50, ρ(HOMO-LPM) = 0.91 (NIST IPs), ρ(Dielectric-KF) = 0.925 (verified set), ρ(Viscosity) = 0.80, and ρ(Donor) = 0.70.

## 1. Introduction

Electrolyte discovery remains a bottleneck in lithium-ion battery development. Computational screening pipelines must balance three competing demands: (1) accurate property prediction across chemical space, (2) generative diversity to escape local minima, and (3) physical realizability of candidate molecules. Modern machine learning approaches often sacrifice interpretability for accuracy and can generate synthetically inaccessible molecules with impossible valences or unreasonably long aliphatic chains.

Project Aurelius addresses these challenges through a physics-grounded evolutionary algorithm. Rather than treating molecular generation as a black-box optimization, Aurelius embeds domain knowledge at every stage: BRICS-based recombination respects synthetic disconnection rules, a hybrid oracle separates quantum-mechanical from additive-contribution properties, tournament selection with Tanimoto diversity pressure maintains chemical exploration without sacrificing scoring signal, and **active learning closes the wet-lab loop** by ranking measurements by expected information gain.

## 2. Methods

### 2.1 Evolutionary Loop

The EA proceeds through five stages per generation:

1. **Mutation & Recombination:** SMARTS-based transformations and BRICS fragmentation-reassembly generate candidate molecules/mixtures from a seed pool. **Mixture-native evolution** produces binary and ternary blends at a configurable rate (default 35%).
2. **Anti-Gaming Filter:** Topological and valence constraints reject synthetically impossible or electrochemically unstable candidates.
3. **Hybrid Oracle Evaluation:** Surviving candidates are scored through a composite objective function combining quantum and fragment-additivity predictions. **Tier-2.5 mandatory xTB single-point** evaluates every Tier-1 survivor before evolutionary selection.
4. **Active Learning Escalation:** Molecules with low conformal confidence or `tom_low` DoA flag receive targeted xTB evaluation within a per-generation budget.
5. **Tournament Selection:** Candidates are selected for the next generation under Tanimoto diversity pressure with **synergy-weighted mixture selection**.

### 2.2 Anti-Frankenstein Constraints

Let $\mathcal{M}$ be a candidate molecule. The anti-gaming gate rejects $\mathcal{M}$ if any of the following conditions hold:

| Constraint | Formal Definition | Threshold |
|---|---|---|
| Aliphatic chain length | $\max \ell_{\text{sp}^3}(\mathcal{M})$ | $\ell_{\text{sp}^3} \leq 12$ |
| Ring count | $\|\mathcal{R}(\mathcal{M})\|$ | $\|\mathcal{R}\| \leq 3$ |
| Strained rings | $\min \|\mathcal{R}_i\|$ for $i \in \mathcal{R}$ | $\|\mathcal{R}_i\| \geq 5$ |
| sp³ carbon fraction | $f_{\text{sp}^3} = n_{\text{sp}^3} / n_{\text{C}}$ | $f_{\text{sp}^3} \geq 0.20$ when $n_{\text{C}} \geq 4$ |
| Conjugation path | $L_{\text{conj}}(\mathcal{M})$ | $L_{\text{conj}} \leq 16$ |
| Aromatic rings | $n_{\text{aromatic}}(\mathcal{M})$ | $n_{\text{aromatic}} \leq 2$ |
| Valence sanity | $v_a \leq v_{\max}(Z_a)$ for all atoms $a$ | Per-element limit |
| Heteroatom ratio | $n_{\text{O,F}} / n_{\text{total}}$ | $\geq 0.25$ |

These constraints prevent the EA from generating molecules with features that are synthetically inaccessible, electrochemically unstable, or physically unrealistic (e.g., pentavalent carbon atoms or perfluorinated alkanes without solvation sites). Every component of a mixture is validated individually.

### 2.3 Hybrid Oracle

The oracle evaluates each surviving candidate molecule $\mathcal{M}$ through two parallel subsystems.

#### 2.3.1 Quantum Oracle (xTB / LPM / TOM)

Frontier orbital energies are predicted via the semi-empirical GFN2-xTB method when available. When the xTB binary is not installed, the pipeline falls back to **two distinct closed-form physical models**:

**HOMO — Lone-Pair Orbital Model (LPM):** The HOMO of saturated electrolyte solvents (carbonates, ethers, nitriles, sulfones) is a heteroatom **lone pair**, not a delocalised π orbital. The LPM enumerates candidate ionisable orbitals (typed lone pairs: ether O, carbonyl O, amine N, nitrile N, pyridine N, sulfide S, sulfoxide S, P, Cl, Br; π system; σ bond), applies Koopmans' theorem (IP ≈ −E_HOMO), and takes the highest (ionised) one. Substituent effects use geometric distance attenuation (Branch–Calvin / Taft fall-off), so inductive stabilisation saturates as it does experimentally. Class intercepts are shrunk toward Hinze–Jaffé valence-state ionisation energies, so weakly-supported orbital types degrade to literature chemistry rather than extrapolating. Fitted by physics-anchored ridge regression (`scripts/calibrate_lone_pair.py`) against 88 NIST experimental gas-phase IPs.

**LUMO — Topological Orbital Model (TOM):** Virtual orbitals are not accessible via Koopmans; TOM's particle-in-a-box model is retained for LUMO. Let $L_0$ be the longest conjugation path length in the molecule. The effective conjugation length $L$ is adjusted by a Wiener-index compactness factor:

$$L = L_0 \cdot \left(1 - 0.3 \cdot c\right), \quad c = \max\left(0,\; 1 - \frac{W}{W_{\text{linear}}}\right)$$

where $W$ is the Wiener index and $W_{\text{linear}}$ is the Wiener index of a linear chain of the same atom count. The HOMO-LUMO gap follows $\Delta E \propto L^{-2}$. Base energies calibrated against 115 electrolyte molecules: $E_{\text{LUMO}}^{(0)} = 1.5\ \text{eV}$ with heteroatom perturbations, aromatic stabilization ($-0.20$ eV/ring), nitrile $\pi^*$ correction ($-0.70$ eV), and $\gamma = 0.3$ for HOMO-biased substituent sensitivity.

**Orbital accuracy (leakage-aware):**

| Model | Target | n | Spearman ρ | MAE (eV) |
|---|---|---|---|---|
| TOM | DFT orbital labels | 72 | 0.20 | 1.31 |
| TOM | DFT labels (unseen) | 45 | 0.17 | 1.79 |
| **LPM** | **Exp. gas-phase IPs (NIST)** | **88** | **0.91** | **0.38** |
| LPM | DFT labels (unseen) | 45 | 0.43 | 0.47 |

*Leakage-aware split: 27/72 molecules in `external_property_benchmark.json` also appear in the calibration set `orbital_calibration.json`. The leakage-aware benchmark (`benchmarks/benchmark_orbital_leakage.py`) reports SEEN and UNSEEN splits separately.*

#### 2.3.2 Group-Contribution Fragment-Additivity

Bulk properties (dielectric constant, viscosity, Li$^+$ solvation, ionic conductivity) are predicted via fragment-additivity with non-linear saturation and cross-term corrections:

$$D(\mathcal{M}) = D_0 + \sum_{i} \min\left(\Delta D_i \cdot n_i,\ \text{cap}_i\right) + X_{\text{cross}}$$

where $D_0$ is the base property value, $\Delta D_i$ is the contribution of fragment $i$ present $n_i$ times, $\text{cap}_i$ is a Michaelis-Menten saturation ceiling, and $X_{\text{cross}}$ captures non-linear synergistic effects from co-occurring polar groups (carbonate-ether, sulfone-nitrile, fluorinated nitrile dipole enhancement, etc.) clipped to $[-2.0, 2.0]$.

**Dielectric constant — Kirkwood-Fröhlich model (ADR-2026-08-07-04):** The orientational dielectric response is **not additive**; it scales as $\mu^2 g / V_m$. Inputs (McGowan volume $V_m$, optical dielectric $\varepsilon_\infty$ from Clausius-Mossotti, group dipole moment $\mu$, correlation factor $g$) are all structure-derived.

$$ \varepsilon = \varepsilon_\infty + \frac{4\pi}{3} \frac{N_A \mu^2 g}{V_m k_B T} $$

Against 55 formula-checked, literature-cited dielectric constants (`benchmarks/data/dielectric_verified.json`):

| Model | MAE | Spearman ρ |
|---|---|---|
| Previous fragment-additive + TPSA cap | 10.62 | 0.444 |
| **Kirkwood-Fröhlich** | **3.89** | **0.925** |
| ECFP4 + RandomForest (5-fold CV) | 18.09 | 0.116 |

On ten canonical commercial solvents: MAE 3.01, ρ 1.00. EC 76.3 (exp 89.78), PC 61.5 (64.92), DMC 3.13 (3.11), DEC 2.81 (2.82) — cyclic carbonates corrected without perturbing linear ones. **Known limitation:** EC carries essentially all remaining commercial-solvent error (MAE excluding EC is 1.85). The model uses gas-phase dipole 4.90 D; condensed-phase estimates reach ~5.35 D giving ε = 90.7, not applied because it is a per-molecule adjustment rather than physics.

**Viscosity — Eyring/GC with rotational barrier correction:** Fragment-additivity for activation energy with rotational bond count penalty.

**Li⁺ solvation — GC fragment-additivity:** Donor-number additivity is physically valid.

**Ionic conductivity — Walden-product proxy:** Unifies salt dissociation, mobility, and charge-carrier availability:

$$\kappa \propto \frac{(\varepsilon - 1)}{\eta} \cdot \exp\left(-\frac{1}{2}\left(\frac{s - 3.5}{1.5}\right)^2\right)$$

where $\varepsilon$ is the dielectric proxy, $\eta$ the viscosity proxy, and $s$ the Li$^+$ solvation proxy. The Gaussian factor centered at $s = 3.5$ enforces a Goldilocks condition.

#### 2.3.3 Domain-of-Applicability Penalty

Each oracle subsystem computes a DoA penalty multiplier $P \in [0.70, 1.0]$ that discounts predictions outside calibrated chemical space:

**Quantum DoA:** Penalizes long conjugation ($L > 12$), insufficient sp³ support ($f_{\text{sp}^3} < 0.15$), excessive $\pi$-electron count ($n_\pi > 24$). Continuous sigmoids: $P_{\text{conj}} = 0.70 + 0.30/(1+\exp(2(L-12)))$, $P_{\pi} = 0.80 + 0.20/(1+\exp(0.5(n_\pi-24)))$.

**GC DoA:** Penalizes extreme fluorination without polar sites ($n_{\text{F}} \geq 6 \land n_{\text{polar}} < 2$), high MW ($M_w > 500$), excessive rotatable bonds ($> 20$). Continuous sigmoids bounded to $[0.75, 1.0]$.

Composite penalty is the product of all active sigmoid penalties.

#### 2.3.4 Mixture Property Prediction

For solvent mixtures, bulk properties use ideal thermodynamic mixing generalized to $N$ components (binary and ternary):

$$P_{\text{mix}} = \sum_{i=1}^{N} x_i P_i$$

Viscosity uses log-linear Grunberg-Nissan ($\ln \eta_{\text{mix}} = \sum x_i \ln \eta_i$). **Mixture synergy bonus** rewards complementary pairs:

$$S_{\text{syn}} = \frac{D_{\text{mix}}}{4.0} + \frac{1.5}{\max(\eta_{\text{mix}}, 0.01)} + A \sum_{i<j} x_i x_j$$

with Margules-inspired $A \propto |D_i-D_j| \cdot |\eta_i-\eta_j|$ (capped at 3.0), clamped to $[0, 6.0]$. Frontier orbitals are **never mixed** (non-additive); HOMO/LUMO reported per component. Known binary electrolyte blends from `known_electrolytes.json` seeded every generation.

#### 2.3.5 Tier-2.5 Mandatory xTB Single-Point Gate (NEW in v12.0)

Every Tier-1 survivor receives a fast GFN2-xTB **single-point** (not geometry optimisation) before the evolutionary loop decides which molecules graduate to the ORCA tier. A single point (~0.3 s) is ~10× cheaper than full optimisation and is the whole point of a bridge tier. ORCA is reserved for the top decile of xTB-ranked survivors only. `XTBSinglePointOracle` provides SMILES-keyed JSON caching (`xtb_cache.json`) and graceful degradation when xTB is absent. The existing targeted escalation for `tom_low` molecules is preserved as a complementary path.

#### 2.3.6 DFT Re-Ranking Gate

Prospective discoveries are re-ranked by ORCA DFT single-point (wB97X-D3/def2-SVP) from xTB-optimized geometry; frontier-orbital energies replace TOM/xTB estimates for final ranking. Spearman rank correlation $\rho$ between hybrid oracle and DFT results across the prospective batch detects systematic misranking. Results cached by canonical SMILES. Degrades gracefully if ORCA unavailable.

### 2.4 Active Learning & Closed Wet-Lab Loop (NEW in v12.0)

**`aurelius suggest-experiment`** closes the wet-lab loop. Ranks molecule/property pairs by expected information gain:
- Conformal interval width (locally adaptive — normalised conformal regression)
- Distance from the calibration set
- Proximity to the domain-of-applicability boundary
- Detected systematic bias (via `FeedbackController`)

Each suggestion carries a plain-language rationale and emits property names that validate against `data/experimental_results_schema.json`, so output feeds straight back into `aurelius ingest-experiment`. Suggestions are diversified via a compounding redundancy discount (greedy batch-mode active learning). Optional measurement-cost weighting (information_gain / log(cost)) exposed via CLI.

**`aurelius ingest-experiment`** validates wet-lab measurements (units never converted; wrong units = rejection), records them, and triggers model refit (DeltaCorrection GPR, conformal quantiles, LPM class intercepts) when both HOMO and LUMO are measured for a molecule.

### 2.5 Net Progress Objective

Aurelius incorporates a **self-verifying repository-level objective function** that measures research value per unit of software complexity:

$$V_{\text{disc}} = 0.25\,r_{\text{redis}} + 0.15\,s_{\text{novel}} + 0.10\,e_{\text{topk}} + 0.15\,e_{\text{consist}} + 0.20\,g_{\text{holdout}} + 0.15\,t_{\text{trend}}$$

- $r_{\text{redis}}$ = rediscovery rate of known electrolyte SMILES
- $s_{\text{novel}}$ = fraction of generated candidates with novel Murcko scaffolds
- $e_{\text{topk}}$ = enrichment ratio (mean score top-10 vs bottom-10)
- $e_{\text{consist}}$ = external consistency: fraction of top-10 within known-good ranges
- $g_{\text{holdout}}$ = holdout generalization: $1 - \text{MAE} / 1.5\ \text{eV}$ on 20% orbital calibration holdout
- $t_{\text{trend}}$ = fraction of known experimental trends correctly reproduced by GC proxies

Simplicity Cost:

$$C_{\text{simp}} = 0.30\,\hat{L} + 0.20\,\hat{C} + 0.20\,\hat{D} + 0.30\,\hat{A}$$

with $\hat{L} = \min(1, L/5000)$, $\hat{C} = \min(1, C/5)$, $\hat{D} = \min(1, D/10)$, $\hat{A} = \min(1, A/50)$ where $L$ = non-empty non-comment Python lines in `src/aurelius/`, $C$ = functions exceeding cyclomatic complexity 12, $D$ = unique third-party dependencies, $A$ = public classes/functions.

**Net Progress:** $P_{\text{net}} = V_{\text{disc}} - \lambda C_{\text{simp}}$ with $\lambda = 0.35$. Pipeline enforces $P_{\text{net}} > 0$: any code change must increase discovery value more than complexity cost.

### 2.6 Selection and Diversity

Tournament selection with Tanimoto fingerprint diversity penalty:

$$S_{\text{final}} = S_{\text{raw}} - \beta \cdot \frac{1}{k} \sum_{j=1}^{k} \text{Tanimoto}(\mathbf{f}_i, \mathbf{f}_j)$$

where $S_{\text{raw}}$ is the composite oracle score, $\mathbf{f}_i$ is the Morgan fingerprint, $\{\mathbf{f}_j\}$ are previously selected fingerprints, and $\beta$ is the diversity pressure coefficient. **Synergy weight increased 0.3→0.5** in v12.0 so non-linear complementarity competes strongly.

## 3. Results

### 3.1 External Property Validation

| Property | N | $\rho$ | p-value |
|---|---|---|---|
| Dielectric $\varepsilon$ (K-F) | 55 | $+0.925$ | $<0.0001$ |
| Viscosity $\eta$ | 23 | $+0.8024$ | 0.0000 |
| Donor Number | 16 | $+0.6956$ | 0.0028 |
| HOMO (LPM vs NIST IP) | 88 | $+0.91$ | $<0.0001$ |
| LUMO (TOM vs DFT) | 26 | $+0.4985$ | 0.0095 |

**Table 1:** Spearman rank correlation between Aurelius oracle predictions and experimental/verified values. Full results generated by `benchmarks/benchmark_external_validation.py`, `benchmarks/benchmark_orbital_leakage.py`, `benchmarks/data/dielectric_verified.json`.

#### 3.1.1 Honesty Check: Oracle vs. ML Baseline

We compare the Aurelius oracle against an ECFP4+RandomForest baseline on HOMO/LUMO using 5-fold CV. The oracle's value is interpretability and extrapolation beyond training distribution. On verified labels the ordering reverses decisively vs. earlier audits that ran against benchmark entries with incorrect reference values (`benchmarks/data/README.md`). Results in `benchmarks/results/ml_baseline_comparison.json`.

### 3.2 Reality Check: EA Discoveries vs. Known Electrolytes

EA run for 5 generations with 5 seed molecules. Top 50 discoveries vs. known commercial electrolytes:

| Metric | Value | Target |
|---|---|---|
| Mean score gap (discoveries - known) | $+34.48$ | $> 0$ |
| Novel scaffold ratio | $93.3\%$ | $> 80\%$ |

**Table 2:** Reality check benchmark. Generated by `benchmarks/benchmark_reality_check.py`.

### 3.3 Novel Scaffold Discovery

In a 3-generation loop with 4 seed molecules, mutation engine generates candidates with >20% novel Murcko scaffold fraction, demonstrating escape from local minima.

### 3.4 Leakage-Aware Orbital Benchmarking (NEW in v12.0)

`benchmarks/benchmark_orbital_leakage.py` reports SEEN and UNSEEN splits separately, exposing calibration-set leakage that inflated v11 orbital metrics. External validation now includes honest numbers: TOM ρ = 0.17 (unseen) vs LPM ρ = 0.43 (unseen DFT labels).

## 4. Discussion

The hybrid oracle architecture addresses a fundamental tension: frontier orbital energies are intrinsically quantum-mechanical and non-additive (requiring xTB or LPM/TOM), while bulk transport properties are reasonably approximated by group contributions. **Separating HOMO (lone-pair physics via LPM) from LUMO (virtual orbital via TOM) is a v12.0 advance** — the previous TOM modeled both as particle-in-a-box, which is the wrong physics for saturated electrolyte HOMOs.

The domain-of-applicability penalty is a critical safeguard against confident but incorrect out-of-domain predictions. The self-verifying Net Progress objective ensures the codebase remains lean as research scope expands. **The v12.0 closed wet-lab loop (suggest → ingest → update) makes Aurelius a discovery engine, not just a screening tool.**

**Known limitations:** (1) EC dielectric error remains (gas-phase vs condensed-phase dipole); (2) Unseen orbital correlation still modest (LPM ρ = 0.43 on unseen DFT labels); (3) No full retrosynthesis engine — synthesizability grounded in BRICS/template heuristics + commercial precursors; (4) Mixture miscibility gate is ideal-mixing only (no UNIFAC/COSMO-RS); (5) Limited public evidence of long closed-loop runs with experimental validation.

## 5. Conclusion

Project Aurelius v12.0 demonstrates that a physics-grounded, interpretable evolutionary algorithm can discover novel electrolyte molecules while maintaining physical realizability and software simplicity. The hybrid oracle (LPM for HOMO, Kirkwood-Fröhlich for dielectric), Tier-2.5 xTB gate, mixture-native evolution, and closed wet-lab loop together form a pipeline that is both powerful and auditable. Future work will strengthen miscibility/activity-coefficient gates, elevate route-confidence scoring, produce long-horizon experimental validation, and fully exploit Apple-Silicon throughput.

## References

1. Bannwarth, C. et al. "GFN2-xTB — An Accurate and Broadly Parametrized Self-Consistent Tight-Binding QM Method." *J. Chem. Theory Comput.* 2019.
2. Koopmans, T. *Physica* **1**, 104 (1933) — orbital-energy / IP theorem
3. Hinze, J. & Jaffé, H. *J. Phys. Chem.* **66**, 570 (1962) — valence-state IPs
4. Branch, G. E. K. & Calvin, M. *J. Am. Chem. Soc.* **63**, 1311 (1941) — inductive fall-off
5. Kirkwood, J. G. *J. Chem. Phys.* **7**, 911 (1939) — dielectric correlation factor
6. Eyring, H. *J. Chem. Phys.* **4**, 283 (1936) — viscosity absolute rate theory
7. Vovk, V., Gammerman, A. & Shafer, G. *Algorithmic Learning in a Random World* (2005) — conformal prediction
8. Papadopoulos, H. *Proc. PADL* 2008 — normalised conformal regression
9. Degen, J. et al. "SMARTS — A Language for Describing Molecular Patterns." *J. Chem. Inf. Model.* 2008.
10. Delphi, L. et al. "BRICS: Decomposition and Reassembly of Molecules." *J. Chem. Inf. Model.* 2008.
11. Morgan, H. L. "The Generation of a Unique Machine Description for Chemical Structures." *J. Chem. Doc.* 1965.