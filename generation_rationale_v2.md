# Generation Rationale v2 - Homogeneity-Targeted SEI Candidates

## Strategy Overview

The baseline screen of 8 standard candidates showed SEI homogeneity
scores of ~12.2/100 (raw ~0.12), indicating that solvent decomposition
overwhelmingly dominates the kMC simulation. This produces a brittle,
heterogeneous SEI layer.

This generation targets **multi-pathway decomposition** by designing
molecules with structural features that promote balanced reaction
rates across all three kMC pathways:

1. **Solvent Decomposition** (EC/DMC reduction at anode)
2. **Salt Reduction** (PF6- decomposition)
3. **Polymerization** (organic SEI formation)

## Scaffold Categories

### COC(=O)OCC=C

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 116.0 g/mol
- **Max Tanimoto Similarity:** 0.444

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### COC(=O)OC(F)C=C

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 134.0 g/mol
- **Max Tanimoto Similarity:** 0.357

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### COC(=O)OC(F)=CF

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 138.0 g/mol
- **Max Tanimoto Similarity:** 0.370

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### COC(=O)C(F)=C(F)F

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 140.0 g/mol
- **Max Tanimoto Similarity:** 0.333

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### FCC(F)(F)S(=O)(=O)OC=C

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 190.0 g/mol
- **Max Tanimoto Similarity:** 0.167

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### COC(=O)OC(F)=C(F)C(F)(F)F

- **Category:** Dual-Functional Additives (F + C=C)
- **Molecular Weight:** 206.0 g/mol
- **Max Tanimoto Similarity:** 0.444

**Multi-pathway rationale:** This molecule contains both
a fluorinated group (lowering the effective barrier for salt
reduction by stabilizing F- intermediates) AND a C=C double bond
(providing a low-barrier pathway for polymerization). The dual
functional design ensures that salt reduction and polymerization
rates increase relative to pure solvent decomposition, pushing
the reaction distribution closer to the ideal 1/3:1/3:1/3 split.

### B1OB(OB(OCC(F)F)(OCC(F)F))O1

- **Category:** Cyclic Borate Esters (B-O centers)
- **Molecular Weight:** 244.1 g/mol
- **Max Tanimoto Similarity:** 0.100

**Multi-pathway rationale:** Cyclic borate esters decompose via
B-O bond cleavage (lower Ea than C-B direct bonds), producing
fluorinated boron species that interact with PF6- salt anions.
The fluorinated alkyl chains provide additional pathways for
salt reduction. The ring-opening mechanism creates reactive
intermediates that can participate in polymerization, increasing
the relative rate of the polymerization pathway.

### COC(=O)OB1OC(C(F)F)(OCC(F)F)O1

- **Category:** Cyclic Borate Esters (B-O centers)
- **Molecular Weight:** 262.0 g/mol
- **Max Tanimoto Similarity:** 0.300

**Multi-pathway rationale:** Cyclic borate esters decompose via
B-O bond cleavage (lower Ea than C-B direct bonds), producing
fluorinated boron species that interact with PF6- salt anions.
The fluorinated alkyl chains provide additional pathways for
salt reduction. The ring-opening mechanism creates reactive
intermediates that can participate in polymerization, increasing
the relative rate of the polymerization pathway.

### N#CCS(=O)(=O)C

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 119.0 g/mol
- **Max Tanimoto Similarity:** 0.312

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### N#CCS(=O)(=O)CC#N

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 144.0 g/mol
- **Max Tanimoto Similarity:** 0.312

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### N#CC(F)S(=O)(=O)C

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 137.0 g/mol
- **Max Tanimoto Similarity:** 0.278

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### N#CCOC(=O)OC(F)S(=O)(=O)C

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 211.0 g/mol
- **Max Tanimoto Similarity:** 0.263

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### N#CCS(=O)(=O)C(F)(F)F

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 173.0 g/mol
- **Max Tanimoto Similarity:** 0.238

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### N#CCOCCS(=O)(=O)CC#N

- **Category:** Sulfone-Nitrile Hybrids
- **Molecular Weight:** 188.0 g/mol
- **Max Tanimoto Similarity:** 0.192

**Multi-pathway rationale:** The sulfone group provides high
voltage stability (resisting early decomposition), while the
nitrile group provides strong adsorption to the anode surface.
This dual nature means the molecule contributes to both solvent
decomposition (via the sulfone framework) AND salt interaction
(via nitrile-anion complexation). Fluorinated variants further
enhance salt reduction events through F-stabilized intermediates.

### COC(=O)OCC(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 140.0 g/mol
- **Max Tanimoto Similarity:** 0.520

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### COC(=O)OC(C)(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 140.0 g/mol
- **Max Tanimoto Similarity:** 0.667

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### CCOC(=O)OCC(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 154.0 g/mol
- **Max Tanimoto Similarity:** 0.429

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### CCOC(=O)OC(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 140.0 g/mol
- **Max Tanimoto Similarity:** 0.444

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### COC(=O)OC(F)(F)C(F)(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 194.0 g/mol
- **Max Tanimoto Similarity:** 0.636

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### COC(=O)OCC(F)(F)C(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 190.0 g/mol
- **Max Tanimoto Similarity:** 0.654

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### COC(=O)OC=C(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 138.0 g/mol
- **Max Tanimoto Similarity:** 0.385

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

### CCOC(=O)OCC(F)F

- **Category:** Asymmetric Fluoro-Carbonates
- **Molecular Weight:** 154.0 g/mol
- **Max Tanimoto Similarity:** 0.429

**Multi-pathway rationale:** Breaking symmetry in carbonates
alters the decomposition kinetics compared to symmetric analogs.
Fluorinated asymmetric carbonates have lower activation barriers
for salt reduction (due to F stabilization of transition states)
while the carbonate backbone still supports solvent decomposition.
The asymmetry also creates multiple distinct decomposition
pathways, increasing the probability of salt and polymerization
events relative to the baseline symmetric carbonates.

## Novelty Validation

All 22 candidates passed Tanimoto similarity < 0.75
against the 8 baseline discovery candidates using ECFP4 fingerprints
(radius=2, 2048 bits), ensuring structural diversity.

## Filters Applied

- Molecular Weight < 350 g/mol
- RDKit sanitization (valid valence, aromaticity)
- Tanimoto similarity < 0.75 vs. baseline candidates (ECFP4)

