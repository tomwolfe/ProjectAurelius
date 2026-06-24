"""OrganicElectronicsPack — Group-contribution model for organic electronics.

Predicts hole mobility proxy and electron affinity proxy using
fragment-additivity with the same Michaelis-Menten saturation framework
as the electrolyte pack, but with fragments and cross-terms relevant to
organic semiconductors (conjugated backbones, electron-donating/acceptor
groups, pi-stacking motifs).
"""

from __future__ import annotations

import math

from rdkit import Chem

from aurelius.scoring.oracle.gc import BasePropertyModel, _GC_SATURATION_K
from aurelius.types import MoleculeContext

# Fragment definitions: (SMARTS, name, hole_mobility_contrib, electron_affinity_contrib)
_OE_FRAGMENTS: list[tuple[Chem.Mol, str, float, float]] = [
    (Chem.MolFromSmarts("[c]"),                 "aromatic_carbon",      0.3,  0.3),
    (Chem.MolFromSmarts("[n]"),                 "aromatic_nitrogen",    0.2,  0.6),
    (Chem.MolFromSmarts("[o]"),                 "aromatic_oxygen",      0.1,  0.4),
    (Chem.MolFromSmarts("[s]"),                 "aromatic_sulfur",      0.4,  0.3),
    (Chem.MolFromSmarts("[CX3]=[CX3]"),         "alkene",               0.5,  0.2),
    (Chem.MolFromSmarts("[CX2]#[CX2]"),         "alkyne",               0.6,  0.3),
    (Chem.MolFromSmarts("[F]"),                 "fluorine",            -0.2,  0.8),
    (Chem.MolFromSmarts("[Cl]"),                "chlorine",            -0.1,  0.5),
    (Chem.MolFromSmarts("[Br]"),                "bromine",             -0.1,  0.4),
    (Chem.MolFromSmarts("[NX3;H2][CX4]"),       "primary_amine",        0.3, -0.3),
    (Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4]"),"secondary_amine",      0.2, -0.2),
    (Chem.MolFromSmarts("[OH][CX4]"),           "alcohol",              0.1, -0.1),
    (Chem.MolFromSmarts("[CX3](=O)[OX2H0]"),    "ester",                0.1,  0.3),
    (Chem.MolFromSmarts("[CX3](=O)[NX3]"),      "amide",                0.1,  0.4),
    (Chem.MolFromSmarts("[CX3](=O)[CX3]"),      "ketone",               0.2,  0.4),
    (Chem.MolFromSmarts("S(=O)(=O)[CX4]"),      "sulfone",              0.1,  0.5),
    (Chem.MolFromSmarts("[C](F)(F)F"),          "trifluoromethyl",     -0.3,  1.0),
    (Chem.MolFromSmarts("[#6]1[#6][#6][#6][#6][#6]1"), "phenyl",        0.5,  0.2),
    (Chem.MolFromSmarts("[#6]1[#6][#6][#6][#6]1"), "thienyl",           0.6,  0.3),
    (Chem.MolFromSmarts("c1ccc2cc3ccccc3cc2c1"), "anthracene_core",     1.0,  0.5),
    (Chem.MolFromSmarts("c1ccc2c(c1)ccc1ccccc12"), "tetracene_core",    1.2,  0.6),
    (Chem.MolFromSmarts("[C]#[N]"),             "nitrile",              0.0,  0.7),
    (Chem.MolFromSmarts("[NX3](=O)=O"),         "nitro",               -0.2,  1.2),
    (Chem.MolFromSmarts("[OX2][CX4][CX4][OX2]"), "glyme_chelating",     0.1, -0.2),
]

_OE_BASE_HOLE_MOBILITY: float = 1.0
_OE_BASE_ELECTRON_AFFINITY: float = 0.5

_OE_CROSS_TERMS: list[tuple[str, str, float, str]] = [
    ("aromatic_nitrogen", "fluorine", 0.4, "fluorinated acceptor enhancement"),
    ("primary_amine", "nitro", 0.6, "push-pull donor-acceptor synergy"),
    ("phenyl", "nitrile", 0.3, "benzonitrile acceptor enhancement"),
    ("thienyl", "fluorine", 0.3, "fluorinated thiophene backbone"),
    ("alkene", "aromatic_carbon", 0.2, "extended conjugation chain"),
    ("nitro", "aromatic_nitrogen", 0.5, "nitro-pyridine acceptor pair"),
]


class OrganicElectronicsPack(BasePropertyModel):
    """Group-contribution model for organic electronics properties.

    Predicts hole mobility proxy and electron affinity proxy using
    fragment-additivity with fragments relevant to organic semiconductors.
    Cross-terms capture push-pull donor-acceptor interactions and
    extended conjugation effects.
    """

    name: str = "organic_electronics"
    fragments: list[tuple[Chem.Mol, str, float, float]] = _OE_FRAGMENTS
    base_values: dict[str, float] = {
        "hole_mobility": _OE_BASE_HOLE_MOBILITY,
        "electron_affinity": _OE_BASE_ELECTRON_AFFINITY,
    }
    cross_terms: list[tuple[str, str, float, str]] = _OE_CROSS_TERMS

    def _compute_cross_terms(self, counts: dict[str, int]) -> float:
        correction = 0.0
        for frag_a, frag_b, boost, _desc in self.cross_terms:
            if counts.get(frag_a, 0) > 0 and counts.get(frag_b, 0) > 0:
                correction += boost
        return max(-2.0, min(2.0, correction))

    def predict_hole_mobility(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["hole_mobility"]
        for _smarts, _name, hm, _ea in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, hm * 2.0)
        value += self._compute_cross_terms(counts)
        # Conjugation length bonus: longer pi-systems increase mobility
        n_arom = ctx.ring_count
        value += n_arom * 0.3
        return max(0.1, value)

    def predict_electron_affinity(self, ctx: MoleculeContext) -> float:
        mol = ctx.mol
        counts = self.count_fragments(mol)
        value = self.base_values["electron_affinity"]
        for _smarts, _name, _hm, ea in self.fragments:
            n = counts.get(_name, 0)
            value += self.saturate_contrib(n, ea * 2.0)
        value += self._compute_cross_terms(counts)
        # Electron-withdrawing groups boost EA
        n_withdrawing = (
            counts.get("nitrile", 0)
            + counts.get("nitro", 0)
            + counts.get("fluorine", 0)
            + counts.get("trifluoromethyl", 0)
            + counts.get("sulfone", 0)
        )
        value += n_withdrawing * 0.1
        return max(0.0, min(10.0, value))

    def predict_all(self, ctx: MoleculeContext) -> dict[str, float]:
        return {
            "hole_mobility_proxy": self.predict_hole_mobility(ctx),
            "electron_affinity_proxy": self.predict_electron_affinity(ctx),
        }

    def property_keys(self) -> dict[str, str]:
        return {
            "hole_mobility": "hole_mobility_proxy",
            "electron_affinity": "electron_affinity_proxy",
        }
