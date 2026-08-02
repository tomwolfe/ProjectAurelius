"""OrganicElectronicsPack — Group-contribution model for organic electronics.

Predicts hole mobility proxy and electron affinity proxy using
fragment-additivity with fragments and cross-terms relevant to
organic semiconductors (conjugated backbones, donor-acceptor motifs,
common OLED/OPV building blocks).
"""

from __future__ import annotations

from rdkit import Chem

from aurelius.scoring.oracle.gc import BasePropertyModel
from aurelius.types import MoleculeContext

# Fragment definitions: (SMARTS, name, hole_mobility_contrib, electron_affinity_contrib)
# Values are unitless proxy contributions; positive = increases the property.
# Hole mobility: higher for electron-rich, planar, pi-extended systems.
# Electron affinity: higher for electron-withdrawing groups.
_OE_FRAGMENTS: list[tuple[Chem.Mol, str, float, float]] = [
    (Chem.MolFromSmarts("c1ccsc1"), "thiophene", 0.5, 0.1),
    (Chem.MolFromSmarts("c1ccc2c(c1)c1ccccc1[nH]2"), "carbazole", 0.8, 0.0),
    (Chem.MolFromSmarts("N(c1ccccc1)(c1ccccc1)c1ccccc1"), "triphenylamine", 1.0, -0.2),
    (Chem.MolFromSmarts("C=Cc1ccccc1"), "phenylenevinylene", 0.6, 0.2),
    (Chem.MolFromSmarts("c1ccc2nsnc2c1"), "benzothiadiazole", 0.0, 1.2),
    (Chem.MolFromSmarts("c1ccncc1"), "pyridine", 0.1, 0.6),
    (Chem.MolFromSmarts("c1ncncn1"), "triazine", 0.0, 1.0),
    (Chem.MolFromSmarts("c1nnco1"), "oxadiazole", 0.0, 0.8),
    (Chem.MolFromSmarts("c1ccc2nc[nH]c2c1"), "benzimidazole", 0.1, 0.7),
    (Chem.MolFromSmarts("c1ccc2c(c1)Nc1ccccc1S2"), "phenothiazine", 0.9, 0.0),
    (Chem.MolFromSmarts("c1csc2ccsc21"), "thienothiophene", 0.6, 0.1),
    (Chem.MolFromSmarts("c1ccc2cc3ccccc3cc2c1"), "anthracene", 0.9, 0.3),
    (Chem.MolFromSmarts("c1ccc2c(c1)Cc1ccccc1-2"), "fluorene", 0.7, 0.1),
    (Chem.MolFromSmarts("C#N"), "cyano", -0.1, 1.0),
    (Chem.MolFromSmarts("[F]"), "fluorine", -0.2, 0.8),
    (Chem.MolFromSmarts("[C](F)(F)F"), "trifluoromethyl", -0.3, 1.0),
    (Chem.MolFromSmarts("[N+](=O)[O-]"), "nitro", -0.2, 1.2),
    (Chem.MolFromSmarts("P(=O)"), "phosphine_oxide", 0.0, 0.6),
]

_OE_BASE_HOLE_MOBILITY: float = 1.0
_OE_BASE_ELECTRON_AFFINITY: float = 0.5

_OE_CROSS_TERMS: list[tuple[str, str, float, str]] = [
    ("triphenylamine", "benzothiadiazole", 0.6, "D-A push-pull OPV donor-acceptor"),
    ("carbazole", "triazine", 0.4, "carbazole-triazine TADF host synergy"),
    ("phenothiazine", "triazine", 0.5, "phenothiazine-triazine TADF emitter"),
    ("thiophene", "benzothiadiazole", 0.4, "thiophene-BTD donor-acceptor backbone"),
    ("anthracene", "cyano", 0.3, "cyanoanthracene acceptor enhancement"),
    ("triphenylamine", "nitro", 0.5, "push-pull donor-acceptor charge transfer"),
    ("fluorene", "cyano", 0.2, "cyanated fluorene acceptor"),
    ("thienothiophene", "benzothiadiazole", 0.5, "fused donor-acceptor backbone"),
]


class OrganicElectronicsPack(BasePropertyModel):
    """Group-contribution model for organic electronics properties.

    Predicts hole mobility proxy and electron affinity proxy using
    fragment-additivity with fragments relevant to organic semiconductors
    and OLED/OPV motifs. Cross-terms capture donor-acceptor push-pull
    interactions and extended conjugation effects.

    Fragment count is kept intentionally small (<20) to maintain simplicity
    while covering the most common OLED/OPV building blocks.
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
        n_withdrawing = (
            counts.get("cyano", 0)
            + counts.get("nitro", 0)
            + counts.get("fluorine", 0)
            + counts.get("trifluoromethyl", 0)
            + counts.get("benzothiadiazole", 0)
            + counts.get("triazine", 0)
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
