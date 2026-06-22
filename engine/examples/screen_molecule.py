"""Example: screen a single molecule with the Aurelius Discovery Engine."""
from aurelius.pipeline import AureliusPipeline

pipeline = AureliusPipeline()
result = pipeline.screen("CC(=O)OC1=CC=CC=C1")
print(f"Aurelius Score: {result.composite_score:.2f}")
print(f"HOMO: {result.homo:.3f} eV")
print(f"LUMO: {result.lumo:.3f} eV")
print(f"Dielectric: {result.dielectric:.1f}")
print(f"Viscosity: {result.viscosity:.3f} cP")
