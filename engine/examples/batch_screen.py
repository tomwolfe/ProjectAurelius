"""Example: batch-screen molecules from a SMILES file."""
from aurelius.pipeline import AureliusPipeline

pipeline = AureliusPipeline()

smiles_list = [
    "C1COC(=O)O1",       # Ethylene carbonate
    "COCCOC",             # DME
    "CCS(=O)(=O)C",       # DMSO analogue
]

for smi in smiles_list:
    result = pipeline.screen(smi)
    print(f"{smi:30s} Score: {result.composite_score:6.2f}")
