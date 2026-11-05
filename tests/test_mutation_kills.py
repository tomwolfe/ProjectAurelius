"""Mutation-kill guards: each test fails if the described mutant is applied."""
from rdkit import Chem


def test_predict_corrected_batch_not_empty():
    from aurelius.scoring.oracle.delta_correction import DeltaCorrection
    dc = DeltaCorrection.__new__(DeltaCorrection)
    import numpy as np
    dc._prior_std_homo, dc._prior_std_lumo = 0.5, 0.5
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    base = [(-6.0, -1.0), (-6.5, -1.5)]
    d_h = np.array([0.1, -0.1], dtype=np.float32)
    d_l = np.array([0.05, -0.05], dtype=np.float32)
    s = np.array([0.1, 0.1], dtype=np.float32)
    dc.predict_deltas_batch = lambda ms, return_std=True: (d_h, d_l, s, s)
    h, l = dc.predict_corrected_batch(mols, base=base)
    assert len(h) == 2 and len(l) == 2  # kills break_return (empty return)
    assert abs(float(h[0]) - (-6.0)) < 0.5


def test_gaussian_peaks_at_target():
    from aurelius.pipeline import _gaussian
    assert _gaussian(1.0, 1.0, 0.5) == 1.0
    assert _gaussian(5.0, 1.0, 0.5) < _gaussian(1.5, 1.0, 0.5)  # kills negate_compare


def test_is_novel_scaffold_bool():
    from aurelius.agent.selection import _is_novel_scaffold
    known = frozenset({"c1ccccc1"})
    assert _is_novel_scaffold("C1CCOC1", known) is True  # kills flip_bool
    assert _is_novel_scaffold("c1ccccc1", known) is False
    assert _is_novel_scaffold(None, known) is False


def test_cavity_radius_positive_scale():
    from aurelius.scoring.oracle.reduction import _cavity_radius
    small = _cavity_radius(Chem.MolFromSmiles("C"))
    big = _cavity_radius(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    assert small > 0 and big > small  # kills arithmetic mutants
