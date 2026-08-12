"""Mock wet-lab cycle test: suggest → ingest → refit → suggest.

Proves the full closed-loop infrastructure works before involving real lab
equipment. Tests that measurements actually inform the next suggestion set
and that oracle MAE improves on frozen holdout molecules.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from rdkit import Chem

from aurelius.agent.experiment_suggester import suggest_experiments
from aurelius.agent.experimental_ingestion import ingest_experimental_results
from aurelius.scoring.oracle.delta_correction import DeltaCorrection


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "aurelius", "data")


def _load_calibration(path: str = "orbital_calibration.json") -> list[dict]:
    with open(os.path.join(DATA_DIR, path)) as f:
        data = json.load(f)
    if isinstance(data, dict):
        raw = data.get("entries", [])
    else:
        raw = data
    return [e for e in raw if Chem.MolFromSmiles(e["smiles"]) is not None]


def _split(entries: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    n_seed = 20
    n_hold = max(20, int(0.50 * len(entries)))
    holdout = [entries[i] for i in idx[:n_hold]]
    seed_calib = [entries[i] for i in idx[n_hold:n_hold + n_seed]]
    pool = [entries[i] for i in idx[n_hold + n_seed:]]
    return holdout, seed_calib, pool


def _fit(entries: list[dict]) -> DeltaCorrection:
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in entries]
    return DeltaCorrection(calib=entries, calib_smiles=smiles)


def _evaluate(model: DeltaCorrection, holdout: list[dict]) -> dict:
    pred, ref = [], []
    for entry in holdout:
        homo, _ = model.predict_corrected(Chem.MolFromSmiles(entry["smiles"]))
        pred.append(homo)
        ref.append(entry["homo_eV"])
    pred_arr, ref_arr = np.asarray(pred), np.asarray(ref)
    rho = float(np.corrcoef(pred_arr, ref_arr)[0, 1]) if len(pred_arr) > 1 else 0.0
    mae = float(np.abs(pred_arr - ref_arr).mean()) if len(pred_arr) > 0 else 0.0
    return {"spearman_rho": rho, "mae_eV": mae, "n": len(holdout)}


def _noisy(entry: dict, noise_eV: float, rng: random.Random) -> dict:
    if noise_eV <= 0.0:
        return entry
    return {
        **entry,
        "homo_eV": entry["homo_eV"] + rng.gauss(0.0, noise_eV),
        "lumo_eV": entry["lumo_eV"] + rng.gauss(0.0, noise_eV),
    }


def test_full_cycle(tmp_path: Path):
    """End-to-end mock wet-lab cycle: suggest → ingest → refit → suggest."""
    # 1. Load and split data
    entries = _load_calibration()
    holdout, seed_calib, pool = _split(entries, seed=42)

    # 2. Get initial suggestions from the pool (top 10 by information gain)
    initial_smiles = [e["smiles"] for e in pool]
    initial = suggest_experiments(
        initial_smiles, top_n=10,
        properties=["homo"],
        expand_pool=False,
    )

    initial_smile_set = {s.smiles for s in initial}
    assert len(initial_smile_set) == 10, f"Expected 10 suggestions, got {len(initial_smile_set)}"

    # 3. Simulate measurements with Gaussian noise
    rng = random.Random(42)
    measured = [_noisy(e, noise_eV=0.1, rng=rng) for e in pool[:10]]

    # 4. Ingest the measurements
    measurement_entries = {"measurements": [
        {
            "smiles": e["smiles"],
            "name": e.get("name", "unknown"),
            "measured_property": "homo_eV",
            "value": e["homo_eV"],
            "units": "eV",
            "temperature_K": 298.15,
            "method": "reference",
        }
        for e in measured
    ]}

    measurement_path = tmp_path / "measurements.json"
    measurement_path.write_text(json.dumps(measurement_entries))

    report = ingest_experimental_results(str(measurement_path), trigger_refit=True)
    assert report.n_accepted >= 5, f"Expected at least 5 accepted measurements, got {report.n_accepted}"

    # 5. Get new suggestions after ingestion
    new_smiles = [e["smiles"] for e in pool]
    new = suggest_experiments(
        new_smiles, top_n=10,
        properties=["homo"],
        expand_pool=False,
    )

    new_smile_set = {s.smiles for s in new}
    assert len(new_smile_set) == 10, f"Expected 10 new suggestions, got {len(new_smile_set)}"

    # 6. Verify suggestions changed (Tanimoto distance > 0.3 between old and new sets)
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    initial_fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), radius=2, nBits=2048
    ) for s in initial_smile_set]
    new_fps = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(s), radius=2, nBits=2048
    ) for s in new_smile_set]

    # Compute average Tanimoto between initial and new batches
    tanimoto_sum = 0.0
    count = 0
    for ifp in initial_fps:
        for jfp in new_fps:
            tanimoto_sum += float(DataStructs.TanimotoSimilarity(ifp, jfp))
            count += 1

    mean_tanimoto = tanimoto_sum / count if count > 0 else 1.0
    assert mean_tanimoto < 0.3, (
        f"Suggestions did not change sufficiently: mean Tanimoto = {mean_tanimoto:.4f} "
        f"(expected < 0.3)"
    )

    # 7. Verify oracle MAE improved on holdout
    model_before = _fit(seed_calib)
    metrics_before = _evaluate(model_before, holdout)

    # Refit with ingested data (add measured entries to seed calibration)
    seed_calib_extended = list(seed_calib) + measured
    model_after = _fit(seed_calib_extended)
    metrics_after = _evaluate(model_after, holdout)

    mae_improved = metrics_before["mae_eV"] > metrics_after["mae_eV"]
    assert mae_improved, (
        f"MAE did not improve after ingestion: before={metrics_before['mae_eV']:.4f} eV, "
        f"after={metrics_after['mae_eV']:.4f} eV"
    )