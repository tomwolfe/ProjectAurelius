"""Verify the Aurelius QED export (no Lean toolchain needed)."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from aurelius.scoring.oracle import reduction as R

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: ok = False

check("ea_calibration", R._EA_CALIBRATION == (0.6590, -2.9176))
check("born_const", R._BORN_CONSTANT_EV_ANGSTROM == 13.598)
check("identity", R.calibrate_ea_solution(1.5) == 1.5)
check("born_pos", R._born_solvation_correction(30.0, 3.0) > 0)
check("born_vacuum", R._born_solvation_correction(1.0, 3.0) == 0.0)

r = subprocess.run([sys.executable, os.path.join(HERE, "export_aurelius_to_qed.py"),
                    "--out", "/tmp/aurelius_lemmas.txt",
                    "--lean-out", os.path.join(ROOT, "QED", "AureliusExport.lean")],
                   capture_output=True, text=True)
check("export_runs", r.returncode == 0)
lean = open(os.path.join(ROOT, "QED", "AureliusExport.lean")).read()
for lemma in ["ea_calibration_affine", "born_solvation_pos", "ea_bounds", "reduction_stability_mono"]:
    check("lemma_" + lemma, lemma in lean)
check("no_sorry", "sorry" not in lean.lower())
check("lemmas_txt", os.path.exists("/tmp/aurelius_lemmas.txt"))
sys.exit(0 if ok else 1)
