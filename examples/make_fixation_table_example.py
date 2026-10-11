"""Write a fixation table, as exported by other software, for one recording.

Exports the right-eye fixations detected by the EyeLink parser for
participant 1950138 (examples/oralReading/1950138/1950138.asc) as a plain
CSV file with column names of the kind other programs use. The file,
examples/fixation_table/1950138_fixations.csv, shows how fixations exported
from any tracker or program enter the pyemread pipeline with
ext_generic.read_fixations (Step 5 of the tutorial).

Run from the repository root:  python examples/make_fixation_table_example.py
"""

import shutil
import tempfile
from pathlib import Path

from pyemread import ext

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixation_table"
SUBJ = "1950138"

tmp = Path(tempfile.mkdtemp())
(tmp / SUBJ).mkdir()
shutil.copy(HERE / "oralReading" / SUBJ / f"{SUBJ}.asc", tmp / SUBJ)
# the fixations as detected, without pyemread's validity rules (these are
# applied again when the table is processed)
_, efix = ext.read_srr_asc(str(tmp), SUBJ, "RP", rec_last_fix=True, lump_fix=False)
shutil.rmtree(tmp)
efix = efix[efix.eye == "R"]

OUT.mkdir(exist_ok=True)
table = efix.rename(columns={"trial_id": "trial", "trial_type": "stimulus", "eye": "eye",
                             "start_time": "start_ms", "end_time": "end_ms",
                             "duration": "duration_ms",
                             "x_pos": "x_px", "y_pos": "y_px"})
table = table[["trial", "stimulus", "eye", "start_ms", "end_ms", "duration_ms", "x_px", "y_px"]]
table.to_csv(OUT / f"{SUBJ}_fixations.csv", index=False)
print(f"Wrote {len(table)} fixations to {OUT / f'{SUBJ}_fixations.csv'}")
