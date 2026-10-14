"""Write an Eye-Tracking-BIDS version of one example recording.

Converts the right-eye samples and the EyeLink-detected fixations of
participant 1950138 (examples/oralReading/1950138/1950138.asc) to the
Eye-Tracking-BIDS layout (BIDS 1.11, physiological recordings with
PhysioType "eyetrack"), with trial messages in the physioevents file. The
result, examples/bids_example/, shows how recordings from any tracker that
can be exported to BIDS enter the pyemread pipeline (see the "BIDS input"
section of the tutorial).

Run from the repository root:  python examples/make_bids_example.py
"""

import gzip
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from pyemread import ext

HERE = Path(__file__).resolve().parent
OUT = HERE / "bids_example"
SUBJ = "1950138"

tmp = Path(tempfile.mkdtemp())
(tmp / SUBJ).mkdir()
shutil.copy(HERE / "oralReading" / SUBJ / f"{SUBJ}.asc", tmp / SUBJ)
st = ext.read_timestamp(str(tmp), SUBJ, "RP")
_, efix = ext.read_srr_asc(str(tmp), SUBJ, "RP", rec_last_fix=True, lump_fix=False)
shutil.rmtree(tmp)
efix = efix[efix.eye == "R"]

if OUT.exists():
    shutil.rmtree(OUT)
beh = OUT / "sub-1950138" / "beh"
beh.mkdir(parents=True)
(OUT / "dataset_description.json").write_text(json.dumps({
    "Name": "pyemread example: oral reading of three English passages (right eye)",
    "BIDSVersion": "1.11.0", "DatasetType": "raw",
    "License": "CC-BY-4.0",
    "HowToAcknowledge": "Please cite pyemread, https://doi.org/10.5281/zenodo.19622177.",
    "GeneratedBy": [{"Name": "pyemread examples/make_bids_example.py"}]}, indent=2))
(OUT / "README").write_text(
    "Eye-Tracking-BIDS version of the right-eye data of participant 1950138 from the\n"
    "pyemread example data (EyeLink II, 250 Hz, binocular recording). Samples are in\n"
    "screen pixels (1280 x 1024, origin top left). Fixations are those detected online by\n"
    "the EyeLink parser. Trials are delimited by TRIALID / TRIAL_RESULT messages, and the\n"
    "stimulus of each trial is given by the message '!V TRIAL_VAR picture_name <file>'.\n")
stem = "sub-1950138_task-reading_recording-eye1"
with gzip.open(beh / f"{stem}_physio.tsv.gz", "wt") as fh:
    pd.DataFrame({"t": st.time, "x": st.x_pos2, "y": st.y_pos2, "p": st.pup_size2}).to_csv(
        fh, sep="\t", header=False, index=False, na_rep="n/a")
(OUT / "task-reading_physio.json").write_text(json.dumps({
    "PhysioType": "eyetrack", "Manufacturer": "SR-Research", "ManufacturersModelName": "EyeLink II",
    "Columns": ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
    "SamplingFrequency": 250, "SampleCoordinateSystem": "gaze-on-screen",
    "EnvironmentCoordinates": "top-left", "ScreenResolution": [1280, 1024],
    "timestamp": {"Description": "EyeLink time stamp", "Units": "ms"},
    "x_coordinate": {"Units": "pixel"}, "y_coordinate": {"Units": "pixel"},
    "pupil_size": {"Units": "a.u."}}, indent=2))
(beh / f"{stem}_physio.json").write_text(json.dumps({"RecordedEye": "right", "StartTime": 0}, indent=2))
t0 = float(st.time.iloc[0])
ev = [((r.start_time - t0) / 1000, r.duration / 1000, "fixation", 0, "n/a") for r in efix.itertuples()]
for tr in st.trial_id.unique():
    g = st[st.trial_id == tr]
    ev += [((g.time.min() - t0) / 1000, 0, "n/a", 0, f"TRIALID {tr}"),
           ((g.time.min() + 1 - t0) / 1000, 0, "n/a", 0,
            f"!V TRIAL_VAR picture_name {g.trial_type.iloc[0]}.png"),
           ((g.time.max() + 1 - t0) / 1000, 0, "n/a", 0, "TRIAL_RESULT 0")]
with gzip.open(beh / f"{stem}_physioevents.tsv.gz", "wt") as fh:
    pd.DataFrame(ev).sort_values(0).to_csv(fh, sep="\t", header=False, index=False)
(OUT / "task-reading_physioevents.json").write_text(json.dumps({
    "Columns": ["onset", "duration", "trial_type", "blink", "message"],
    "onset": {"Units": "s"}, "duration": {"Units": "s"},
    "trial_type": {"Levels": {"fixation": "Fixation detected by the EyeLink parser"}}}, indent=2))
print("Wrote", OUT)
