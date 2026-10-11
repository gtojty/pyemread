"""Complete pyemread workflow on the example data (the tutorial in the paper).

Run from the repository root:

    python examples/example_workflow.py

All outputs are written to ``tutorial_output/``. The script uses only the
files distributed in ``examples/``: three English passages and the EyeLink
recordings of three participants (1950138, 1950149, 1950168) reading them
aloud (Gong & Shuai, 2023).
"""

import shutil
from pathlib import Path

import pandas as pd
from matplotlib import font_manager

from pyemread import cal, ext, ext_robust, gen

HERE = Path(__file__).resolve().parent
OUT = Path("tutorial_output")
DATA = OUT / "data"
SUBJECTS = ["1950138", "1950149", "1950168"]
REGIONS = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]

# ---------------------------------------------------------------------------
# Step 1. Stimulus bitmap and region file (Listing 1)
# ---------------------------------------------------------------------------
FONTPATH = font_manager.findfont("DejaVu Sans Mono")   # ships with matplotlib
with open(HERE / "textRasters" / "story01.txt", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
gen.praster(direct=str(OUT / "stimuli"), fontpath=FONTPATH, st_pos="TopLeft",
            lang_type="English", text=lines, ID="story01")
print(pd.read_csv(OUT / "stimuli" / "story01.region.csv").head())

# ---------------------------------------------------------------------------
# Step 2. Extract fixations and assign them to lines and words (Listing 2)
# The recordings were made with the region files distributed with the data,
# so those are used from here on.
# ---------------------------------------------------------------------------
DATA.mkdir(parents=True, exist_ok=True)
for r in REGIONS:
    shutil.copy(HERE / "oralReading" / r, DATA / r)
for s in SUBJECTS:
    (DATA / s).mkdir(exist_ok=True)
    shutil.copy(HERE / "oralReading" / s / f"{s}.asc", DATA / s)

for s in SUBJECTS:
    sac, crl_sac, fix, crl_fix = ext.process_subject(str(DATA), s, REGIONS)
    print(s, len(fix), "fixations;", len(crl_fix), "line changes")

# ---------------------------------------------------------------------------
# Step 3. Check the line assignment and resolve flagged trials (Listing 3)
# ---------------------------------------------------------------------------
regions = ext.load_regions(str(DATA), REGIONS)
reports = []
for s in SUBJECTS:
    fix = pd.read_csv(DATA / s / f"{s}_Fix.csv")
    rep = ext_robust.trial_report(fix, regions)
    rep.insert(0, "subj", s)
    reports.append(rep)
reports = pd.concat(reports, ignore_index=True)
print(reports[["subj", "trial_id", "trial_type", "eye", "agree_slice_diff", "flag"]])

# Plot both assignments for every flagged trial and inspect them.
for r in reports[reports["flag"]].itertuples():
    fix = pd.read_csv(DATA / r.subj / f"{r.subj}_Fix.csv")
    trial = fix[(fix.trial_id == r.trial_id) & (fix.eye == r.eye) & (fix.valid == "yes")]
    gen.draw_line_assignment(trial, regions[r.trial_type],
                             out_file=str(OUT / f"check_{r.subj}_{r.trial_id}{r.eye}.png"),
                             title=f"{r.subj} trial {r.trial_id} {r.eye}: ")

# Decisions after inspection. An assignment is accepted when each row of
# fixations is assigned to a single text line and the line numbers increase
# down the passage, except at visible regressions to earlier lines.
# Participant 1950138, trial 1, right eye: the slice algorithm splits sloping
# rows between lines, while the original heuristic assigns each row to one
# line, so "DIFF" is used for that trial. In the other flagged trials the
# slice assignment was retained, except for participant 1950149, trial 1,
# left eye, which neither method assigned correctly; it is excluded.
CHOICES = {"1950138": {(1, "R"): "DIFF", "default": "SLICE"}}
EXCLUDE = {("1950149", 1, "L")}
for s, choice in CHOICES.items():
    ext.process_subject(str(DATA), s, REGIONS, classify_method=choice)

# Optional: an animation of a trial, synchronized with the audio recorded
# during oral reading (requires FFmpeg).
if gen.FFMPEG_AVAILABLE:
    gen.animate(str(DATA), "1950138", 1, eye="R",
                bitmap_file=str(HERE / "textRasters" / "story01.png"),
                sound_file=str(HERE / "oralReading" / "1950138" / "1950138-1.wav"))

# ---------------------------------------------------------------------------
# Step 4. Word-level measures for analysis (Listing 4)
# ---------------------------------------------------------------------------
em = pd.concat([cal.cal_write_em(str(DATA), s, REGIONS) for s in SUBJECTS],
               ignore_index=True)
keep = [(s, t, e) not in EXCLUDE for s, t, e in zip(em.subj, em.trial_id, em.eye)]
em = em[keep]
em.to_csv(OUT / "word_measures_both_eyes.csv", index=False)
# The recordings are binocular; the analysis uses one eye (here the right eye).
em = em[em.eye == "R"]
em.to_csv(OUT / "word_measures.csv", index=False)
print(em[["subj", "trial_type", "region", "word", "ffixurt", "fpurt", "rpurt",
          "tfurt"]].head(10))
print("Words fixated: {:.0%}".format(em["tfurt"].notna().mean()))
print("Words with a first pass: {:.0%}".format(em["fpurt"].notna().mean()))

# ---------------------------------------------------------------------------
# Step 5. Input from other trackers (Listing 5)
# (a) A fixation table exported by other software: the right-eye fixations of
# participant 1950138 as a plain CSV file (see
# examples/make_fixation_table_example.py).
# ---------------------------------------------------------------------------
from pyemread import ext_generic as eg

table = eg.read_fixations(
    HERE / "fixation_table" / "1950138_fixations.csv", sampfreq=250,
    column_map={"trial_id": "trial", "trial_type": "stimulus", "start": "start_ms",
                "end": "end_ms", "duration": "duration_ms", "x": "x_px", "y": "y_px"})
eg.process_fixations(str(DATA), "table-1950138", table, REGIONS,
                     classify_method=CHOICES["1950138"])   # the same choices as in Step 3
em_table = cal.cal_write_em(str(DATA), "table-1950138", REGIONS)
same = em[em.subj == "1950138"].set_index(["trial_id", "region"])["tfurt"]
other = em_table.set_index(["trial_id", "region"])["tfurt"].loc[same.index]
print("Total time identical to the EyeLink route for {:.0%} of words".format(
    ((same == other) | (same.isna() & other.isna())).mean()))

# (b) An Eye-Tracking-BIDS recording: examples/bids_example holds the
# right-eye data of participant 1950138 (see examples/make_bids_example.py).

physio = (HERE / "bids_example" / "sub-1950138" / "beh" /
          "sub-1950138_task-reading_recording-eye1_physio.tsv.gz")
samples, events, meta = eg.read_bids_eyetrack(str(physio))
trials = eg.trials_from_messages(events, r"TRIALID", r"TRIAL_RESULT",
                                 r"picture_name\s+(\S+?)\.png")
fix = eg.epoch(eg.bids_fixations(samples, events), trials)   # tracker's fixations
eg.process_fixations(str(DATA), "bids-1950138", fix, REGIONS)
em_bids = cal.cal_write_em(str(DATA), "bids-1950138", REGIONS)

# Alternatively, detect fixations in the raw samples (screen-based trackers):
_, fix_ivt = eg.detect_events(eg.epoch(samples, trials), method="IVT",
                              velocity_threshold=30, px_per_deg=40.5)
print(len(fix), "tracker fixations;", len(fix_ivt), "I-VT fixations")
