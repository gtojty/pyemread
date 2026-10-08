"""Non-EyeLink input: fixation tables, Eye-Tracking-BIDS, vendor exports,
event detection and epoching."""

import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pyemread import cal, ext, ext_generic as eg
from conftest import EXAMPLES, REGIONS, SUBJECT


# ---------------------------------------------------------------- fixation tables

def test_read_fixations_with_column_map(tmp_path):
    src = pd.DataFrame({"onset_s": [1.0, 1.3, 1.6], "dur_s": [0.25, 0.2, 0.22],
                        "gx": [300, 360, 420], "gy": [100, 101, 99],
                        "stim": ["story01.png"] * 3})
    p = tmp_path / "fix.csv"
    src.to_csv(p, index=False)
    fix = eg.read_fixations(str(p), {"start": "onset_s", "duration": "dur_s", "x": "gx",
                                     "y": "gy", "trial_type": "stim"}, time_unit="s", subj="P1")
    assert list(fix.columns) == ext.FIX_COLUMNS
    assert fix["start_time"].tolist() == [1000, 1300, 1600]
    assert fix["duration"].tolist() == pytest.approx([250, 200, 220])
    assert (fix["trial_id"] == 1).all() and fix["valid"].eq("yes").all()


# ---------------------------------------------------------------- event detection

def synthetic_samples(hz=500, ppd=40.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1000.0 / hz
    targets = [(300, 200), (380, 202), (460, 199), (540, 201), (300, 260), (380, 262)]
    t, x, y, truth = [], [], [], []
    now = 0.0
    for k, (tx, ty) in enumerate(targets):
        if k:
            px, py = targets[k - 1]
            for s in range(1, 13):                      # 24-ms saccade, smooth profile
                f = (1 - np.cos(np.pi * s / 13)) / 2
                t.append(now); x.append(px + f * (tx - px)); y.append(py + f * (ty - py)); now += dt
        truth.append((now, now + 200))
        for _ in range(int(200 / dt)):
            t.append(now); x.append(tx + rng.normal(0, 0.3)); y.append(ty + rng.normal(0, 0.3)); now += dt
    smp = pd.DataFrame({"time": t, "x": x, "y": y})
    smp.loc[(smp.time > truth[2][0] + 80) & (smp.time < truth[2][0] + 110), ["x", "y"]] = np.nan
    return eg.read_samples(smp), truth


def test_detect_events_recovers_fixations():
    smp, truth = synthetic_samples()
    # I-DT absorbs the start of each saccade into the preceding fixation, so its
    # boundaries are less precise than those of I-VT.
    for method, kw, tol in (("IVT", {"velocity_threshold": 30}, 6),
                            ("IDT", {"dispersion_threshold": 1.0}, 15)):
        sac, fix = eg.detect_events(smp, method=method, px_per_deg=40, **kw)
        assert len(fix) == len(truth), method
        assert np.allclose(fix["start_time"], [a for a, _ in truth], atol=tol)
        assert np.allclose(fix["end_time"], [b for _, b in truth], atol=tol)
        assert list(fix.columns) == ext.FIX_COLUMNS and list(sac.columns) == ext.SAC_COLUMNS
        assert len(sac) == len(truth) - 1


def test_detect_events_requires_units():
    smp, _ = synthetic_samples()
    with pytest.raises(ValueError, match="px_per_deg"):
        eg.detect_events(smp, velocity_threshold=30)
    sac, fix = eg.detect_events(smp, velocity_threshold=1200, velocity_unit="px/s",
                                dispersion_unit="px", merge_dist=20)
    assert len(fix) == 6


def test_long_gap_splits_fixation():
    smp, truth = synthetic_samples()
    t0 = truth[1][0]
    smp.loc[(smp.time > t0 + 50) & (smp.time < t0 + 150), ["x", "y"]] = np.nan   # 100-ms loss
    _, fix = eg.detect_events(smp, px_per_deg=40, min_fix_duration=40)
    assert len(fix) == len(truth) + 1


# ---------------------------------------------------------------- vendor exports

def tobii_export(path, encoding, modern=True):
    ts = "Recording timestamp [μs]" if modern else "Recording timestamp"
    unit = "[DACS px]" if modern else "(MCSpx)"
    rows = []
    t = 1_000_000
    for stim, fixes in (("story01.png", [(300, 100), (360, 101)]), ("story02.png", [(310, 160)])):
        for k, (fx, fy) in enumerate(fixes):
            for s in range(30):                                    # 600 Hz, 50 ms
                rows.append({ts: t, f"Gaze point X {unit}": fx + 1, f"Gaze point Y {unit}": fy,
                             "Eye movement type": "Fixation", "Eye movement type index": k + 1,
                             "Gaze event duration [ms]": 50, f"Fixation point X {unit}": fx,
                             f"Fixation point Y {unit}": fy, "Presented Stimulus name": stim})
                t += 1667
            rows.append({ts: t, f"Gaze point X {unit}": fx + 40, f"Gaze point Y {unit}": fy,
                         "Eye movement type": "Saccade", "Eye movement type index": k + 1,
                         "Gaze event duration [ms]": 10, f"Fixation point X {unit}": np.nan,
                         f"Fixation point Y {unit}": np.nan, "Presented Stimulus name": stim})
            t += 10000
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding=encoding)


@pytest.mark.parametrize("encoding,modern", [("utf-8", True), ("utf-16", False)])
def test_tobii_readers(tmp_path, encoding, modern):
    p = tmp_path / "export.tsv"
    tobii_export(p, encoding, modern)
    fix = eg.tobii_fixations(str(p), subj="T1")
    assert fix["trial_type"].tolist() == ["story01", "story01", "story02"]
    assert fix["trial_id"].tolist() == [1, 1, 2]
    assert fix["x_pos"].tolist() == [300, 360, 310]
    assert fix["duration"].tolist() == pytest.approx([50, 50, 50])
    raw = eg.tobii_to_raw(str(p))
    assert raw["time"].iloc[1] - raw["time"].iloc[0] == pytest.approx(1.667)
    assert set(raw["trial_type"]) == {"story01", "story02"}


def test_pupil_core_surface(tmp_path):
    fx = pd.DataFrame({"world_timestamp": [10.0, 10.03, 10.5], "world_index": [1, 2, 3],
                       "fixation_id": [1, 1, 2], "start_timestamp": [10.0, 10.0, 10.5],
                       "duration": [250.0, 250.0, 300.0], "dispersion": [1, 1, 1],
                       "norm_pos_x": [0.25, 0.25, 0.5], "norm_pos_y": [0.75, 0.75, 0.5],
                       "x_scaled": 0, "y_scaled": 0, "on_surf": [True, True, False]})
    fx.to_csv(tmp_path / "fixations_on_surface_screen.csv", index=False)
    out = eg.pupil_core_surface_fixations(str(tmp_path / "fixations_on_surface_screen.csv"), 1280, 1024)
    assert len(out) == 1 and out["x_pos"].iloc[0] == 320 and out["y_pos"].iloc[0] == 256
    assert out["duration"].iloc[0] == 250
    gz = pd.DataFrame({"world_timestamp": [1.0, 1.0], "world_index": [1, 1],
                       "gaze_timestamp": [1.0, 1.004], "x_norm": [0.5, 0.5], "y_norm": [0.25, 0.25],
                       "x_scaled": 0, "y_scaled": 0, "on_surf": [True, False], "confidence": [0.9, 0.9]})
    gz.to_csv(tmp_path / "g.csv", index=False)
    raw = eg.pupil_core_surface_to_raw(str(tmp_path / "g.csv"), 1280, 1024)
    assert raw["x"].iloc[0] == 640 and raw["y"].iloc[0] == 768 and np.isnan(raw["x"].iloc[1])


def test_neon_surface(tmp_path):
    fx = pd.DataFrame({"section id": "s", "recording id": "r", "fixation id": [1, 2],
                       "start timestamp [ns]": [1_000_000_000, 1_400_000_000],
                       "end timestamp [ns]": [1_250_000_000, 1_600_000_000], "duration [ms]": [250, 200],
                       "fixation detected on surface": [True, True],
                       "fixation x [normalized]": [0.25, 0.5], "fixation y [normalized]": [0.1, 0.2]})
    fx.to_csv(tmp_path / "fixations.csv", index=False)
    out = eg.neon_surface_fixations(str(tmp_path / "fixations.csv"), 1000, 800)
    assert out["x_pos"].tolist() == [250, 500] and out["y_pos"].tolist() == [80, 160]
    assert out["duration"].tolist() == [250, 200]


# ---------------------------------------------------------------- Eye-Tracking-BIDS round trip

def write_bids_from_asc(eyelink_dir, bids_dir):
    """Convert the right-eye data of the example ASC file to Eye-Tracking-BIDS."""
    d = str(eyelink_dir)
    st = ext.read_timestamp(d, SUBJECT, "RP")
    _, efix = ext.read_srr_asc(d, SUBJECT, "RP", rec_last_fix=True, lump_fix=False)
    efix = efix[efix.eye == "R"]
    beh = bids_dir / "sub-01" / "beh"
    beh.mkdir(parents=True)
    (bids_dir / "dataset_description.json").write_text(json.dumps({"Name": "t", "BIDSVersion": "1.11.0"}))
    stem = "sub-01_task-reading_recording-eye1"
    phys = pd.DataFrame({"timestamp": st.time, "x": st.x_pos2, "y": st.y_pos2, "p": st.pup_size2})
    with gzip.open(beh / f"{stem}_physio.tsv.gz", "wt") as fh:
        phys.to_csv(fh, sep="\t", header=False, index=False, na_rep="n/a")
    (bids_dir / "task-reading_physio.json").write_text(json.dumps({
        "PhysioType": "eyetrack", "Columns": ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
        "SamplingFrequency": 250, "SampleCoordinateSystem": "gaze-on-screen",
        "timestamp": {"Units": "ms"}, "x_coordinate": {"Units": "pixel"}, "y_coordinate": {"Units": "pixel"}}))
    (beh / f"{stem}_physio.json").write_text(json.dumps({"RecordedEye": "right", "StartTime": 0}))
    t0 = float(st.time.iloc[0])
    ev = [((r.start_time - t0) / 1000, r.duration / 1000, "fixation", 0, "n/a") for r in efix.itertuples()]
    for tr in st.trial_id.unique():
        g = st[st.trial_id == tr]
        ev.append(((g.time.min() - t0) / 1000, 0, "n/a", 0, f"TRIALID {tr}"))
        ev.append(((g.time.min() + 1 - t0) / 1000, 0, "n/a", 0,
                   f"!V TRIAL_VAR picture_name {g.trial_type.iloc[0]}.png"))
        ev.append(((g.time.max() + 1 - t0) / 1000, 0, "n/a", 0, "TRIAL_RESULT 0"))
    evdf = pd.DataFrame(ev).sort_values(0)
    with gzip.open(beh / f"{stem}_physioevents.tsv.gz", "wt") as fh:
        evdf.to_csv(fh, sep="\t", header=False, index=False)
    (bids_dir / "task-reading_physioevents.json").write_text(json.dumps({
        "Columns": ["onset", "duration", "trial_type", "blink", "message"],
        "onset": {"Units": "s"}, "duration": {"Units": "s"}}))
    return beh / f"{stem}_physio.tsv.gz", efix


def test_bids_round_trip(eyelink_dir, tmp_path):
    physio, efix = write_bids_from_asc(eyelink_dir, tmp_path / "bids")
    samples, events, meta = eg.read_bids_eyetrack(str(physio))
    assert meta["SamplingFrequency"] == 250 and samples["eye"].iloc[0] == "R"
    fix = eg.bids_fixations(samples, events)
    assert len(fix) == len(efix)
    assert np.allclose(fix["start_time"], efix["start_time"], atol=0.5)
    assert np.nanmedian(np.abs(fix["x_pos"].to_numpy() - efix["x_pos"].to_numpy())) < 2
    trials = eg.trials_from_messages(events, r"TRIALID", r"TRIAL_RESULT",
                                     r"picture_name\s+(\S+?)\.png")
    assert trials["trial_type"].tolist() == ["story01", "story02", "story03"]
    fix = eg.epoch(fix, trials)
    out = tmp_path / "out"
    out.mkdir()
    for r in REGIONS:
        shutil.copy(EXAMPLES / r, out / r)
    eg.process_fixations(str(out), "sub-01", fix, REGIONS)
    em = cal.cal_write_em(str(out), "sub-01", REGIONS)
    assert em["tfurt"].notna().mean() > 0.7 and em["fpurt"].notna().mean() > 0.4


def test_epoch_samples():
    smp = pd.DataFrame({"time": np.arange(0, 1000, 10.0), "x": 1.0, "y": 1.0, "eye": "R", "trial_id": 1})
    trials = pd.DataFrame({"trial_id": [1, 2], "start": [100, 500], "end": [300, 600],
                           "trial_type": ["a", "b"]})
    ep = eg.epoch(smp, trials)
    assert ep.groupby("trial_id").size().tolist() == [20, 10]
    assert set(ep["trial_type"]) == {"a", "b"}


def test_official_bids_example_sidecar(tmp_path):
    """Sidecar inheritance and a header-less, BOM-prefixed file as in bids-examples."""
    d = tmp_path / "ds"
    (d / "sub-1" / "eeg").mkdir(parents=True)
    (d / "dataset_description.json").write_text("{}")
    stem = d / "sub-1" / "eeg" / "sub-1_task-dots_recording-eye1"
    (stem.parent / (stem.name + "_physio.json")).write_text(json.dumps({
        "SamplingFrequency": 10.0, "StartTime": 0.0, "PhysioType": "eyetrack", "RecordedEye": "left",
        "Columns": ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
        "SampleCoordinateSystem": "gaze-on-screen", "timestamp": {"Units": "s"},
        "x_coordinate": {"Units": "pixel"}, "y_coordinate": {"Units": "pixel"}}))
    with gzip.open(str(stem) + "_physio.tsv.gz", "wt", encoding="utf-8-sig") as fh:
        fh.write("0.0\t85.1\t81.9\t1017.7\n0.1\t292.5\t281.3\t3487.1\n")
    samples, events, meta = eg.read_bids_eyetrack(str(stem) + "_physio.tsv.gz")
    assert samples["time"].tolist() == [0.0, 100.0] and samples["eye"].iloc[0] == "L"
    assert samples["x"].iloc[1] == pytest.approx(292.5) and len(events) == 0
