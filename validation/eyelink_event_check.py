"""Check pyemread's sample-based event detector against the EyeLink parser.

The ASC files distributed with pyemread contain both the raw binocular
samples (250 Hz) and the fixations detected online by the EyeLink 1000
parser (EFIX events). This script runs :func:`pyemread.ext_generic.detect_events`
on the raw samples and compares its fixations with EyeLink's, for the
three example participants, three passages and both eyes.

Agreement measures (see Andersson et al., 2017):

* ``kappa`` - Cohen's kappa for the sample-by-sample classification
  (inside a fixation vs not);
* ``recall`` / ``precision`` - proportion of EyeLink (detected) fixations
  that overlap a detected (EyeLink) fixation by at least half of their
  duration;
* ``onset_ms`` / ``offset_ms`` - median absolute difference in onset and
  offset of matched fixations;
* ``pos_px`` - median distance between the positions of matched fixations;
* ``dur_*`` - median fixation durations.

Output: ``results/eyelink_event_check.csv``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO, RESULTS
from pyemread import ext, ext_generic as eg

EXAMPLES = REPO / "examples" / "oralReading"
SUBJECTS = ["1950138", "1950149", "1950168"]


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return (po - pe) / (1 - pe)


def _in_fix(t: np.ndarray, fix: pd.DataFrame) -> np.ndarray:
    lab = np.zeros(len(t), bool)
    for s, e in zip(fix["start_time"], fix["end_time"]):
        lab |= (t >= s) & (t <= e)
    return lab


def _match(ref: pd.DataFrame, det: pd.DataFrame):
    """Fixations of ``ref`` overlapped >= 50% by a fixation of ``det``."""
    pairs = []
    ds, de = det["start_time"].to_numpy(), det["end_time"].to_numpy()
    for i, (s, e) in enumerate(zip(ref["start_time"], ref["end_time"])):
        ov = np.clip(np.minimum(de, e) - np.maximum(ds, s), 0, None)
        j = int(np.argmax(ov)) if len(ov) else -1
        if j >= 0 and ov[j] >= 0.5 * (e - s):
            pairs.append((i, j))
    return pairs


def main(px_per_deg: float | None = None):
    logging.getLogger("pyemread").setLevel(logging.ERROR)
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for subj in SUBJECTS:
        (tmp / subj).mkdir()
        shutil.copy(EXAMPLES / subj / f"{subj}.asc", tmp / subj)
        sac, efix = ext.read_srr_asc(str(tmp), subj, "RP", rec_last_fix=True, lump_fix=False)
        if px_per_deg is None:
            # pixels per degree implied by the EyeLink saccade amplitudes
            d = np.hypot(sac.x2_pos - sac.x1_pos, sac.y2_pos - sac.y1_pos)
            ok = sac.ampl > 0.5
            ppd = float(np.median(d[ok] / sac.ampl[ok]))
        else:
            ppd = px_per_deg
        st = ext.read_timestamp(str(tmp), subj, "RP")
        for eye, xc, yc in (("L", "x_pos1", "y_pos1"), ("R", "x_pos2", "y_pos2")):
            smp = eg.read_samples(st, "time", xc, yc, trial_col="trial_id", eye=eye)
            for method, kw in (("IVT", dict(velocity_threshold=30.0)),
                               ("IDT", dict(dispersion_threshold=1.0))):
                _, det = eg.detect_events(smp, method=method, px_per_deg=ppd, **kw)
                for tr in sorted(efix["trial_id"].unique()):
                    ref = efix[(efix.trial_id == tr) & (efix.eye == eye)].reset_index(drop=True)
                    dd = det[det.trial_id == tr].reset_index(drop=True)
                    t = smp.loc[smp.trial_id == tr, "time"].to_numpy(float)
                    a, b = _in_fix(t, ref), _in_fix(t, dd)
                    m1, m2 = _match(ref, dd), _match(dd, ref)
                    on = [abs(ref.start_time[i] - dd.start_time[j]) for i, j in m1]
                    off = [abs(ref.end_time[i] - dd.end_time[j]) for i, j in m1]
                    pos = [np.hypot(ref.x_pos[i] - dd.x_pos[j], ref.y_pos[i] - dd.y_pos[j]) for i, j in m1]
                    rows.append({"subj": subj, "trial_id": tr, "eye": eye, "method": method,
                                 "px_per_deg": round(ppd, 1), "n_eyelink": len(ref), "n_detected": len(dd),
                                 "kappa": _kappa(a, b), "recall": len(m1) / max(len(ref), 1),
                                 "precision": len(m2) / max(len(dd), 1),
                                 "onset_ms": float(np.median(on)) if on else np.nan,
                                 "offset_ms": float(np.median(off)) if off else np.nan,
                                 "pos_px": float(np.median(pos)) if pos else np.nan,
                                 "dur_eyelink": float(ref.duration.median()),
                                 "dur_detected": float((dd.end_time - dd.start_time).median())})
    shutil.rmtree(tmp)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "eyelink_event_check.csv", index=False)
    summ = out.groupby("method").agg(
        n_eyelink=("n_eyelink", "sum"), n_detected=("n_detected", "sum"),
        kappa=("kappa", "median"), recall=("recall", "median"), precision=("precision", "median"),
        onset_ms=("onset_ms", "median"), offset_ms=("offset_ms", "median"),
        pos_px=("pos_px", "median"), dur_eyelink=("dur_eyelink", "median"),
        dur_detected=("dur_detected", "median"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
