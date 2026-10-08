"""Effect of line assignment on word-level reading measures (Carr et al., 2022 data).

For each of the 48 trials, word-level measures are computed with
:func:`pyemread.cal.compute_word_measures` from the same fixations, once with
the human (gold-standard) line assignment and once with each algorithm's
assignment. Fixations discarded by the human correctors are excluded in
every condition, so that the comparison isolates the effect of line
assignment. For each measure, the script reports the percentage of words
whose value is identical to the gold-standard value (a word missing the
measure in both counts as identical) and the mean absolute difference
over words with a value in both.

Requires ``results/carr_assignments.csv`` from ``carr_benchmark.py``.
Output: ``results/downstream_measures.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import eyekit

from common import DRIFT, RESULTS, passage_to_region
from pyemread import cal, lines

MEASURES = ["ffixurt", "fpurt", "rpurt", "tfurt", "fpregres"]
METHODS = ["pyemread_slice", "pyemread_diff", "attach", "cluster", "merge", "slice", "warp"]


def word_measures(d: pd.DataFrame, line_col: str, reg: pd.DataFrame) -> pd.DataFrame:
    fix = pd.DataFrame({"subj": "s", "trial_id": 1, "trial_type": "t", "eye": "R",
                        "start_time": d["start"].to_numpy(float), "end_time": d["end"].to_numpy(float),
                        "duration": (d["end"] - d["start"]).to_numpy(float),
                        "x_pos": d["x"].to_numpy(float), "y_pos": d["y"].to_numpy(float),
                        "valid": np.where(d["gold"] > 0, "yes", "no"),
                        "line_no": d[line_col].replace(0, np.nan).to_numpy(float)})
    fix = lines.assign_regions(fix, reg)
    return cal.compute_word_measures(fix, reg)


def main():
    passages = eyekit.io.load(DRIFT / "data" / "passages.json")
    A = pd.read_csv(RESULTS / "carr_assignments.csv")
    rows = []
    for trial, d in A.groupby("trial", sort=False):
        reg = passage_to_region(passages[d["passage"].iloc[0]])
        gold = word_measures(d, "gold", reg)
        for m in METHODS:
            em = word_measures(d, m, reg)
            for meas in MEASURES:
                g, v = gold[meas].to_numpy(float), em[meas].to_numpy(float)
                same = (g == v) | (np.isnan(g) & np.isnan(v))
                both = np.isfinite(g) & np.isfinite(v)
                rows.append({"trial": trial, "age_group": d["age_group"].iloc[0], "method": m,
                             "measure": meas, "pct_identical": 100 * same.mean(),
                             "mad": float(np.mean(np.abs(g[both] - v[both]))) if both.any() else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "downstream_measures.csv", index=False)
    print(out.pivot_table(index="method", columns="measure", values="pct_identical",
                          aggfunc="mean").round(1).to_string())


if __name__ == "__main__":
    main()
