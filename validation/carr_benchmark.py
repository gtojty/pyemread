"""Line-assignment accuracy on the human-corrected reading data of Carr et al. (2022).

The data set contains 48 trials (24 adults, 24 children) of passage
reading in Italian, recorded with an EyeLink 1000 Plus. Each trial was
corrected by hand (``gold.json``); fixations removed by the correctors are
marked ``discarded``. Carr et al. also published the output of ten
correction algorithms and the ``attach`` baseline on the same trials.

This script runs pyemread's line-assignment methods (``diff``, the original
pyemread heuristic, and ``slice``, pyemread's adaptation of Carr et al.'s
implementation of Glandorf & Schroeder's algorithm) on the uncorrected
fixations (``sample.json``) and scores every method against the gold
standard in two ways:

* ``acc_all`` - Carr et al.'s measure: percentage of all fixations whose
  line matches the gold standard, with discarded fixations coded as line 0
  (so every algorithm is penalised for them equally);
* ``acc_kept`` - percentage of the fixations kept by the human correctors
  that were assigned to the correct line (unassigned fixations count as
  errors).

Outputs: ``results/carr_accuracy.csv`` (one row per trial and method),
``results/carr_assignments.csv`` (one row per fixation) and
``results/carr_diagnostics.csv`` (the trial-level diagnostics of
:func:`pyemread.lines.line_assignment_report`).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import eyekit

from common import (CARR_ALGORITHMS, DRIFT, PYEMREAD_METHODS, RESULTS,
                    passage_to_region, run_pyemread, y_to_line)


def main():
    fixdir = DRIFT / "data" / "fixations"
    passages = eyekit.io.load(DRIFT / "data" / "passages.json")
    sample = eyekit.io.load(fixdir / "sample.json")
    gold = eyekit.io.load(fixdir / "gold.json")
    outputs = {a: eyekit.io.load(fixdir / f"{a}.json") for a in CARR_ALGORITHMS}

    diag_rows = []
    acc_rows, fix_rows, timing = [], [], {m: 0.0 for m in PYEMREAD_METHODS}
    for trial_id, trial in sample.items():
        passage = passages[trial["passage_id"]]
        mid = np.array(passage.midlines, float)
        reg = passage_to_region(passage)
        fx = list(trial["fixations"])
        x = np.array([f.x for f in fx], float)
        y = np.array([f.y for f in fx], float)
        g = list(gold[trial_id]["fixations"])
        g_disc = np.array([f.discarded for f in g])
        g_line = np.where(g_disc, 0, y_to_line([f.y for f in g], mid))
        assigned = {}
        for a in CARR_ALGORITHMS:
            assigned[a] = y_to_line([f.y for f in outputs[a][trial_id]["fixations"]], mid)
        for m in PYEMREAD_METHODS:
            t0 = time.perf_counter()
            assigned[f"pyemread_{m}"] = run_pyemread(m, x, y, reg)
            timing[m] += time.perf_counter() - t0
        kept = ~g_disc
        from pyemread import lines
        rep = lines.line_assignment_report(
            pd.DataFrame({"x_pos": x, "y_pos": y}), reg)
        diag_rows.append({"trial": trial_id, "age_group": trial["age_group"], **rep})
        for method, lab in assigned.items():
            acc_rows.append({
                "trial": trial_id, "participant": trial["participant_id"],
                "age_group": trial["age_group"], "passage": trial["passage_id"],
                "method": method, "n_fix": len(fx), "n_kept": int(kept.sum()),
                "acc_all": 100 * np.mean(lab == g_line),
                "acc_kept": 100 * np.mean(lab[kept] == g_line[kept]),
            })
        for i, f in enumerate(fx):
            row = {"trial": trial_id, "age_group": trial["age_group"],
                   "passage": trial["passage_id"], "fix": i, "x": f.x, "y": f.y,
                   "start": f.start, "end": f.end, "gold": int(g_line[i])}
            row.update({k: int(v[i]) for k, v in assigned.items()})
            fix_rows.append(row)

    acc = pd.DataFrame(acc_rows)
    acc.to_csv(RESULTS / "carr_accuracy.csv", index=False)
    pd.DataFrame(fix_rows).to_csv(RESULTS / "carr_assignments.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(RESULTS / "carr_diagnostics.csv", index=False)
    summ = (acc.groupby(["method", "age_group"])[["acc_all", "acc_kept"]]
            .agg(["mean", "median"]).round(1))
    print(summ.to_string())
    n_trials = acc["trial"].nunique()
    for m, t in timing.items():
        print(f"pyemread_{m}: {1000 * t / n_trials:.0f} ms per trial on average")


if __name__ == "__main__":
    main()
