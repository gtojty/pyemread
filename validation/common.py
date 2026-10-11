"""Shared helpers for the line-assignment validation.

The validation uses the public data and code of Carr et al. (2022),
"Algorithms for the automated correction of vertical drift in
eye-tracking data", Behavior Research Methods, 54, 287-310,
https://github.com/jwcarr/drift (CC BY 4.0), commit
5b4b6c475b5118950514dc01960391ef0d95bd19.

Set the environment variable ``DRIFT_REPO`` to the path of a clone of that
repository (default: ``../drift`` next to the pyemread repository).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

DRIFT = Path(os.environ.get("DRIFT_REPO", REPO.parent / "drift")).resolve()
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

CARR_ALGORITHMS = ["attach", "chain", "cluster", "compare", "merge", "regress",
                   "segment", "slice", "split", "stretch", "warp"]
PYEMREAD_METHODS = ["diff", "slice"]


def passage_to_region(passage) -> pd.DataFrame:
    """Convert an eyekit TextBlock into a pyemread region table."""
    rows, wid = [], 1
    for line_n in range(passage.n_rows):
        for w in passage.words(line_n=line_n):
            # eyekit word boxes extend half a character beyond each end of the
            # word, so a box spans len(text) + 1 characters (as a pyemread
            # region, which includes the preceding space, does)
            rows.append({"WordID": wid, "Word": w.text, "length": len(w.text) + 1,
                         "line_no": line_n + 1, "x1_pos": float(w.x_tl),
                         "y1_pos": float(w.y_tl), "x2_pos": float(w.x_br),
                         "y2_pos": float(w.y_br)})
            wid += 1
    return pd.DataFrame(rows)


def y_to_line(y: np.ndarray, midlines: np.ndarray) -> np.ndarray:
    """1-based index of the nearest line midline."""
    return np.argmin(np.abs(np.asarray(y, float)[:, None] - midlines[None, :]), axis=1) + 1


def run_pyemread(method: str, x: np.ndarray, y: np.ndarray, reg: pd.DataFrame) -> np.ndarray:
    """Line assignment (1-based; 0 = unassigned) for one trial."""
    from pyemread import lines
    fix = pd.DataFrame({"x_pos": np.asarray(x, float), "y_pos": np.asarray(y, float)})
    out = lines.assign_lines(fix, reg, method=method)
    return np.nan_to_num(out["line_no"].to_numpy(float), nan=0).astype(int)


def install_carr_algorithms():
    """Import Carr et al.'s algorithm implementations (code/algorithms.py)."""
    sys.path.insert(0, str(DRIFT / "code"))
    import algorithms  # noqa: E402
    return algorithms


def word_centers(passage):
    return [w.center for w in passage.words()]
