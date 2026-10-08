# -*- coding: utf-8 -*-
"""
Fixation-to-line assignment for multi-line reading data.

This module assigns each fixation of one trial (one eye) to a line of text
and to a word (area of interest, AOI) defined in a region file. It works
on in-memory pandas DataFrames, so it can be used with fixations from any
source: an EyeLink ASC file parsed by :mod:`pyemread.ext`, fixations
exported by another tracker's software, or events detected from raw
samples by :mod:`pyemread.ext_generic`.

Three assignment methods are provided.

``"slice"`` (default)
    The *slice* algorithm of Glandorf and Schroeder (2021). It splits the
    fixation sequence into runs of nearby fixations, merges runs into
    proto-lines from the longest run outwards, and maps proto-lines to text
    lines in vertical order. This implementation is adapted from the Python
    version by Carr et al. (2022), https://github.com/jwcarr/drift
    (CC BY 4.0). It was the most accurate of the algorithms evaluated on
    Carr et al.'s human-corrected data (see the validation in the paper and
    ``validation/``). The distance thresholds, given in pixels by Carr et al.
    for a line spacing of 64 px and a character width of 16 px, are scaled
    here to the line spacing and character width of the region file.

``"diff"``
    The original pyemread heuristic (Gong & Shuai, 2023). Fixations are
    walked in temporal order. Line changes are counted from horizontal
    displacements: a long leftward move is a return sweep to the next line,
    and a long rightward move from the start of a line is a regression to
    the previous line. It is retained unchanged so that earlier analyses can
    be reproduced. It is insensitive to slow vertical drift, but a long
    within-line regression is counted as a return sweep, which shifts every
    later fixation by one line, and regressions to earlier lines are often
    missed.

``"nearest"``
    Assigns each fixation to the line whose vertical centre is nearest (the
    "attach" baseline of Carr et al., 2022). It is provided as a reference
    and for diagnostics.

Because each method fails in different situations, :func:`line_assignment_report`
compares them, and :func:`pyemread.ext_robust.trial_report` applies the
comparison to every trial. Trials on which the methods disagree should be
inspected visually.

All methods return the input DataFrame with ``line_no`` filled in (1-based
line index as numbered in the region file, ``NaN`` for fixations that were
not assigned). The geometry of the text (line centres, line spacing, line
extents) is read from the region file produced by
:func:`pyemread.gen.praster`, or from any table with the columns
``line_no``, ``x1_pos``, ``x2_pos``, ``y1_pos``, ``y2_pos`` (one row per
word, in reading order).

References
----------
Carr, J. W., Pescuma, V. N., Furlan, M., Ktori, M., & Crepaldi, D. (2022).
Algorithms for the automated correction of vertical drift in eye-tracking
data. *Behavior Research Methods, 54*, 287-310.

Glandorf, D., & Schroeder, S. (2021). Slice: An algorithm to assign
fixations in multi-line texts. *Procedia Computer Science, 192*, 2971-2979.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "TextGeometry",
    "text_geometry",
    "assign_lines",
    "assign_lines_slice",
    "assign_lines_diff",
    "assign_lines_nearest",
    "assign_regions",
    "line_assignment_report",
]


# ---------------------------------------------------------------------------
# Text geometry
# ---------------------------------------------------------------------------

@dataclass
class TextGeometry:
    """Line-level geometry derived from a region (AOI) table.

    Attributes
    ----------
    line_ids : numpy.ndarray
        Line numbers as they appear in the region file (usually 1..K).
    centers : numpy.ndarray
        Vertical centre of each line in pixels (screen coordinates,
        y increasing downwards).
    spacing : float
        Median distance between adjacent line centres in pixels.
    x_first : numpy.ndarray
        Horizontal centre of the first word of each line.
    x_last : numpy.ndarray
        Horizontal centre of the last word of each line.
    width : float
        Median distance between the centre of the last word of a line and
        the centre of the first word of the next line. This is the expected
        horizontal amplitude of a return sweep. It is the quantity used by
        the ``"diff"`` method.
    """

    line_ids: np.ndarray
    centers: np.ndarray
    spacing: float
    x_first: np.ndarray
    x_last: np.ndarray
    width: float


def text_geometry(reg_df: pd.DataFrame) -> TextGeometry:
    """Derive line centres, spacing and extents from a region table.

    Parameters
    ----------
    reg_df : pandas.DataFrame
        One row per word with columns ``line_no``, ``x1_pos``, ``x2_pos``,
        ``y1_pos`` and ``y2_pos``, in reading order.

    Returns
    -------
    TextGeometry
    """
    required = ["line_no", "x1_pos", "x2_pos", "y1_pos", "y2_pos"]
    missing = [c for c in required if c not in reg_df.columns]
    if missing:
        raise KeyError(f"Region table lacks required columns: {missing}")
    reg = reg_df.reset_index(drop=True)
    line_ids = np.array(sorted(reg["line_no"].dropna().unique()))
    centers, x_first, x_last = [], [], []
    for ln in line_ids:
        g = reg[reg["line_no"] == ln]
        centers.append(float(((g["y1_pos"] + g["y2_pos"]) / 2.0).mean()))
        x_first.append(float((g["x1_pos"].iloc[0] + g["x2_pos"].iloc[0]) / 2.0))
        x_last.append(float((g["x1_pos"].iloc[-1] + g["x2_pos"].iloc[-1]) / 2.0))
    centers = np.asarray(centers)
    x_first = np.asarray(x_first)
    x_last = np.asarray(x_last)
    if len(centers) > 1:
        spacing = float(np.median(np.diff(centers)))
        width = float(np.median(x_last[:-1] - x_first[1:]))
    else:
        h = float((reg["y2_pos"] - reg["y1_pos"]).median())
        spacing = max(h * 2.0, 1.0)
        width = float(x_last[0] - x_first[0])
    if width <= 0:
        width = float(reg["x2_pos"].max() - reg["x1_pos"].min())
    return TextGeometry(line_ids, centers, spacing, x_first, x_last, width)


# ---------------------------------------------------------------------------
# Slice (Glandorf & Schroeder, 2021), adapted from Carr et al. (2022)
# ---------------------------------------------------------------------------

def _char_width(reg_df: pd.DataFrame) -> float:
    if 'length' not in reg_df:
        return float((reg_df['x2_pos'] - reg_df['x1_pos']).median() / 5.0)
    w = (reg_df['x2_pos'] - reg_df['x1_pos']) / reg_df['length'].clip(lower=1)
    return float(w.median())


def _slice(xy: np.ndarray, line_y: np.ndarray, x_thresh: float, y_thresh: float,
           w_thresh: float, n_thresh: float) -> np.ndarray:
    """Line index (0-based) of each fixation. Adapted from Carr et al. (2022),
    ``algorithms/Python/drift_algorithms.py``, function ``slice`` (CC BY 4.0).
    Changes: returns line indices instead of modifying the fixation array,
    takes the thresholds as arguments, and handles fewer than two lines."""
    n = len(xy)
    if len(line_y) < 2:
        return np.zeros(n, dtype=int)
    line_height = np.mean(np.diff(line_y))
    proto_lines, phantom = {}, {}
    dist_x = np.abs(np.diff(xy[:, 0]))
    dist_y = np.abs(np.diff(xy[:, 1]))
    ends = list(np.where((dist_x > x_thresh) | (dist_y > y_thresh))[0] + 1)
    runs = [list(range(a, b)) for a, b in zip([0] + ends, ends + [n])]
    longest = int(np.argmax([xy[r[-1], 0] - xy[r[0], 0] for r in runs]))
    proto_lines[0] = runs.pop(longest)

    def diffs(pl_xy, run):
        return np.mean([y - pl_xy[np.argmin(np.abs(pl_xy[:, 0] - x)), 1] for x, y in xy[run]])

    while runs:
        merged = False
        for pl_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
            proto_lines[pl_i + direction] = []
            pl_xy = xy[proto_lines[pl_i]] if proto_lines[pl_i] else phantom[pl_i]
            d = np.array([diffs(pl_xy, r) for r in runs])
            cur = list(np.where(np.abs(d) < w_thresh)[0])
            adj = list(np.where((d * direction >= w_thresh) & (d * direction < n_thresh))[0])
            for k in cur:
                proto_lines[pl_i].extend(runs[k])
            for k in adj:
                proto_lines[pl_i + direction].extend(runs[k])
            if not adj:
                ax, ay = np.mean(pl_xy, axis=0)
                phantom[pl_i + direction] = np.array([[ax, ay + line_height * direction]])
            for k in sorted(cur + adj, reverse=True):
                del runs[k]
                merged = True
        if not merged:
            break
    for run in runs:
        best, best_i = np.inf, None
        for pl_i in proto_lines:
            pl_xy = xy[proto_lines[pl_i]] if proto_lines[pl_i] else phantom[pl_i]
            dist = abs(diffs(pl_xy, run))
            if dist < best:
                best, best_i = dist, pl_i
        proto_lines[best_i].extend(run)
    while len(proto_lines) > len(line_y):
        top, bot = min(proto_lines), max(proto_lines)
        if len(proto_lines[top]) < len(proto_lines[bot]):
            proto_lines[top + 1].extend(proto_lines[top])
            del proto_lines[top]
        else:
            proto_lines[bot - 1].extend(proto_lines[bot])
            del proto_lines[bot]
    out = np.zeros(n, dtype=int)
    for line_i, pl_i in enumerate(sorted(proto_lines)):
        out[proto_lines[pl_i]] = line_i
    return out


def assign_lines_slice(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                       x_thresh: Optional[float] = None, y_thresh: Optional[float] = None,
                       w_thresh: Optional[float] = None, n_thresh: Optional[float] = None,
                       x_col: str = "x_pos", y_col: str = "y_pos") -> pd.DataFrame:
    """Assign fixations of one trial (one eye) to lines with the *slice* algorithm.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations in temporal order, with positions in screen pixels.
    reg_df : pandas.DataFrame
        Region (AOI) table of the stimulus; see :func:`text_geometry`.
    x_thresh, y_thresh : float, optional
        Horizontal and vertical distances (px) between consecutive fixations
        that end a run. Defaults: 12 character widths and half the line
        spacing (192 px and 32 px in Carr et al.'s data).
    w_thresh, n_thresh : float, optional
        A run whose mean vertical distance from a proto-line is below
        ``w_thresh`` is merged into it; one between ``w_thresh`` and
        ``n_thresh`` is merged into the adjacent proto-line. Defaults: 0.5
        and 1.4 line spacings (32 px and 90 px in Carr et al.'s data).

    Returns
    -------
    pandas.DataFrame
        A copy of ``fix_df`` with ``line_no``.
    """
    geo = text_geometry(reg_df)
    L, cw = geo.spacing, _char_width(reg_df)
    x_thresh = 12.0 * cw if x_thresh is None else x_thresh
    y_thresh = 0.5 * L if y_thresh is None else y_thresh
    w_thresh = 0.5 * L if w_thresh is None else w_thresh
    n_thresh = (90.0 / 64.0) * L if n_thresh is None else n_thresh
    out = fix_df.copy()
    out["line_no"] = np.nan
    ok = np.isfinite(out[x_col].to_numpy(float)) & np.isfinite(out[y_col].to_numpy(float))
    if ok.sum() == 0:
        return out
    xy = np.column_stack([out.loc[ok, x_col].to_numpy(float), out.loc[ok, y_col].to_numpy(float)])
    idx = _slice(xy, geo.centers, x_thresh, y_thresh, w_thresh, n_thresh)
    out.loc[ok, "line_no"] = geo.line_ids[idx].astype(float)
    return out


# ---------------------------------------------------------------------------
# Original heuristic ("diff") and nearest-line baseline
# ---------------------------------------------------------------------------

def assign_lines_diff(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                      diff_ratio: float = 0.6, frontrange_ratio: float = 0.2,
                      y_range: float = 60, x_col: str = "x_pos",
                      y_col: str = "y_pos") -> pd.DataFrame:
    """Assign fixations of one trial (one eye) to lines with the original heuristic.

    This calls the unchanged implementation in :mod:`pyemread.ext`
    (``classify_method="DIFF"``). Fixations after the detected end of
    reading (wrap-up) are left unassigned (``NaN``), as in earlier
    versions.

    Parameters
    ----------
    fix_df, reg_df : pandas.DataFrame
        As in :func:`assign_lines_slice`.
    diff_ratio : float
        Fraction of the return-sweep amplitude that a horizontal move must
        exceed to count as a line change (default 0.6).
    frontrange_ratio : float
        Fraction of the line treated as its front when testing for a
        regression to the previous line (default 0.2).
    y_range : float
        Vertical distance in pixels used to detect the end of reading
        (default 60).

    Returns
    -------
    pandas.DataFrame
        A copy of ``fix_df`` with ``line_no`` filled in.
    """
    from . import ext as _ext
    out = fix_df.copy()
    out["line_no"] = np.nan
    ok = np.isfinite(out[x_col].to_numpy(float)) & np.isfinite(out[y_col].to_numpy(float))
    sub = out.loc[ok].reset_index()
    if len(sub) < 2:
        return out
    work = pd.DataFrame({
        "eye": "R",
        "x_pos": sub[x_col].astype(float).to_numpy(),
        "y_pos": sub[y_col].astype(float).to_numpy(),
        "line_no": np.nan,
    })
    reg = reg_df.reset_index(drop=True)
    if reg["line_no"].nunique() < 2:
        out.loc[ok, "line_no"] = float(reg["line_no"].iloc[0])
        return out
    try:
        _ext._get_fix_line(reg, pd.DataFrame(), work, "DIFF", diff_ratio,
                           frontrange_ratio, y_range)
    except IndexError:
        # the heuristic found no line change at all
        return out
    out.loc[sub["index"].to_numpy(), "line_no"] = work["line_no"].to_numpy()
    return out


def assign_lines_nearest(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                         x_col: str = "x_pos", y_col: str = "y_pos") -> pd.DataFrame:
    """Assign each fixation to the line with the nearest vertical centre."""
    geo = text_geometry(reg_df)
    out = fix_df.copy()
    y = out[y_col].to_numpy(float)
    ok = np.isfinite(y)
    idx = np.argmin(np.abs(y[ok, None] - geo.centers[None, :]), axis=1)
    out["line_no"] = np.nan
    out.loc[ok, "line_no"] = geo.line_ids[idx].astype(float)
    return out


_METHODS = {"slice": assign_lines_slice, "diff": assign_lines_diff,
            "nearest": assign_lines_nearest}


def assign_lines(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                 method: str = "slice", **kwargs) -> pd.DataFrame:
    """Assign fixations of one trial (one eye) to text lines.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations of a single trial and a single eye, in temporal order.
    reg_df : pandas.DataFrame
        Region table of the stimulus shown on that trial.
    method : {"slice", "diff", "nearest"}
        Assignment method (see the module documentation).
    **kwargs
        Passed to the method-specific function.

    Returns
    -------
    pandas.DataFrame
        A copy of ``fix_df`` with ``line_no``.
    """
    key = method.lower()
    if key not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {sorted(_METHODS)}")
    return _METHODS[key](fix_df, reg_df, **kwargs)


# ---------------------------------------------------------------------------
# Word (region) assignment
# ---------------------------------------------------------------------------

def _padded_bounds(reg_df: pd.DataFrame, add_char_sp: float) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal word bounds, widened at line edges to catch overshoots."""
    reg = reg_df.reset_index(drop=True)
    x1 = reg["x1_pos"].to_numpy(float).copy()
    x2 = reg["x2_pos"].to_numpy(float).copy()
    char_w = (x2[0] - x1[0]) / max(float(reg["length"].iloc[0]), 1.0) if "length" in reg else 0.0
    pad = add_char_sp * char_w
    ln = reg["line_no"].to_numpy()
    first = np.r_[True, ln[1:] != ln[:-1]]
    last = np.r_[ln[1:] != ln[:-1], True]
    x1[first] -= pad
    x2[last] += pad
    return x1, x2


def assign_regions(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                   add_char_sp: float = 1, x_col: str = "x_pos") -> pd.DataFrame:
    """Assign line-classified fixations to words.

    A fixation is assigned to the word on its assigned line whose
    horizontal extent contains the fixation. The first and last words of
    each line are widened by ``add_char_sp`` character widths to catch
    fixations that overshoot the line edges.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations with ``line_no`` already assigned.
    reg_df : pandas.DataFrame
        Region table with ``WordID``, ``line_no``, ``x1_pos``, ``x2_pos``
        and ``length``.
    add_char_sp : float
        Number of character widths added at line edges (default 1).

    Returns
    -------
    pandas.DataFrame
        A copy of ``fix_df`` with ``region_no`` (the ``WordID`` of the
        fixated word, ``NaN`` if none).
    """
    reg = reg_df.reset_index(drop=True)
    x1, x2 = _padded_bounds(reg, add_char_sp)
    ln = reg["line_no"].to_numpy(float)
    wid = reg["WordID"].to_numpy(float)
    out = fix_df.copy()
    xs = out[x_col].to_numpy(float)
    ls = out["line_no"].to_numpy(float)
    res = np.full(len(out), np.nan)
    for i, (xv, lv) in enumerate(zip(xs, ls)):
        if not (np.isfinite(xv) and np.isfinite(lv)):
            continue
        hit = np.where((ln == lv) & (x1 <= xv) & (xv <= x2))[0]
        if len(hit) >= 1:
            res[i] = wid[hit[0]]
    out["region_no"] = res
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def line_assignment_report(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                           threshold: float = 0.9, x_col: str = "x_pos",
                           y_col: str = "y_pos") -> Dict[str, float]:
    """Compare the line assignments of the three methods for one trial.

    The ``"slice"`` and ``"diff"`` methods rest on different principles
    (grouping fixations by vertical position versus counting line changes
    in temporal order), and they tend to fail on different trials. A trial
    on which they disagree on many fixations should be inspected visually.

    Parameters
    ----------
    fix_df, reg_df : pandas.DataFrame
        Fixations of one trial (one eye) and the region table.
    threshold : float
        Minimum proportion of fixations on which ``"slice"`` and ``"diff"``
        must agree for the trial not to be flagged (default 0.9).

    Returns
    -------
    dict
        ``n_fixations``, ``agree_slice_diff``, ``agree_slice_nearest``,
        ``agree_diff_nearest``, ``n_lines_slice`` and ``n_lines_diff``
        (number of text lines that received at least one fixation),
        ``n_lines`` (number of text lines) and ``flag``
        (``agree_slice_diff < threshold``).
    """
    a = assign_lines_slice(fix_df, reg_df, x_col=x_col, y_col=y_col)["line_no"].to_numpy(float)
    b = assign_lines_diff(fix_df, reg_df, x_col=x_col, y_col=y_col)["line_no"].to_numpy(float)
    c = assign_lines_nearest(fix_df, reg_df, x_col=x_col, y_col=y_col)["line_no"].to_numpy(float)
    ok = np.isfinite(a)
    agree = lambda u, v: float(np.mean(u[ok] == v[ok])) if ok.any() else np.nan
    ab = agree(a, b)
    return {"n_fixations": int(ok.sum()), "agree_slice_diff": ab,
            "agree_slice_nearest": agree(a, c), "agree_diff_nearest": agree(b, c),
            "n_lines_slice": int(len(np.unique(a[ok]))),
            "n_lines_diff": int(len(np.unique(b[np.isfinite(b)]))),
            "n_lines": int(len(text_geometry(reg_df).centers)),
            "flag": bool(ab < threshold)}
