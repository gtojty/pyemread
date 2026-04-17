# -*- coding: utf-8 -*-
"""
ext_robust.py  —  Parameter tuning and edge-case handling for pyemread's
cross-line saccade/fixation classifier.

This module wraps ``ext.cal_crlSacFix`` with three additions motivated
by the BRM Reviewer #1 feedback:

1. ``recommend_diff_ratio`` — automatic suggestion of the
   ``diff_ratio`` and ``frontrange_ratio`` parameters from a small
   calibration set, replacing the fixed defaults (0.6 / 0.2) that are
   sometimes too aggressive for short-line layouts or noisy data.

2. ``detect_prescan`` / ``mask_prescan_fixations`` — explicit
   identification of whole-passage scanning that sometimes precedes
   line-by-line reading.  Fixations flagged as pre-scan receive
   ``valid=0`` and are excluded from first-/second-pass classification.

3. ``reclassify_orphans`` — post-hoc re-assignment of fixations that
   the heuristic classifier placed far from the matched line's
   baseline.  A fixation is considered orphaned if its vertical
   distance from the mean ``y`` of its assigned line exceeds
   ``line_tolerance`` (default: one line-space).

4. ``sensitivity_report`` — re-runs classification over a grid of
   (diff_ratio × frontrange_ratio) values and reports agreement
   (Cohen's κ) with the default, letting the user quantify how
   parameter-sensitive their own data are.

These functions operate on the pandas DataFrames that
``ext.cal_crlSacFix`` already writes — no changes to the upstream
pipeline are required.

Author: Tao Gong
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

__all__ = [
    "recommend_diff_ratio",
    "detect_prescan",
    "mask_prescan_fixations",
    "reclassify_orphans",
    "sensitivity_report",
]


# ---------------------------------------------------------------------------
# I. Parameter recommendation
# ---------------------------------------------------------------------------

def recommend_diff_ratio(reg_df: pd.DataFrame,
                         FixDF: pd.DataFrame) -> dict:
    """Recommend ``diff_ratio`` and ``frontrange_ratio`` for a trial.

    The heuristic is simple: we compute the distribution of
    between-fixation horizontal displacements, locate the mode of the
    *large-jump* population (right-tail), and set the threshold to a
    value that separates it from the within-line mode.

    Parameters
    ----------
    reg_df : DataFrame
        Region file of the stimulus (one row per word; must have
        ``x1_pos``, ``x2_pos``, ``line_no``).
    FixDF : DataFrame
        Fixation DataFrame with ``x`` and ``trial_id`` columns.

    Returns
    -------
    dict with keys ``diff_ratio``, ``frontrange_ratio``,
    ``line_extent_px``, ``n_fixations``, ``note``.
    """
    # Line horizontal extent = first-word-centre → last-word-centre
    extents = []
    for _, g in reg_df.groupby('line_no'):
        x_first = 0.5 * (g['x1_pos'].iloc[0] + g['x2_pos'].iloc[0])
        x_last = 0.5 * (g['x1_pos'].iloc[-1] + g['x2_pos'].iloc[-1])
        extents.append(abs(x_last - x_first))
    line_extent = float(np.mean(extents)) if extents else 1.0

    dx = np.diff(FixDF.sort_values('start')['x'].to_numpy())
    abs_dx = np.abs(dx)
    if len(abs_dx) < 5 or line_extent <= 0:
        return {'diff_ratio': 0.6, 'frontrange_ratio': 0.2,
                'line_extent_px': line_extent,
                'n_fixations': int(len(FixDF)),
                'note': 'too few fixations; using defaults'}

    # Separator = midpoint between 75th and 95th percentile of |dx|,
    # expressed as a fraction of the line extent, clipped into [0.4, 0.8]
    p75 = np.percentile(abs_dx, 75)
    p95 = np.percentile(abs_dx, 95)
    sep = 0.5 * (p75 + p95)
    diff_ratio = float(np.clip(sep / line_extent, 0.4, 0.8))

    # frontrange_ratio — front 15–30 % of a line, based on the ratio of
    # observed "start of line" fixations (low x) to total fixations
    low_x = (FixDF['x'] - FixDF['x'].min()) / max(line_extent, 1.0)
    pct_front = float(np.mean(low_x < 0.25))
    frontrange = float(np.clip(0.15 + 0.5 * pct_front, 0.15, 0.30))

    return {'diff_ratio': round(diff_ratio, 2),
            'frontrange_ratio': round(frontrange, 2),
            'line_extent_px': line_extent,
            'n_fixations': int(len(FixDF)),
            'note': 'ok'}


# ---------------------------------------------------------------------------
# II. Pre-scan detection
# ---------------------------------------------------------------------------

def detect_prescan(FixDF: pd.DataFrame,
                   reg_df: pd.DataFrame,
                   min_scan_fixations: int = 4,
                   y_tolerance: float = 1.5) -> pd.Series:
    """Return a boolean mask marking fixations that form a pre-scan.

    A pre-scan is defined as the contiguous block of fixations starting
    at t=0 that (a) contains at least ``min_scan_fixations`` fixations
    and (b) spans more than ``y_tolerance × linespace`` vertically.
    The block ends at the last fixation before the reader settles on
    line 1 for a sustained (≥3-fixation) run.

    Parameters
    ----------
    FixDF : DataFrame
        Must contain ``start``, ``x``, ``y``, ``trial_id``.
    reg_df : DataFrame
        Region file; used to estimate line-space.
    min_scan_fixations : int
        Fewer fixations than this → no pre-scan.
    y_tolerance : float
        Vertical span (in line-spaces) above which the block is
        considered a scan.

    Returns
    -------
    pandas.Series[bool]
        Index-aligned with ``FixDF``; True where fixation is pre-scan.
    """
    # Line-space estimate = distance between consecutive baselines
    if 'baseline' in reg_df.columns:
        baselines = np.sort(reg_df['baseline'].unique())
    else:
        baselines = np.sort(reg_df.groupby('line_no')['y1_pos'].mean().values)
    if len(baselines) >= 2:
        linespace = float(np.mean(np.diff(baselines)))
    else:
        linespace = 65.0                                   # reasonable default

    mask = pd.Series(False, index=FixDF.index)
    for tr in FixDF['trial_id'].unique():
        idx = FixDF.index[FixDF['trial_id'] == tr]
        tr_fix = FixDF.loc[idx].sort_values('start')
        if len(tr_fix) < min_scan_fixations:
            continue
        line1_y = baselines[0] if len(baselines) else tr_fix['y'].iloc[0]
        # scan = everything before the reader settles on line 1
        settled_idx = None
        window = 3
        for k in range(len(tr_fix) - window + 1):
            chunk = tr_fix.iloc[k:k + window]
            if np.all(np.abs(chunk['y'].to_numpy() - line1_y) < linespace * 0.7):
                settled_idx = tr_fix.index[k]
                break
        if settled_idx is None:
            continue
        scan_idx = tr_fix.loc[:settled_idx].index[:-1]     # exclude the settle row
        if len(scan_idx) < min_scan_fixations:
            continue
        y_span = tr_fix.loc[scan_idx, 'y'].max() - tr_fix.loc[scan_idx, 'y'].min()
        if y_span > y_tolerance * linespace:
            mask.loc[scan_idx] = True
    return mask


def mask_prescan_fixations(FixDF: pd.DataFrame,
                           reg_df: pd.DataFrame,
                           **kwargs) -> pd.DataFrame:
    """Return a copy of ``FixDF`` with ``valid=0`` on pre-scan fixations."""
    out = FixDF.copy()
    if 'valid' not in out.columns:
        out['valid'] = 1
    mask = detect_prescan(out, reg_df, **kwargs)
    out.loc[mask, 'valid'] = 0
    return out


# ---------------------------------------------------------------------------
# III. Orphan-fixation reclassification
# ---------------------------------------------------------------------------

def reclassify_orphans(FixDF: pd.DataFrame,
                       reg_df: pd.DataFrame,
                       line_tolerance: float | None = None) -> pd.DataFrame:
    """Re-assign fixations whose y is far from their line's baseline.

    Parameters
    ----------
    FixDF : DataFrame
        Must contain ``y``, ``line_no``.
    reg_df : DataFrame
        Region file.
    line_tolerance : float, optional
        Pixels.  Fixations farther than this from their line's mean y
        are re-assigned to the nearest line.  Defaults to one
        line-space.

    Returns
    -------
    A copy of ``FixDF`` with possibly-updated ``line_no`` and a new
    boolean column ``reclassified``.
    """
    if 'baseline' in reg_df.columns:
        baselines = {int(ln): float(reg_df.loc[reg_df['line_no'] == ln,
                                               'baseline'].iloc[0])
                     for ln in reg_df['line_no'].unique()}
    else:
        baselines = {int(ln): float(g['y1_pos'].mean())
                     for ln, g in reg_df.groupby('line_no')}

    if line_tolerance is None:
        ys = np.sort(list(baselines.values()))
        line_tolerance = float(np.mean(np.diff(ys))) if len(ys) >= 2 else 65.0

    out = FixDF.copy()
    out['reclassified'] = False
    for i, row in out.iterrows():
        ln = row.get('line_no', -1)
        if ln not in baselines or np.isnan(row['y']):
            continue
        y_err = abs(row['y'] - baselines[ln])
        if y_err > line_tolerance:
            # Find the closest line
            best = min(baselines.items(), key=lambda kv: abs(row['y'] - kv[1]))
            if best[0] != ln:
                out.at[i, 'line_no'] = best[0]
                out.at[i, 'reclassified'] = True
    return out


# ---------------------------------------------------------------------------
# IV. Parameter-sensitivity report
# ---------------------------------------------------------------------------

def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Simple implementation of Cohen's κ for two 1-D integer arrays."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return float('nan')
    po = float(np.mean(a == b))
    cats = np.unique(np.concatenate([a, b]))
    pe = 0.0
    n = len(a)
    for c in cats:
        pe += (np.mean(a == c) * np.mean(b == c))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def sensitivity_report(classify_fn,
                       grid_diff_ratio=(0.4, 0.5, 0.6, 0.7, 0.8),
                       grid_frontrange=(0.15, 0.20, 0.25, 0.30),
                       reference=(0.6, 0.2)) -> pd.DataFrame:
    """Run the classifier over a parameter grid and report agreement.

    Parameters
    ----------
    classify_fn : callable
        ``classify_fn(diff_ratio, frontrange_ratio)`` must return a
        vector of line assignments (one per fixation).  The user
        supplies this closure so the report is agnostic to whether
        classification is in-memory or disk-based.
    grid_diff_ratio, grid_frontrange : iterable of float
    reference : (float, float)
        Parameters used for the reference solution.

    Returns
    -------
    DataFrame with columns ``diff_ratio``, ``frontrange_ratio``,
    ``kappa_vs_reference``, ``pct_agreement``.
    """
    ref = np.asarray(classify_fn(*reference))
    rows = []
    for dr, fr in product(grid_diff_ratio, grid_frontrange):
        out = np.asarray(classify_fn(dr, fr))
        if len(out) != len(ref):
            rows.append({'diff_ratio': dr, 'frontrange_ratio': fr,
                         'kappa_vs_reference': float('nan'),
                         'pct_agreement': float('nan')})
            continue
        rows.append({'diff_ratio': dr, 'frontrange_ratio': fr,
                     'kappa_vs_reference': _cohens_kappa(ref, out),
                     'pct_agreement': float(np.mean(ref == out))})
    return pd.DataFrame(rows)
