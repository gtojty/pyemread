# -*- coding: utf-8 -*-
"""
Diagnostics for line assignment: trials to inspect and pre-reading scans.

No automatic line assignment is error-free. This module helps users find
the trials in which it probably went wrong, so that they can be inspected
and, if needed, corrected by hand.

* :func:`trial_report` compares, per trial and eye, the ``"slice"`` and
  ``"diff"`` assignments (and nearest-line assignment). The two methods
  rest on different principles and tend to fail on different trials. In the
  validation reported with this package, every trial on which either
  method assigned fewer than 90% of fixations correctly was flagged.
  Flagged trials should be inspected visually (e.g. with
  :func:`pyemread.gen.draw_line_assignment`).
* :func:`detect_prescan` and :func:`mask_prescan_fixations` mark an initial
  scan of the page that precedes line-by-line reading, so that it does not
  enter first-pass measures.

Examples
--------
>>> from pyemread import ext, ext_robust as er
>>> sac, crl_sac, fix, crl_fix = ext.process_subject("data", "1950138",
...     ["story01.region.csv"])                            # doctest: +SKIP
>>> er.trial_report(fix, ext.load_regions("data", ["story01.region.csv"]))  # doctest: +SKIP
"""

from __future__ import annotations

__author__ = "Tao Gong"
__license__ = "MIT"

from typing import Dict

import numpy as np
import pandas as pd

from . import lines as _lines

__all__ = ["trial_report", "detect_prescan", "mask_prescan_fixations"]


def trial_report(fix_df: pd.DataFrame, regions: Dict[str, pd.DataFrame],
                 threshold: float = 0.9) -> pd.DataFrame:
    """Line-assignment diagnostics for every trial and eye.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixation table (pyemread schema) with ``trial_id``, ``trial_type``,
        ``eye``, ``x_pos`` and ``y_pos``. Only valid fixations are used.
    regions : dict
        ``{trial_type: region DataFrame}``, e.g. from
        :func:`pyemread.ext.load_regions`.
    threshold : float
        Minimum agreement between the ``"slice"`` and ``"diff"`` assignments
        for a trial not to be flagged (default 0.9).

    Returns
    -------
    pandas.DataFrame
        One row per trial and eye with the fields of
        :func:`pyemread.lines.line_assignment_report`.

    Raises
    ------
    KeyError
        If a trial's stimulus (``trial_type``) is not in ``regions``.
    """
    rows = []
    fx = fix_df
    if 'valid' in fx:
        fx = fx[fx['valid'].astype(str).str.lower().isin(['yes', '1', 'true', '1.0'])]
    for (tr, eye), g in fx.groupby(['trial_id', 'eye'], sort=False):
        tt = str(g['trial_type'].iloc[0])
        rep = _lines.line_assignment_report(g.reset_index(drop=True), regions[tt],
                                            threshold=threshold)
        rows.append({'trial_id': tr, 'trial_type': tt, 'eye': eye, **rep})
    return pd.DataFrame(rows)


def detect_prescan(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                   min_scan_fixations: int = 3, settle_run: int = 3) -> pd.Series:
    """Mark fixations of an initial scan that precedes reading of the first line.

    For each trial and eye, reading is taken to start with the first run of
    ``settle_run`` consecutive fixations that lie within 0.7 line spacings
    of the first line and progress rightwards. The fixations before that
    run are a pre-scan if there are at least ``min_scan_fixations`` of them
    and at least one lies more than 1.5 line spacings below the first line.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations (``trial_id``, ``eye``, ``x_pos``, ``y_pos``), in temporal
        order within each trial.
    reg_df : pandas.DataFrame
        Region table of the stimulus.

    Returns
    -------
    pandas.Series of bool
        Aligned with ``fix_df``; True for pre-scan fixations.
    """
    geo = _lines.text_geometry(reg_df)
    first, L = geo.centers[0], geo.spacing
    mask = pd.Series(False, index=fix_df.index)
    for _, g in fix_df.groupby(['trial_id', 'eye'], sort=False):
        y = g['y_pos'].to_numpy(float)
        x = g['x_pos'].to_numpy(float)
        start = None
        for k in range(len(g) - settle_run + 1):
            yy, xx = y[k:k + settle_run], x[k:k + settle_run]
            if np.all(np.abs(yy - first) < 0.7 * L) and np.all(np.diff(xx) > -0.1 * geo.width):
                start = k
                break
        if start is None or start < min_scan_fixations:
            continue
        if np.nanmax(y[:start]) - first > 1.5 * L:
            mask.loc[g.index[:start]] = True
    return mask


def mask_prescan_fixations(fix_df: pd.DataFrame, reg_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return a copy of ``fix_df`` with pre-scan fixations marked ``valid = 'no'``."""
    out = fix_df.copy()
    if 'valid' not in out:
        out['valid'] = 'yes'
    out.loc[detect_prescan(out, reg_df, **kwargs), 'valid'] = 'no'
    return out
