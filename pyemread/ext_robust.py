# -*- coding: utf-8 -*-
"""
Diagnostics for line assignment: trials to inspect, pre-reading scans and
parameter sensitivity.

No automatic line assignment is error-free. This module helps users find
the trials in which it probably went wrong, so that they can be inspected
and, if needed, corrected by hand.

* :func:`trial_report` compares, per trial and eye, the ``"slice"`` and
  ``"diff"`` assignments (and nearest-line assignment). The two methods
  rest on different principles and tend to fail on different trials. In the
  validation reported with this package, every trial on which either
  method assigned fewer than 90% of fixations correctly was flagged.
  Flagged trials should be inspected visually (e.g. with
  :func:`pyemread.gen.draw_sac_fix`).
* :func:`detect_prescan` and :func:`mask_prescan_fixations` mark an initial
  scan of the page that precedes line-by-line reading, so that it does not
  enter first-pass measures.
* :func:`sensitivity_report` re-runs an assignment over a parameter grid
  and reports how much the result changes. Stability is not accuracy:
  a stable assignment can still be wrong.

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

from itertools import product
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from . import lines as _lines

__all__ = ["trial_report", "detect_prescan", "mask_prescan_fixations", "sensitivity_report"]


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


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return float('nan')
    po = float(np.mean(a == b))
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in np.unique(np.r_[a, b]))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def sensitivity_report(classify_fn: Callable[..., Iterable],
                       grid: Dict[str, Iterable[float]] = None,
                       reference: Dict[str, float] = None) -> pd.DataFrame:
    """Re-run a line assignment over a parameter grid and compare with a reference.

    Parameters
    ----------
    classify_fn : callable
        Called with keyword arguments from ``grid``; must return one line
        number per fixation (e.g. a closure around
        :func:`pyemread.lines.assign_lines`).
    grid : dict
        Parameter names mapped to the values to try. Default:
        ``{"diff_ratio": (0.4, 0.5, 0.6, 0.7, 0.8),
        "frontrange_ratio": (0.1, 0.2, 0.3)}``.
    reference : dict
        Parameter values of the reference assignment. Default: the package
        defaults of the ``"diff"`` method (0.6 and 0.2).

    Returns
    -------
    pandas.DataFrame
        One row per parameter combination with ``kappa_vs_reference`` and
        ``pct_agreement``.

    Notes
    -----
    Agreement with a reference measures stability, not accuracy.
    """
    grid = grid or {'diff_ratio': (0.4, 0.5, 0.6, 0.7, 0.8), 'frontrange_ratio': (0.1, 0.2, 0.3)}
    reference = reference or {'diff_ratio': 0.6, 'frontrange_ratio': 0.2}
    ref = np.nan_to_num(np.asarray(classify_fn(**reference), float), nan=0)
    rows = []
    keys = list(grid)
    for vals in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        out = np.nan_to_num(np.asarray(classify_fn(**params), float), nan=0)
        rows.append({**params, 'kappa_vs_reference': _cohens_kappa(ref, out),
                     'pct_agreement': 100 * float(np.mean(ref == out))})
    return pd.DataFrame(rows)
