# -*- coding: utf-8 -*-
"""
ext_generic.py  —  Tracker-agnostic raw-gaze processing for pyemread.

This module extends pyemread with support for raw gaze samples of the
form [t, x, y] produced by Tobii, SMI, Pupil Labs, EyeTribe, GazePoint,
webcam-based trackers (e.g. WebGazer), or any device that outputs
sample-level gaze coordinates.  The output is a pair of DataFrames
(saccades, fixations) whose columns match those produced by
``ext.read_SRRasc`` / ``ext.read_TimeStamp`` for EyeLink ASC files, so
the downstream ``cal_crlSacFix`` classifier and ``cal.cal_write_EM``
metric calculator can be used without modification.

Typical usage
-------------
    from pyemread import ext_generic as extg, ext, cal

    # 1. Load raw samples from any tracker into a pandas DataFrame
    #    with at least the columns ['time', 'x', 'y'].  Optional columns
    #    'eye' ('L'/'R') and 'trial_id' are respected when present.
    samples = extg.read_raw_csv('Tobii_export.csv',
                                time_col='device_time_stamp',
                                x_col='left_gaze_x',
                                y_col='left_gaze_y',
                                time_unit='us')

    # 2. Detect fixations and saccades with an I-VT or I-DT algorithm
    SacDF, FixDF = extg.detect_events(samples, method='IVT',
                                      velocity_threshold=30.0,
                                      min_fix_duration=50)

    # 3. Hand off to the standard pyemread pipeline
    SacDF, crlSac, FixDF, crlFix = ext.cal_crlSacFix(
        direct='.', subjID='P01',
        regfileNameList=['story01.region.csv'],
        ExpType='RAW',
        SacDF_in=SacDF, FixDF_in=FixDF)    # see note below

    cal.cal_write_EM(direct='.', subjID='P01',
                     regfileNameList=['story01.region.csv'])

Note
----
``ext.cal_crlSacFix`` currently reads CSV files from disk.  A thin
convenience wrapper ``classify_lines`` is provided below that works
directly with in-memory DataFrames.

References
----------
* Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and
  saccades in eye-tracking protocols. In *Proc. ETRA 2000* (pp. 71–78).
* Olsson, P. (2007). Real-time and offline filters for eye tracking.
  MSc thesis, KTH.

Author: Tao Gong
License: MIT
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

__all__ = [
    "read_raw_csv",
    "read_raw_tsv",
    "detect_events",
    "classify_lines",
    "write_events",
    "tobii_to_raw",
    "pupil_labs_to_raw",
    "smi_to_raw",
    "webgazer_to_raw",
]

# ---------------------------------------------------------------------------
# I. Raw-sample loaders
# ---------------------------------------------------------------------------

def _coerce_time_to_ms(series: pd.Series, unit: str) -> pd.Series:
    """Return a time series in milliseconds (float, monotone-increasing)."""
    unit = unit.lower()
    if unit in ('ms', 'millisecond', 'milliseconds'):
        t = series.astype(float)
    elif unit in ('s', 'second', 'seconds', 'sec'):
        t = series.astype(float) * 1000.0
    elif unit in ('us', 'microsecond', 'microseconds', 'usec'):
        t = series.astype(float) / 1000.0
    elif unit in ('ns', 'nanosecond', 'nanoseconds'):
        t = series.astype(float) / 1.0e6
    else:
        raise ValueError(f"Unrecognised time unit: {unit!r}. "
                         "Use 'ms', 's', 'us' or 'ns'.")
    # Normalise to start at 0 (EyeLink convention after alignment)
    t = t - t.iloc[0]
    return t


def read_raw_csv(path: str,
                 time_col: str = 'time',
                 x_col: str = 'x',
                 y_col: str = 'y',
                 eye_col: str | None = None,
                 trial_col: str | None = None,
                 time_unit: str = 'ms',
                 sep: str = ',',
                 encoding: str = 'utf-8') -> pd.DataFrame:
    """Read raw gaze samples from a CSV/TSV file.

    Parameters
    ----------
    path : str
        File path.
    time_col, x_col, y_col : str
        Column names holding sample time, x-pixel and y-pixel.
    eye_col : str, optional
        Column with 'L'/'R' labels if binocular.  Absent column → 'R'.
    trial_col : str, optional
        Column with an integer trial index.  Absent column → single trial.
    time_unit : {'ms', 's', 'us', 'ns'}
        Unit of the time column; converted to milliseconds internally.
    sep : str
        Field separator (',' or '\\t').
    encoding : str
        File encoding.

    Returns
    -------
    DataFrame with columns ``['time', 'x', 'y', 'eye', 'trial_id']``.
    """
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    missing = [c for c in (time_col, x_col, y_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in file: {missing}")

    out = pd.DataFrame({
        'time': _coerce_time_to_ms(df[time_col], time_unit),
        'x':    df[x_col].astype(float),
        'y':    df[y_col].astype(float),
    })
    out['eye'] = (df[eye_col].astype(str).str.upper().str[0]
                  if eye_col else 'R')
    out['trial_id'] = (df[trial_col].astype(int)
                       if trial_col else 1)
    return out


def read_raw_tsv(path: str, **kwargs) -> pd.DataFrame:
    """Convenience wrapper for tab-separated raw-sample files."""
    kwargs.setdefault('sep', '\t')
    return read_raw_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# II. Event detection: I-VT (velocity-threshold) and I-DT (dispersion)
# ---------------------------------------------------------------------------

def _smooth_xy(x: np.ndarray, y: np.ndarray, win: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving-average smoothing of raw gaze (window in samples)."""
    if win <= 1:
        return x, y
    kernel = np.ones(win, dtype=float) / win
    xs = np.convolve(x, kernel, mode='same')
    ys = np.convolve(y, kernel, mode='same')
    return xs, ys


def _ivt_detect(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                velocity_threshold: float,
                min_fix_duration: float,
                min_sac_duration: float) -> tuple[list, list]:
    """Identify fixations and saccades by instantaneous velocity.

    Returns two lists of dicts with keys:
        fix : start, end, dur, x, y
        sac : start, end, dur, x1, y1, x2, y2, ampl
    """
    dt = np.diff(t)
    dt[dt <= 0] = np.finfo(float).eps        # guard against dup timestamps
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx * dx + dy * dy) / dt  # pixels per ms
    speed_pix_per_s = speed * 1000.0         # convert to pixels/second

    # Convert the user-specified velocity threshold from deg/s to px/s
    # using a conservative assumption (≈ 30 px per visual degree on a
    # typical 1280×1024 monitor at 60 cm).  Users who want finer control
    # can pass their own px/s threshold by setting the ``px_per_deg``
    # kwarg via :func:`detect_events`.
    is_sac = speed_pix_per_s > velocity_threshold

    fixations, saccades = [], []
    n = len(is_sac)
    i = 0
    while i < n:
        j = i
        while j < n and is_sac[j] == is_sac[i]:
            j += 1
        t_start, t_end = t[i], t[j]            # inclusive endpoints
        dur = t_end - t_start
        if is_sac[i]:
            if dur >= min_sac_duration:
                saccades.append({
                    'start': t_start, 'end': t_end, 'dur': dur,
                    'x1': x[i], 'y1': y[i],
                    'x2': x[j],  'y2': y[j],
                    'ampl': float(np.hypot(x[j] - x[i], y[j] - y[i])),
                })
        else:
            if dur >= min_fix_duration:
                fixations.append({
                    'start': t_start, 'end': t_end, 'dur': dur,
                    'x': float(np.nanmean(x[i:j + 1])),
                    'y': float(np.nanmean(y[i:j + 1])),
                })
        i = j
    return fixations, saccades


def _idt_detect(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                dispersion_threshold: float,
                min_fix_duration: float) -> tuple[list, list]:
    """Identify fixations by dispersion (Salvucci & Goldberg 2000)."""
    fixations = []
    n = len(t)
    i = 0
    while i < n:
        # grow the window while dispersion < threshold AND duration < min
        j = i
        while j < n:
            xs, ys = x[i:j + 1], y[i:j + 1]
            disp = (np.nanmax(xs) - np.nanmin(xs)
                    + np.nanmax(ys) - np.nanmin(ys))
            if disp > dispersion_threshold:
                break
            j += 1
        dur = t[j - 1] - t[i] if j > i else 0.0
        if dur >= min_fix_duration and j > i + 1:
            fixations.append({
                'start': t[i], 'end': t[j - 1], 'dur': dur,
                'x': float(np.nanmean(x[i:j])),
                'y': float(np.nanmean(y[i:j])),
            })
            i = j
        else:
            i += 1
    # Saccades are implied by the gaps between fixations
    saccades = []
    for a, b in zip(fixations[:-1], fixations[1:]):
        saccades.append({
            'start': a['end'], 'end': b['start'],
            'dur': b['start'] - a['end'],
            'x1': a['x'], 'y1': a['y'],
            'x2': b['x'], 'y2': b['y'],
            'ampl': float(np.hypot(b['x'] - a['x'], b['y'] - a['y'])),
        })
    return fixations, saccades


def detect_events(samples: pd.DataFrame,
                  method: str = 'IVT',
                  velocity_threshold: float = 30.0,
                  dispersion_threshold: float = 35.0,
                  min_fix_duration: float = 50.0,
                  min_sac_duration: float = 10.0,
                  smooth_window: int = 3,
                  px_per_deg: float | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect fixations and saccades from raw gaze samples.

    Parameters
    ----------
    samples : DataFrame
        Output of :func:`read_raw_csv`.  Must contain
        ``time`` (ms), ``x``, ``y``, ``eye``, ``trial_id``.
    method : {'IVT', 'IDT'}
        Velocity-threshold (Salvucci & Goldberg's I-VT) or
        dispersion-threshold (I-DT) detection.
    velocity_threshold : float
        Velocity cutoff.  If ``px_per_deg`` is None, the threshold is
        interpreted directly in pixels/second; otherwise it is read as
        degrees/second and internally converted.
    dispersion_threshold : float
        For I-DT: maximum x+y range (pixels) within a candidate fixation.
    min_fix_duration, min_sac_duration : float
        Minimum event duration in milliseconds.
    smooth_window : int
        Samples of centred moving-average smoothing (1 = no smoothing).
    px_per_deg : float, optional
        Pixels per visual degree.  If given, ``velocity_threshold`` is
        interpreted as deg/s.

    Returns
    -------
    SacDF, FixDF : tuple of DataFrames
        Columns match those produced by ``ext.read_SRRasc``:
            SacDF : trial_id, eye, start, end, dur, x1, y1, x2, y2, ampl
            FixDF : trial_id, eye, start, end, dur, x, y, valid
    """
    method = method.upper()
    if method not in ('IVT', 'IDT'):
        raise ValueError("method must be 'IVT' or 'IDT'")

    sac_threshold = velocity_threshold
    if px_per_deg is not None:
        sac_threshold = velocity_threshold * px_per_deg

    all_fix, all_sac = [], []
    for (tr, eye), g in samples.groupby(['trial_id', 'eye'], sort=False):
        g = g.sort_values('time').reset_index(drop=True)
        t = g['time'].to_numpy()
        x, y = g['x'].to_numpy(), g['y'].to_numpy()
        x, y = _smooth_xy(x, y, smooth_window)
        # Drop NaN samples (data loss during tracking)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        t, x, y = t[ok], x[ok], y[ok]
        if method == 'IVT':
            fx, sc = _ivt_detect(t, x, y, sac_threshold,
                                 min_fix_duration, min_sac_duration)
        else:
            fx, sc = _idt_detect(t, x, y, dispersion_threshold,
                                 min_fix_duration)
        for ev in fx:
            ev.update({'trial_id': tr, 'eye': eye, 'valid': 1})
            all_fix.append(ev)
        for ev in sc:
            ev.update({'trial_id': tr, 'eye': eye})
            all_sac.append(ev)

    FixDF = pd.DataFrame(all_fix, columns=['trial_id', 'eye', 'start', 'end',
                                           'dur', 'x', 'y', 'valid'])
    SacDF = pd.DataFrame(all_sac, columns=['trial_id', 'eye', 'start', 'end',
                                           'dur', 'x1', 'y1', 'x2', 'y2',
                                           'ampl'])
    return SacDF, FixDF


# ---------------------------------------------------------------------------
# III. In-memory cross-line classification (thin wrapper round the disk-based
#      ``ext.cal_crlSacFix``) — exposes the same algorithmic knobs
# ---------------------------------------------------------------------------

def classify_lines(SacDF: pd.DataFrame,
                   FixDF: pd.DataFrame,
                   reg_df: pd.DataFrame,
                   method: str = 'DIFF',
                   diff_ratio: float = 0.6,
                   frontrange_ratio: float = 0.2,
                   y_range: int = 60,
                   handle_prescan: bool = True) -> tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame]:
    """Classify saccades and fixations into text lines.

    This is a tracker-agnostic entry point: hand it a region DataFrame
    (loaded from a ``*_region.csv`` file) and the saccade/fixation
    DataFrames produced by :func:`detect_events` (or any other source
    with the required columns) and it returns the same four objects
    that ``ext.cal_crlSacFix`` writes to disk.

    Parameters
    ----------
    SacDF, FixDF : DataFrame
        Saccade/fixation DataFrames.
    reg_df : DataFrame
        Region DataFrame (one ROI per word; the ``line_no`` column is
        used to distinguish text lines).
    method : {'DIFF', 'SAC'}
        'DIFF' uses between-fixation horizontal displacements; 'SAC'
        uses explicitly detected saccades.
    diff_ratio : float, default 0.6
        Fraction of a line's horizontal extent that a saccade/fixation
        step must exceed to be flagged as crossing-line.  Smaller
        values are more permissive (useful for noisy data) but risk
        misclassifying long within-line regressions as cross-line.
    frontrange_ratio : float, default 0.2
        Fraction of a line's extent counted as its "front" region.
        Used when checking backward cross-line moves.
    y_range : int, default 60 (pixels)
        Vertical threshold used to detect end-of-reading wrap-up.
        Lowering this value is useful for scanning/pre-scan behaviour.
    handle_prescan : bool, default True
        If True, fixations preceding the first forward cross-line
        movement whose y-position suggests pre-scanning are flagged
        ``valid=0`` instead of being assigned to line 0.  This addresses
        Reviewer #1's comment about readers scanning the whole passage
        before reading line-by-line.

    Returns
    -------
    (SacDF_out, crlSac, FixDF_out, crlFix)
    """
    # Defer the import so this module can be imported without the heavy
    # EyeLink-ASC parser side of pyemread.
    try:
        from . import ext as _ext                           # pragma: no cover
    except (ImportError, ValueError):
        import ext as _ext                                  # flat-layout fallback

    SacDF = SacDF.copy()
    FixDF = FixDF.copy()

    # Helpful columns expected by the downstream helpers
    if 'line_no' not in SacDF.columns:
        SacDF['line_no'] = -1
    if 'line_no' not in FixDF.columns:
        FixDF['line_no'] = -1
    if 'region_no' not in FixDF.columns:
        FixDF['region_no'] = -1

    if method == 'SAC':
        crlSac = _ext._getcrlSac(reg_df, SacDF, diff_ratio,
                                 frontrange_ratio, y_range)
        crlFix = _ext._getcrlFix(reg_df, crlSac, FixDF, 'SAC',
                                 diff_ratio, frontrange_ratio, y_range)
    else:
        crlFix = _ext._getcrlFix(reg_df, _pd_empty_sac(), FixDF, 'DIFF',
                                 diff_ratio, frontrange_ratio, y_range)
        crlSac = _pd_empty_sac()

    if handle_prescan:
        FixDF = _flag_prescan(FixDF, crlFix, y_range)

    return SacDF, crlSac, FixDF, crlFix


def _pd_empty_sac() -> pd.DataFrame:
    return pd.DataFrame(columns=['trial_id', 'eye', 'start', 'end',
                                 'dur', 'x1', 'y1', 'x2', 'y2',
                                 'ampl', 'line_no'])


def _flag_prescan(FixDF: pd.DataFrame,
                  crlFix: pd.DataFrame,
                  y_range: int) -> pd.DataFrame:
    """Mark fixations that look like pre-reading scans as invalid.

    Heuristic: if the *first* cross-line fixation in the trial moves
    vertically by more than ``y_range`` from the one immediately before
    it *and* that preceding fixation is not on the expected line-1
    baseline, all fixations prior to it are flagged invalid.
    """
    if crlFix is None or len(crlFix) == 0 or len(FixDF) == 0:
        return FixDF
    FixDF = FixDF.copy()
    for tr in FixDF['trial_id'].unique():
        mask_t = FixDF['trial_id'] == tr
        tr_fix = FixDF.loc[mask_t].sort_values('start').reset_index()
        tr_crl = crlFix[crlFix['trial_id'] == tr].sort_values('start')
        if tr_crl.empty:
            continue
        first_crl_time = tr_crl.iloc[0]['start']
        # candidate pre-scan fixations = those before first cross-line
        pre = tr_fix[tr_fix['start'] < first_crl_time]
        if len(pre) < 3:               # heuristic: need ≥3 fixations to "scan"
            continue
        y_span = pre['y'].max() - pre['y'].min()
        if y_span > y_range:
            original_idx = pre['index'].to_list()
            FixDF.loc[original_idx, 'valid'] = 0
    return FixDF


# ---------------------------------------------------------------------------
# IV. File-I/O convenience
# ---------------------------------------------------------------------------

def write_events(direct: str, subjID: str,
                 SacDF: pd.DataFrame, FixDF: pd.DataFrame) -> None:
    """Write CSV reports mirroring ``ext.write_Sac_Report`` / ``write_Fix_Report``."""
    os.makedirs(direct, exist_ok=True)
    SacDF.to_csv(os.path.join(direct, f'{subjID}_Sac.csv'),
                 index=False, encoding='utf-8')
    FixDF.to_csv(os.path.join(direct, f'{subjID}_Fix.csv'),
                 index=False, encoding='utf-8')


# ---------------------------------------------------------------------------
# V. Thin format-specific loaders for the three trackers named by Reviewer #1
# ---------------------------------------------------------------------------

def tobii_to_raw(path: str, eye: str = 'average') -> pd.DataFrame:
    """Load a Tobii Pro Lab ``.tsv`` export.

    Parameters
    ----------
    path : str
    eye : {'left', 'right', 'average'}
        Which columns to use.  'average' means midpoint of left & right.
    """
    df = pd.read_csv(path, sep='\t', encoding='utf-16', low_memory=False)
    if eye == 'average':
        x = (df['Gaze point X (MCSpx)'].astype(float) * 0  # placeholder
             + 0.5 * df.get('Gaze point left X (MCSpx)', np.nan).astype(float)
             + 0.5 * df.get('Gaze point right X (MCSpx)', np.nan).astype(float))
        y = (0.5 * df.get('Gaze point left Y (MCSpx)', np.nan).astype(float)
             + 0.5 * df.get('Gaze point right Y (MCSpx)', np.nan).astype(float))
    else:
        x = df[f'Gaze point {eye} X (MCSpx)'].astype(float)
        y = df[f'Gaze point {eye} Y (MCSpx)'].astype(float)
    time = df['Recording timestamp'].astype(float) / 1000.0   # µs → ms
    eye_col = 'L' if eye == 'left' else ('R' if eye == 'right' else 'R')
    out = pd.DataFrame({'time': time - time.iloc[0], 'x': x, 'y': y,
                        'eye': eye_col,
                        'trial_id': df.get('Recording number', 1)})
    return out.dropna(subset=['x', 'y']).reset_index(drop=True)


def pupil_labs_to_raw(gaze_positions_csv: str,
                      screen_w: int, screen_h: int) -> pd.DataFrame:
    """Load Pupil Labs ``gaze_positions.csv`` (normalized 0–1 coords).

    Parameters
    ----------
    gaze_positions_csv : str
    screen_w, screen_h : int
        Pixel resolution used to de-normalize coordinates.
    """
    df = pd.read_csv(gaze_positions_csv)
    time = df['gaze_timestamp'].astype(float) * 1000.0        # s → ms
    x = df['norm_pos_x'].astype(float) * screen_w
    y = (1.0 - df['norm_pos_y'].astype(float)) * screen_h      # Pupil: y-up
    return pd.DataFrame({'time': time - time.iloc[0], 'x': x, 'y': y,
                         'eye': 'R', 'trial_id': 1})


def smi_to_raw(path: str) -> pd.DataFrame:
    """Load an SMI RED/iView sample-report .txt file.

    Expects whitespace-delimited columns with a header line containing
    ``Time``, ``L POR X [px]``, ``L POR Y [px]``, ``R POR X [px]``,
    ``R POR Y [px]``.
    """
    df = pd.read_csv(path, sep=r'\s+', comment='#', encoding='utf-8',
                     engine='python')
    if 'Time' not in df.columns:
        raise KeyError("Could not find a 'Time' column — is this an "
                       "SMI sample report?")
    time_ms = df['Time'].astype(float) / 1000.0                # µs → ms
    rows = []
    for eye_lbl, xcol, ycol in (('L', 'L POR X [px]', 'L POR Y [px]'),
                                ('R', 'R POR X [px]', 'R POR Y [px]')):
        if xcol in df.columns and ycol in df.columns:
            sub = pd.DataFrame({'time': time_ms, 'x': df[xcol].astype(float),
                                'y': df[ycol].astype(float),
                                'eye': eye_lbl, 'trial_id': 1})
            rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out['time'] = out['time'] - out['time'].min()
    return out.dropna(subset=['x', 'y']).reset_index(drop=True)


def webgazer_to_raw(path: str) -> pd.DataFrame:
    """Load a WebGazer.js JSON-lines log (one object per sample)."""
    import json
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            rows.append({'time': float(obj['t']),
                         'x': float(obj['x']),
                         'y': float(obj['y']),
                         'eye': 'R', 'trial_id': obj.get('trial', 1)})
    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    df['time'] = df['time'] - df['time'].min()
    return df
