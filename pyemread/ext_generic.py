# -*- coding: utf-8 -*-
"""
Input from eye trackers other than EyeLink.

pyemread's line assignment and measures operate on a fixation table (see
:data:`pyemread.ext.FIX_COLUMNS`). This module creates that table from
data recorded with other systems, in three ways, listed from most to
least recommended:

1. **Fixations detected by the recording software.** Most trackers ship
   validated event detectors (e.g. Tobii Pro Lab's I-VT filter). For
   head-mounted (mobile) trackers, detectors that compensate for head
   movement are required (Drews & Dierkes, 2024). :func:`read_fixations`
   accepts any fixation table, given a mapping of its column names.
   :func:`tobii_fixations`, :func:`pupil_core_surface_fixations` and
   :func:`neon_surface_fixations` read specific exports. :func:`bids_fixations`
   reads fixation events stored in Eye-Tracking-BIDS format.
2. **Eye-Tracking-BIDS recordings** (:func:`read_bids_eyetrack`), which
   bundle samples, events and metadata (screen geometry, sampling rate,
   coordinate system) in a documented structure.
3. **Raw samples**, from which fixations are detected with a
   velocity-threshold (I-VT) or dispersion-threshold (I-DT) algorithm
   (:func:`detect_events`). This route is intended for screen-based,
   head-fixed recordings at sampling rates of roughly 250 Hz or more.

Trials can be cut from a continuous recording with :func:`epoch`, using a
table of trial onsets and offsets or the messages logged during recording
(:func:`trials_from_messages`). :func:`process_fixations` then assigns
lines and words and writes the tables that :func:`pyemread.cal.cal_write_em`
reads.

The readers for specific vendors' exports (Tobii Pro Lab, Pupil Labs
Core and Neon) are **experimental**. They were written against those
vendors' documented export formats and tested on synthetic files, but
not on a validated corpus of real recordings. Check their output against
the vendor software before relying on it.

Examples
--------
>>> from pyemread import ext_generic as eg, cal
>>> regions = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]
>>> fix = eg.read_fixations("examples/fixation_table/1950138_fixations.csv",
...                         column_map={"trial_id": "trial", "trial_type": "stimulus",
...                                     "start": "start_ms", "end": "end_ms",
...                                     "duration": "duration_ms",
...                                     "x": "x_px", "y": "y_px"},
...                         sampfreq=250)                    # doctest: +SKIP
>>> eg.process_fixations("data", "table-1950138", fix, regions)   # doctest: +SKIP
>>> cal.cal_write_em("data", "table-1950138", regions)            # doctest: +SKIP
"""

from __future__ import annotations

__author__ = "Tao Gong"
__license__ = "MIT"

import glob
import gzip
import json
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

__all__ = [
    "read_samples", "read_fixations", "read_bids_eyetrack", "bids_fixations",
    "trials_from_messages", "epoch", "detect_events", "process_fixations",
    "classify_lines", "write_events",
    "tobii_to_raw", "tobii_fixations",
    "pupil_core_surface_to_raw", "pupil_core_surface_fixations",
    "neon_surface_to_raw", "neon_surface_fixations",
    "read_raw_csv", "read_raw_tsv",
]

_TIME_FACTORS = {'ms': 1.0, 'millisecond': 1.0, 'milliseconds': 1.0,
                 's': 1000.0, 'sec': 1000.0, 'second': 1000.0, 'seconds': 1000.0,
                 'us': 1e-3, 'µs': 1e-3, 'μs': 1e-3, 'microsecond': 1e-3, 'microseconds': 1e-3,
                 'ns': 1e-6, 'nanosecond': 1e-6, 'nanoseconds': 1e-6}


def _to_ms(values, unit: str) -> np.ndarray:
    try:
        f = _TIME_FACTORS[unit.lower()]
    except KeyError:
        raise ValueError(f"Unknown time unit {unit!r}; use one of 'ms', 's', 'us', 'ns'.")
    return np.asarray(values, dtype=float) * f


def _sniff_table(path: str, sep: Optional[str] = None, encoding: Optional[str] = None,
                 **kwargs) -> pd.DataFrame:
    """Read a delimited text file, detecting encoding (UTF-8/UTF-16) and separator."""
    if encoding is None:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rb') as fh:
            head = fh.read(4)
        if head[:2] in (b'\xff\xfe', b'\xfe\xff'):
            encoding = 'utf-16'
        else:
            encoding = 'utf-8-sig'
    if sep is None:
        sep = '\t' if str(path).lower().endswith(('.tsv', '.tsv.gz', '.txt')) else ','
    return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False, **kwargs)


# ---------------------------------------------------------------------------
# Generic readers
# ---------------------------------------------------------------------------

def read_samples(path: Union[str, pd.DataFrame], time_col: str = 'time', x_col: str = 'x',
                 y_col: str = 'y', eye_col: Optional[str] = None,
                 trial_col: Optional[str] = None, pupil_col: Optional[str] = None,
                 time_unit: str = 'ms', sep: Optional[str] = None,
                 encoding: Optional[str] = None, eye: str = 'R') -> pd.DataFrame:
    """Read raw gaze samples from a delimited file or a DataFrame.

    Parameters
    ----------
    path : str or pandas.DataFrame
        File (CSV/TSV, optionally gzipped) or an existing table.
    time_col, x_col, y_col : str
        Columns holding the time stamp and the horizontal and vertical gaze
        position in screen pixels (origin at the top-left corner of the
        screen, y increasing downwards).
    eye_col : str, optional
        Column with the eye ('L'/'R' or 'left'/'right'). If omitted, all
        samples are labelled ``eye``.
    trial_col : str, optional
        Column with a trial identifier. If omitted, ``trial_id`` is 1; use
        :func:`epoch` to cut trials from a continuous recording.
    pupil_col : str, optional
        Column with pupil size.
    time_unit : {'ms', 's', 'us', 'ns'}
        Unit of ``time_col``. Times are converted to milliseconds but are not
        re-zeroed, so they stay comparable with the recording's messages.
    sep, encoding : str, optional
        Separator and encoding; detected automatically if omitted.
    eye : str
        Eye label used when ``eye_col`` is None (default 'R').

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (ms), ``x``, ``y``, ``pupil``, ``eye``, ``trial_id``.
        Missing samples (track loss) are kept as ``NaN``.
    """
    df = path if isinstance(path, pd.DataFrame) else _sniff_table(path, sep, encoding)
    missing = [c for c in (time_col, x_col, y_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    out = pd.DataFrame({
        'time': _to_ms(pd.to_numeric(df[time_col], errors='coerce'), time_unit),
        'x': pd.to_numeric(df[x_col], errors='coerce').astype(float),
        'y': pd.to_numeric(df[y_col], errors='coerce').astype(float),
        'pupil': (pd.to_numeric(df[pupil_col], errors='coerce').astype(float)
                  if pupil_col else np.nan),
    })
    out['eye'] = (df[eye_col].astype(str).str.strip().str.upper().str[0].to_numpy()
                  if eye_col else eye)
    out['trial_id'] = df[trial_col].to_numpy() if trial_col else 1
    return out.dropna(subset=['time']).sort_values(['trial_id', 'eye', 'time'],
                                                   kind='stable').reset_index(drop=True)


def read_raw_csv(path, time_col='time', x_col='x', y_col='y', eye_col=None, trial_col=None,
                 time_unit='ms', sep=',', encoding=None) -> pd.DataFrame:
    """Alias of :func:`read_samples` kept from earlier versions."""
    return read_samples(path, time_col, x_col, y_col, eye_col, trial_col, None,
                        time_unit, sep, encoding)


def read_raw_tsv(path, **kwargs) -> pd.DataFrame:
    """Alias of :func:`read_samples` for tab-separated files."""
    kwargs.setdefault('sep', '\t')
    return read_raw_csv(path, **kwargs)


def _fix_table(start, end, x, y, subj='', trial_id=1, trial_type='', eye='R',
               pupil=np.nan, duration=None, sampfreq=np.nan) -> pd.DataFrame:
    """Assemble a fixation table in the pyemread schema."""
    from .ext import FIX_COLUMNS
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    n = len(start)
    dur = np.asarray(duration, float) if duration is not None else end - start
    df = pd.DataFrame({
        'subj': subj, 'trial_id': trial_id, 'trial_type': trial_type,
        'sampfreq': sampfreq, 'script': '', 'sessdate': '', 'srcfile': '',
        'trialstart': np.nan, 'trialend': np.nan, 'tdur': np.nan,
        'recstart': np.nan, 'recend': np.nan, 'blinks': np.nan, 'eye': eye,
        'start_time': start, 'end_time': end, 'duration': dur,
        'x_pos': np.asarray(x, float), 'y_pos': np.asarray(y, float),
        'pup_size': pupil, 'valid': 'yes', 'line_no': np.nan, 'region_no': np.nan,
    }, index=range(n))
    return df[FIX_COLUMNS]


def _finish_fix_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fill trial-level timing columns and sort."""
    if len(df) == 0:
        return df
    df = df.sort_values(['trial_id', 'eye', 'start_time'], kind='stable').reset_index(drop=True)
    g = df.groupby(['trial_id', 'eye'])
    df['trialstart'] = g['start_time'].transform('min')
    df['trialend'] = g['end_time'].transform('max')
    df['tdur'] = df['trialend'] - df['trialstart']
    df['recstart'] = df['trialstart']
    df['recend'] = df['trialend']
    df['blinks'] = df['blinks'].fillna(0)
    return df


def read_fixations(path: Union[str, pd.DataFrame], column_map: Optional[Dict[str, str]] = None,
                   time_unit: str = 'ms', subj: str = '', trial_type: Optional[str] = None,
                   eye: str = 'R', sep: Optional[str] = None,
                   encoding: Optional[str] = None,
                   sampfreq: Optional[float] = None) -> pd.DataFrame:
    """Read a fixation table produced by any software into the pyemread schema.

    Parameters
    ----------
    path : str or pandas.DataFrame
        File (CSV/TSV) or table with one row per fixation.
    column_map : dict, optional
        Maps pyemread names to the file's column names. Recognised keys:
        ``start`` (required), ``end`` or ``duration`` (one required), ``x``,
        ``y`` (required), ``trial_id``, ``trial_type``, ``eye``, ``pupil``.
        Keys that are omitted are looked up under their own names. If both
        ``end`` and ``duration`` are present, the duration is taken from
        the file (some programs, e.g. the EyeLink parser, count durations
        inclusively); otherwise it is ``end - start``.
    time_unit : {'ms', 's', 'us', 'ns'}
        Unit of the start/end/duration columns.
    subj : str
        Participant identifier.
    trial_type : str, optional
        Stimulus (trial type) for all rows, used when the table has no
        ``trial_type`` column. It must match a region file name, e.g.
        ``"story01"`` for ``story01.region.csv``.
    eye : str
        Eye label used when the table has no ``eye`` column.
    sep, encoding : str, optional
        Delimiter and encoding of the file (detected if omitted).
    sampfreq : float, optional
        Sampling rate of the recording in Hz, used when short fixations are
        merged (:func:`pyemread.ext.apply_fixation_validity`).

    Returns
    -------
    pandas.DataFrame
        Fixation table with the columns of :data:`pyemread.ext.FIX_COLUMNS`.

    Raises
    ------
    KeyError
        If the start, x, y and end-or-duration columns cannot be found.
    """
    df = path if isinstance(path, pd.DataFrame) else _sniff_table(path, sep, encoding)
    cmap = {k: k for k in ('start', 'end', 'duration', 'x', 'y', 'trial_id', 'trial_type',
                           'eye', 'pupil')}
    cmap.update(column_map or {})
    col = lambda k: cmap[k] if cmap[k] in df.columns else None
    if col('start') is None or col('x') is None or col('y') is None:
        raise KeyError("A fixation table needs start, x and y columns; supply column_map.")
    start = _to_ms(df[col('start')], time_unit)
    duration = _to_ms(df[col('duration')], time_unit) if col('duration') is not None else None
    if col('end') is not None:
        end = _to_ms(df[col('end')], time_unit)
    elif duration is not None:
        end = start + duration
    else:
        raise KeyError("A fixation table needs an end or a duration column.")
    out = _fix_table(start, end, df[col('x')], df[col('y')], subj=subj,
                     trial_id=df[col('trial_id')].to_numpy() if col('trial_id') else 1,
                     trial_type=(df[col('trial_type')].astype(str).to_numpy()
                                 if col('trial_type') else (trial_type or '')),
                     eye=(df[col('eye')].astype(str).str.upper().str[0].to_numpy()
                          if col('eye') else eye),
                     pupil=df[col('pupil')].to_numpy(float) if col('pupil') else np.nan,
                     duration=duration,
                     sampfreq=float(sampfreq) if sampfreq else np.nan)
    if not col('trial_id') and col('trial_type'):
        # number trials by order of first appearance of each stimulus
        codes = {t: i + 1 for i, t in enumerate(dict.fromkeys(out['trial_type']))}
        out['trial_id'] = out['trial_type'].map(codes)
    return _finish_fix_table(out)


# ---------------------------------------------------------------------------
# Eye-Tracking-BIDS
# ---------------------------------------------------------------------------

def _bids_sidecar(tsv_path: str) -> dict:
    """Merge the sidecars that apply to a BIDS file (inheritance principle).

    Sidecars are collected from the file's directory up to the dataset root
    (the directory containing ``dataset_description.json``, or the
    filesystem root). A sidecar applies if its entities are a subset of the
    file's entities and its suffix matches. Deeper sidecars override
    shallower ones.
    """
    base = os.path.basename(tsv_path)
    stem = re.sub(r'\.tsv(\.gz)?$', '', base)
    suffix = stem.split('_')[-1]
    ents = set(stem.split('_')[:-1])
    found, d = [], os.path.dirname(os.path.abspath(tsv_path))
    while True:
        for js in glob.glob(os.path.join(d, f'*_{suffix}.json')):
            js_ents = set(os.path.basename(js)[:-len('.json')].split('_')[:-1])
            if js_ents <= ents:
                found.append((len(js_ents), js))
        if os.path.exists(os.path.join(d, 'dataset_description.json')) or os.path.dirname(d) == d:
            break
        d = os.path.dirname(d)
    meta = {}
    for _, js in sorted(found):
        with open(js, encoding='utf-8') as fh:
            meta.update(json.load(fh))
    return meta


def _read_bids_tsv(path: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read a header-less BIDS physio(events) file whose columns are in the sidecar."""
    if os.path.getsize(path) <= 20:            # an empty gzip stream
        return pd.DataFrame(columns=list(columns))
    df = pd.read_csv(path, sep='\t', header=None, encoding='utf-8-sig', na_values=['n/a'],
                     names=list(columns), compression='infer', quoting=3)
    return df


def read_bids_eyetrack(physio_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read an Eye-Tracking-BIDS recording (``*_physio.tsv.gz`` with ``PhysioType`` 'eyetrack').

    Parameters
    ----------
    physio_path : str
        Path to a ``*_recording-<label>_physio.tsv.gz`` file. Its sidecar
        JSON files and the matching ``*_physioevents.tsv.gz`` file (if
        present) are found automatically.

    Returns
    -------
    samples : pandas.DataFrame
        ``time`` (ms, as recorded), ``x``, ``y``, ``pupil``, ``eye``,
        ``trial_id`` (1).
    events : pandas.DataFrame
        The physioevents table with ``onset`` and ``duration`` converted to
        milliseconds on the same clock as ``samples.time``, or an empty
        table.
    meta : dict
        The merged sidecar metadata of the samples (e.g. ``SamplingFrequency``,
        ``RecordedEye``, ``SampleCoordinateSystem``, ``ScreenSize``).

    Notes
    -----
    Coordinates are returned as stored. pyemread expects screen pixels with
    the origin at the top left (``SampleCoordinateSystem`` "gaze-on-screen",
    ``SampleCoordinateUnits`` "pixel"); a warning is issued otherwise.

    Raises
    ------
    ValueError
        If the file is not an eye-tracking recording (``PhysioType``).
    """
    meta = _bids_sidecar(physio_path)
    if meta.get('PhysioType', 'eyetrack') != 'eyetrack':
        raise ValueError(f"{physio_path} is not an eye-tracking recording "
                         f"(PhysioType={meta.get('PhysioType')!r}).")
    cols = meta.get('Columns') or ['timestamp', 'x_coordinate', 'y_coordinate', 'pupil_size']
    raw = _read_bids_tsv(physio_path, cols)
    t_unit = (meta.get('timestamp') or {}).get('Units', 'ms') if isinstance(meta.get('timestamp'), dict) else 'ms'
    eye_name = str(meta.get('RecordedEye', 'right')).lower()
    eye = {'left': 'L', 'right': 'R'}.get(eye_name, 'C')
    units = (meta.get('x_coordinate') or {}).get('Units', meta.get('SampleCoordinateUnits', 'pixel')) \
        if isinstance(meta.get('x_coordinate'), dict) else meta.get('SampleCoordinateUnits', 'pixel')
    system = meta.get('SampleCoordinateSystem', 'gaze-on-screen')
    if str(units).lower() not in ('pixel', 'pixels', 'px') or system != 'gaze-on-screen':
        _log.warning(f"{os.path.basename(physio_path)}: coordinates are in {units!r} "
                     f"({system}); pyemread expects gaze-on-screen pixels.")
    samples = pd.DataFrame({
        'time': _to_ms(raw['timestamp'], t_unit),
        'x': pd.to_numeric(raw.get('x_coordinate'), errors='coerce'),
        'y': pd.to_numeric(raw.get('y_coordinate'), errors='coerce'),
        'pupil': pd.to_numeric(raw['pupil_size'], errors='coerce') if 'pupil_size' in raw else np.nan,
    })
    samples['eye'] = eye
    samples['trial_id'] = 1
    ev_path = re.sub(r'_physio\.tsv(\.gz)?$', r'_physioevents.tsv\1', physio_path)
    events = pd.DataFrame(columns=['onset', 'duration', 'trial_type', 'message'])
    if ev_path != physio_path and os.path.exists(ev_path):
        emeta = _bids_sidecar(ev_path)
        ecols = emeta.get('Columns') or ['onset', 'duration', 'trial_type', 'blink', 'message']
        ev = _read_bids_tsv(ev_path, ecols)
        o_unit = (emeta.get('onset') or {}).get('Units', 's') if isinstance(emeta.get('onset'), dict) else 's'
        d_unit = (emeta.get('duration') or {}).get('Units', o_unit) if isinstance(emeta.get('duration'), dict) else o_unit
        t0 = float(samples['time'].iloc[0]) if len(samples) else 0.0
        t1 = float(samples['time'].iloc[-1]) if len(samples) else 0.0
        onset = _to_ms(pd.to_numeric(ev['onset'], errors='coerce'), o_unit)
        # With OnsetSource 'timestamp', onsets may be expressed on the device
        # clock; otherwise they are relative to the first sample.
        absolute = (emeta.get('OnsetSource') == 'timestamp' and len(onset) > 0 and
                    np.nanmin(onset) >= t0 - 1000.0 and np.nanmax(onset) <= t1 + 1000.0 and t0 != 0.0)
        ev['onset'] = onset if absolute else t0 + onset
        ev['duration'] = _to_ms(pd.to_numeric(ev['duration'], errors='coerce'), d_unit)
        events = ev
    return samples, events, meta


def bids_fixations(samples: pd.DataFrame, events: pd.DataFrame, subj: str = '',
                   trial_type: str = '', label: str = 'fixation') -> pd.DataFrame:
    """Fixations from Eye-Tracking-BIDS events, positioned with the samples.

    BIDS physioevents store the onset and duration of fixations detected by
    the tracker but not their positions. The position of each fixation is
    the median of the samples recorded during it.

    Parameters
    ----------
    samples, events : pandas.DataFrame
        As returned by :func:`read_bids_eyetrack`.
    subj, trial_type : str
        Participant and stimulus identifiers for the output table.
    label : str
        Value of the ``trial_type`` column that marks fixations
        (default 'fixation').

    Returns
    -------
    pandas.DataFrame
        Fixation table in the pyemread schema.
    """
    if 'trial_type' not in events or not (events['trial_type'] == label).any():
        raise ValueError("The events table contains no fixation events; detect them with "
                         "detect_events() instead.")
    fx = events[events['trial_type'] == label]
    t = samples['time'].to_numpy(float)
    rows = []
    for on, du in zip(fx['onset'].to_numpy(float), fx['duration'].to_numpy(float)):
        sel = (t >= on) & (t <= on + du)
        rows.append((on, on + du, np.nanmedian(samples['x'].to_numpy(float)[sel]) if sel.any() else np.nan,
                     np.nanmedian(samples['y'].to_numpy(float)[sel]) if sel.any() else np.nan))
    rows = np.array(rows, float).reshape(-1, 4)
    dt = np.nanmedian(np.diff(t)) if len(t) > 1 else np.nan
    out = _fix_table(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], subj=subj,
                     trial_type=trial_type, eye=str(samples['eye'].iloc[0]) if len(samples) else 'R',
                     sampfreq=1000.0 / dt if np.isfinite(dt) and dt > 0 else np.nan)
    return _finish_fix_table(out)


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------

def trials_from_messages(events: pd.DataFrame, start_pattern: str, end_pattern: str,
                         type_pattern: Optional[str] = None) -> pd.DataFrame:
    """Build a trial table from messages logged during a continuous recording.

    Parameters
    ----------
    events : pandas.DataFrame
        Table with ``onset`` (ms) and ``message`` columns, e.g. the events
        returned by :func:`read_bids_eyetrack`.
    start_pattern, end_pattern : str
        Regular expressions matching the messages that open and close a
        trial (e.g. ``r"TRIALID"`` and ``r"TRIAL_RESULT"``).
    type_pattern : str, optional
        Regular expression with one capture group that extracts the trial
        type (stimulus name) from a message, e.g.
        ``r"picture_name\\s+(\\S+?)\\.png"``. Only messages between the start
        and end messages of a trial are searched.

    Returns
    -------
    pandas.DataFrame
        ``trial_id`` (1-based), ``start`` and ``end`` (ms) and ``trial_type``.
    """
    msg = events.dropna(subset=['message']).sort_values('onset')
    rows, cur = [], None
    for on, m in zip(msg['onset'], msg['message'].astype(str)):
        if re.search(start_pattern, m):
            cur = {'trial_id': len(rows) + 1, 'start': float(on), 'end': np.nan, 'trial_type': ''}
        elif cur is not None and re.search(end_pattern, m):
            cur['end'] = float(on)
            rows.append(cur)
            cur = None
        elif cur is not None and type_pattern:
            hit = re.search(type_pattern, m)
            if hit:
                cur['trial_type'] = hit.group(1)
    return pd.DataFrame(rows, columns=['trial_id', 'start', 'end', 'trial_type'])


def epoch(data: pd.DataFrame, trials: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    """Cut a continuous recording into trials.

    Parameters
    ----------
    data : pandas.DataFrame
        Samples (``time`` column) or fixations (``start_time``/``end_time``)
        from one continuous recording.
    trials : pandas.DataFrame
        One row per trial with ``trial_id``, ``start``, ``end`` (ms, on the
        same clock as ``data``) and ``trial_type`` (the stimulus, matching a
        region file name).
    time_col : str, optional
        Column used to place rows in trials; defaults to ``time`` for
        samples and ``start_time`` for fixations. A fixation belongs to the
        trial in which it starts.

    Returns
    -------
    pandas.DataFrame
        The rows of ``data`` that fall inside a trial, with ``trial_id``
        and ``trial_type`` set (and, for fixations, the trial timing
        columns).
    """
    if time_col is None:
        time_col = 'time' if 'time' in data else 'start_time'
    t = data[time_col].to_numpy(float)
    parts = []
    for _, tr in trials.iterrows():
        sel = (t >= tr['start']) & (t < tr['end'])
        part = data.loc[sel].copy()
        part['trial_id'] = tr['trial_id']
        part['trial_type'] = tr.get('trial_type', '')
        if 'start_time' in part:
            part['trialstart'] = tr['start']
            part['trialend'] = tr['end']
            part['tdur'] = tr['end'] - tr['start']
            part['recstart'] = tr['start']
            part['recend'] = tr['end']
        parts.append(part)
    if not parts:
        return data.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Event detection from raw samples
# ---------------------------------------------------------------------------

def _sg_derivative(v: np.ndarray, half: int, order: int = 2) -> np.ndarray:
    """First derivative (per sample) with a Savitzky-Golay filter."""
    if half < 1:
        return np.gradient(v)
    k = np.arange(-half, half + 1, dtype=float)
    A = np.vander(k, order + 1, increasing=True)
    coef = np.linalg.pinv(A)[1]                    # derivative at the window centre
    pad = np.pad(v, half, mode='edge')
    return np.convolve(pad, coef[::-1], mode='valid')


def _segments(ok: np.ndarray) -> List[Tuple[int, int]]:
    """Start/end (exclusive) indices of runs of True."""
    d = np.diff(np.r_[0, ok.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _interp_short_gaps(t, x, y, max_gap_ms):
    """Linearly interpolate runs of missing samples no longer than ``max_gap_ms``."""
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if not bad.any():
        return x, y
    x, y = x.copy(), y.copy()
    good = np.where(~bad)[0]
    for s, e in _segments(bad):
        if s == 0 or e >= len(t):
            continue
        if t[e] - t[s - 1] <= max_gap_ms:
            for arr in (x, y):
                arr[s:e] = np.interp(t[s:e], [t[s - 1], t[e]], [arr[s - 1], arr[e]])
    return x, y


def _merge_fixations(fix: List[dict], max_gap: float, max_dist: float) -> List[dict]:
    out = []
    for f in fix:
        if out and f['start'] - out[-1]['end'] <= max_gap and \
                np.hypot(f['x'] - out[-1]['x'], f['y'] - out[-1]['y']) <= max_dist:
            p = out[-1]
            w1, w2 = p['end'] - p['start'], f['end'] - f['start']
            p['x'] = (p['x'] * w1 + f['x'] * w2) / max(w1 + w2, 1e-9)
            p['y'] = (p['y'] * w1 + f['y'] * w2) / max(w1 + w2, 1e-9)
            p['end'] = f['end']
            p['n'] += f['n']
        else:
            out.append(dict(f))
    return out


def detect_events(samples: pd.DataFrame, method: str = 'IVT',
                  velocity_threshold: float = 30.0, velocity_unit: str = 'deg/s',
                  px_per_deg: Optional[float] = None,
                  dispersion_threshold: float = 1.0, dispersion_unit: str = 'deg',
                  smooth_ms: float = 20.0, max_gap_ms: float = 75.0,
                  merge_gap_ms: float = 75.0, merge_dist: float = 0.5,
                  min_fix_duration: float = 60.0, subj: str = '',
                  trial_type: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect fixations and saccades in raw gaze samples (screen-based trackers).

    I-VT classifies samples by their angular velocity, estimated with a
    Savitzky-Golay filter (second-order polynomial over ``smooth_ms``).
    I-DT grows windows whose dispersion stays below a threshold (Salvucci &
    Goldberg, 2000). For both methods, missing samples (track loss, blinks)
    shorter than ``max_gap_ms`` are interpolated, and longer gaps split the
    data so that no event spans them. Fixations separated by less than
    ``merge_gap_ms`` and ``merge_dist`` are merged, and fixations shorter
    than ``min_fix_duration`` are discarded. Saccades are the intervals
    between consecutive fixations that are not interrupted by a gap.

    This detector is meant for head-fixed, screen-based recordings. For
    head-mounted (mobile) trackers, use the fixations detected by the
    vendor's software (see :func:`pupil_core_surface_fixations` and
    :func:`neon_surface_fixations`).

    Parameters
    ----------
    samples : pandas.DataFrame
        Output of :func:`read_samples` (``time`` in ms, ``x``, ``y`` in
        screen pixels, ``eye``, ``trial_id``; optional ``trial_type``).
    method : {'IVT', 'IDT'}
    velocity_threshold : float
        Saccade velocity threshold (default 30).
    velocity_unit : {'deg/s', 'px/s'}
        Unit of ``velocity_threshold``. ``'deg/s'`` requires ``px_per_deg``.
    px_per_deg : float, optional
        Pixels per degree of visual angle at the screen centre. Required
        whenever a threshold is given in degrees.
    dispersion_threshold : float
        I-DT dispersion threshold, (max x - min x) + (max y - min y)
        (default 1.0).
    dispersion_unit : {'deg', 'px'}
    smooth_ms : float
        Width of the Savitzky-Golay window in ms (default 20).
    max_gap_ms : float
        Longest run of missing samples that is interpolated (default 75).
    merge_gap_ms : float
        Fixations separated by at most this interval are merged if they are
        also close in space (default 75).
    merge_dist : float
        Maximum distance between merged fixations, in the dispersion unit
        (default 0.5).
    min_fix_duration : float
        Shortest fixation kept, in ms (default 60).
    subj : str
        Participant identifier for the output tables.
    trial_type : str, optional
        Stimulus name used when ``samples`` has no ``trial_type`` column.

    Returns
    -------
    sac_df, fix_df : pandas.DataFrame
        Saccade and fixation tables in the pyemread schema
        (:data:`pyemread.ext.SAC_COLUMNS`, :data:`pyemread.ext.FIX_COLUMNS`).

    Raises
    ------
    ValueError
        If ``method`` is not 'IVT' or 'IDT', or if a threshold is given in
        degrees without ``px_per_deg``.
    """
    from .ext import SAC_COLUMNS
    method = method.upper()
    if method not in ('IVT', 'IDT'):
        raise ValueError("method must be 'IVT' or 'IDT'")
    in_deg = []
    if method == 'IVT' and velocity_unit == 'deg/s':
        in_deg.append("velocity_threshold (velocity_unit='deg/s')")
    if dispersion_unit == 'deg':
        in_deg.append("dispersion_threshold and merge_dist (dispersion_unit='deg')")
    if in_deg and not px_per_deg:
        raise ValueError("px_per_deg (pixels per degree of visual angle) must be given, because "
                         + " and ".join(in_deg) + " are in degrees. Alternatively, give all "
                         "thresholds in pixels (velocity_unit='px/s', dispersion_unit='px').")
    ppd = float(px_per_deg) if px_per_deg else 1.0
    v_thr = velocity_threshold * (ppd if velocity_unit == 'deg/s' else 1.0)   # px/s
    d_thr = dispersion_threshold * (ppd if dispersion_unit == 'deg' else 1.0)  # px
    m_dist = merge_dist * (ppd if dispersion_unit == 'deg' else 1.0)

    fix_parts, sac_rows = [], []
    for (tr, eye), g in samples.groupby(['trial_id', 'eye'], sort=False):
        g = g.sort_values('time')
        t = g['time'].to_numpy(float)
        if len(t) < 3:
            continue
        x, y = _interp_short_gaps(t, g['x'].to_numpy(float), g['y'].to_numpy(float), max_gap_ms)
        ttype = (str(g['trial_type'].iloc[0]) if 'trial_type' in g else (trial_type or ''))
        dt = float(np.median(np.diff(t)))
        fixes = []
        for s, e in _segments(np.isfinite(x) & np.isfinite(y)):
            ts, xs, ys = t[s:e], x[s:e], y[s:e]
            if len(ts) < 3:
                continue
            if method == 'IVT':
                half = max(1, int(round(smooth_ms / dt / 2)))
                vx = _sg_derivative(xs, half) / dt * 1000.0
                vy = _sg_derivative(ys, half) / dt * 1000.0
                is_fix = np.hypot(vx, vy) < v_thr
                for a, b in _segments(is_fix):
                    fixes.append({'start': ts[a], 'end': ts[b - 1] + dt,
                                  'x': float(np.mean(xs[a:b])), 'y': float(np.mean(ys[a:b])),
                                  'n': b - a, 'seg': s})
            else:
                n_min = max(2, int(np.ceil(min_fix_duration / dt)))
                i = 0
                while i + n_min <= len(ts):
                    j = i + n_min
                    disp = lambda a, b: np.ptp(xs[a:b]) + np.ptp(ys[a:b])
                    if disp(i, j) <= d_thr:
                        while j < len(ts) and disp(i, j + 1) <= d_thr:
                            j += 1
                        fixes.append({'start': ts[i], 'end': ts[j - 1] + dt,
                                      'x': float(np.mean(xs[i:j])), 'y': float(np.mean(ys[i:j])),
                                      'n': j - i, 'seg': s})
                        i = j
                    else:
                        i += 1
        # merge within continuous segments only
        merged = []
        for seg in dict.fromkeys(f['seg'] for f in fixes):
            merged += _merge_fixations([f for f in fixes if f['seg'] == seg], merge_gap_ms, m_dist)
        merged = [f for f in merged if f['end'] - f['start'] >= min_fix_duration]
        fdf = _fix_table([f['start'] for f in merged], [f['end'] for f in merged],
                         [f['x'] for f in merged], [f['y'] for f in merged], subj=subj,
                         trial_id=tr, trial_type=ttype, eye=eye, sampfreq=round(1000.0 / dt))
        fix_parts.append(fdf)
        for a, b in zip(merged[:-1], merged[1:]):
            if a['seg'] != b['seg']:
                continue
            sac_rows.append({'subj': subj, 'trial_id': tr, 'trial_type': ttype,
                             'sampfreq': round(1000.0 / dt), 'eye': eye,
                             'start_time': a['end'], 'end_time': b['start'],
                             'duration': b['start'] - a['end'],
                             'x1_pos': a['x'], 'y1_pos': a['y'], 'x2_pos': b['x'], 'y2_pos': b['y'],
                             'ampl': float(np.hypot(b['x'] - a['x'], b['y'] - a['y']) / ppd)
                             if px_per_deg else np.nan,
                             'pk': np.nan, 'line_no': None})
    fix_df = (_finish_fix_table(pd.concat(fix_parts, ignore_index=True)) if fix_parts
              else _fix_table([], [], [], []))
    sac_df = pd.DataFrame(sac_rows, columns=SAC_COLUMNS)
    if len(sac_df) and len(fix_df):
        timing = fix_df.groupby(['trial_id', 'eye'])[['trialstart', 'trialend', 'tdur', 'recstart',
                                                      'recend', 'blinks']].first().reset_index()
        sac_df = sac_df.drop(columns=['trialstart', 'trialend', 'tdur', 'recstart', 'recend',
                                      'blinks']).merge(timing, on=['trial_id', 'eye'], how='left')
        sac_df = sac_df[SAC_COLUMNS]
    sac_df['line_no'] = sac_df['line_no'].astype(object)
    return sac_df, fix_df


# ---------------------------------------------------------------------------
# Line assignment and writing
# ---------------------------------------------------------------------------

def classify_lines(sac_df: pd.DataFrame, fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                   method: str = 'SLICE', **kwargs):
    """Assign fixations of a single stimulus to lines and words, in memory.

    Thin wrapper around :func:`pyemread.ext.classify_fixations` for data
    whose trials all show the same stimulus. ``reg_df`` is that stimulus's
    region table.

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame
    """
    from . import ext
    fix_df = fix_df.copy()
    tt = 'stimulus'
    fix_df['trial_type'] = tt
    if len(sac_df):
        sac_df = sac_df.copy()
        sac_df['trial_type'] = tt
    out = ext.classify_fixations(sac_df, fix_df, {tt: reg_df}, method, **kwargs)
    return out[:4]


def write_events(direct: str, subj_id: str, sac_df: pd.DataFrame, fix_df: pd.DataFrame) -> None:
    """Write ``<subj_id>_Sac.csv`` and ``<subj_id>_Fix.csv`` to ``<direct>/<subj_id>/``.

    This is the location from which :func:`pyemread.ext.cal_crl_sac_fix`
    and :func:`pyemread.cal.cal_write_em` read.
    """
    folder = os.path.join(direct, subj_id)
    os.makedirs(folder, exist_ok=True)
    sac_df.to_csv(os.path.join(folder, f'{subj_id}_Sac.csv'), index=False, encoding='utf-8')
    fix_df.to_csv(os.path.join(folder, f'{subj_id}_Fix.csv'), index=False, encoding='utf-8')


def process_fixations(direct: str, subj_id: str, fix_df: pd.DataFrame,
                      regfile_name_list: List[str], sac_df: Optional[pd.DataFrame] = None,
                      classify_method: str = 'SLICE', write: bool = True,
                      apply_validity: bool = True, validity_kwargs: Optional[dict] = None,
                      **kwargs):
    """Assign lines and words to fixations from any source and write the tables.

    Parameters
    ----------
    direct : str
        Data directory containing the region files; outputs are written to
        ``<direct>/<subj_id>/``.
    subj_id : str
        Participant identifier.
    fix_df : pandas.DataFrame
        Fixations in the pyemread schema (e.g. from :func:`read_fixations`,
        :func:`bids_fixations` or :func:`detect_events`), with a
        ``trial_type`` that names each trial's region file.
    regfile_name_list : list of str
        Region files.
    sac_df : pandas.DataFrame, optional
        Saccades in the pyemread schema, if available.
    classify_method : str
        See :func:`pyemread.ext.classify_fixations` (default 'SLICE').
    write : bool
        Write ``_Sac.csv``, ``_Fix.csv``, ``_crlSac.csv`` and ``_crlFix.csv``
        (default True).
    apply_validity : bool
        Apply the validity rules used for EyeLink data (merging of short
        fixations, minimum duration, invalid last fixation; see
        :func:`pyemread.ext.apply_fixation_validity`) so that the measures are
        computed in the same way for every source (default True).
    validity_kwargs : dict, optional
        Options passed to :func:`pyemread.ext.apply_fixation_validity`.
    **kwargs
        Passed to :func:`pyemread.ext.classify_fixations`.

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame
    """
    from . import ext
    from .ext import SAC_COLUMNS
    regions = ext.load_regions(direct, regfile_name_list)
    fix_df = fix_df.copy()
    fix_df['subj'] = subj_id
    if apply_validity:
        fix_df = ext.apply_fixation_validity(fix_df, **(validity_kwargs or {}))
    if sac_df is None:
        sac_df = pd.DataFrame(columns=SAC_COLUMNS)
    sac_df, crl_sac, fix_df, crl_fix, flagged = ext.classify_fixations(
        sac_df, fix_df, regions, classify_method, **kwargs)
    ext._log_flags(direct, subj_id, flagged)
    if write:
        write_events(direct, subj_id, sac_df, fix_df)
        folder = os.path.join(direct, subj_id)
        crl_sac.to_csv(os.path.join(folder, f'{subj_id}_crlSac.csv'), index=False)
        crl_fix.to_csv(os.path.join(folder, f'{subj_id}_crlFix.csv'), index=False)
    return sac_df, crl_sac, fix_df, crl_fix


# ---------------------------------------------------------------------------
# Vendor exports (experimental)
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *patterns: str, required: bool = True) -> Optional[str]:
    for p in patterns:
        for c in df.columns:
            if re.fullmatch(p, str(c).strip(), flags=re.IGNORECASE):
                return c
    if required:
        raise KeyError(f"No column matches {patterns}. Columns: {list(df.columns)}")
    return None


def _unit_of(col: str, default: str) -> str:
    m = re.search(r'[\[(]\s*([µμu]s|ms|ns|s)\s*[\])]', col)
    return m.group(1).replace('µ', 'u').replace('μ', 'u') if m else default


def tobii_to_raw(path: str, eye: str = 'average') -> pd.DataFrame:
    """Read gaze samples from a Tobii Pro Lab data export (experimental).

    The export (Analyze > Data export, tab-separated) may be UTF-8 or
    UTF-16. Column names with either the ``[DACS px]``/``[μs]`` or the older
    ``(MCSpx)`` convention are recognised.

    Parameters
    ----------
    path : str
        The ``.tsv`` export.
    eye : {'average', 'left', 'right'}
        Gaze columns to use. 'average' uses Pro Lab's combined
        ``Gaze point X/Y`` columns.

    Returns
    -------
    pandas.DataFrame
        Samples as from :func:`read_samples`, plus ``trial_type`` taken from
        ``Presented Stimulus name`` (or ``Presented Media name``) when present.
    """
    df = _sniff_table(path, sep='\t')
    tcol = _find_col(df, r'Recording timestamp.*')
    side = '' if eye == 'average' else f'{eye} '
    xcol = _find_col(df, rf'Gaze point {side}X.*')
    ycol = _find_col(df, rf'Gaze point {side}Y.*')
    out = read_samples(df, tcol, xcol, ycol, time_unit=_unit_of(tcol, 'us'),
                       eye={'left': 'L', 'right': 'R'}.get(eye, 'R'))
    scol = _find_col(df, r'Presented Stimulus name', r'Presented Media name', required=False)
    if scol:
        names = df.loc[pd.to_numeric(df[tcol], errors='coerce').notna(), scol].astype(str)
        out['trial_type'] = names.str.replace(r'\.[A-Za-z0-9]+$', '', regex=True).to_numpy()
    return out


def tobii_fixations(path: str, subj: str = '', strip_extension: bool = True) -> pd.DataFrame:
    """Read the fixations of Tobii Pro Lab's fixation filter from a data export (experimental).

    Rows whose ``Eye movement type`` is ``Fixation`` are grouped by
    ``Eye movement type index``. Each fixation's position is Pro Lab's
    ``Fixation point X/Y``, and its stimulus is ``Presented Stimulus name``
    (or ``Presented Media name``). Trials are numbered by the order in which
    stimuli appear. Samples recorded while no stimulus was shown are
    ignored.

    Parameters
    ----------
    path : str
        Tab-separated Pro Lab export that includes the eye-movement-type and
        fixation-point columns.
    subj : str
        Participant identifier.
    strip_extension : bool
        Remove a file extension from stimulus names (``story01.png`` becomes
        ``story01``) so that they match region-file names (default True).

    Returns
    -------
    pandas.DataFrame
        Fixation table in the pyemread schema.
    """
    df = _sniff_table(path, sep='\t')
    tcol = _find_col(df, r'Recording timestamp.*')
    typ = _find_col(df, r'Eye movement type')
    idx = _find_col(df, r'Eye movement type index')
    fx = _find_col(df, r'Fixation point X.*')
    fy = _find_col(df, r'Fixation point Y.*')
    dcol = _find_col(df, r'Gaze event duration.*', required=False)
    scol = _find_col(df, r'Presented Stimulus name', r'Presented Media name', required=False)
    t = _to_ms(pd.to_numeric(df[tcol], errors='coerce'), _unit_of(tcol, 'us'))
    d = df.assign(_t=t)
    d = d[d[typ].astype(str).str.lower() == 'fixation']
    if scol:
        d = d[d[scol].notna() & (d[scol].astype(str).str.len() > 0)]
    g = d.groupby([scol, idx] if scol else [idx], sort=False)
    agg = g.agg(start=('_t', 'min'), end=('_t', 'max'), x=(fx, 'first'), y=(fy, 'first'))
    if dcol:
        agg['end'] = agg['start'] + g[dcol].first().astype(float) * (
            1.0 if _unit_of(dcol, 'ms') == 'ms' else _to_ms(1, _unit_of(dcol, 'ms')))
    agg = agg.reset_index()
    stim = agg[scol].astype(str) if scol else pd.Series([''] * len(agg))
    if strip_extension:
        stim = stim.str.replace(r'\.[A-Za-z0-9]+$', '', regex=True)
    out = _fix_table(agg['start'], agg['end'], pd.to_numeric(agg['x'], errors='coerce'),
                     pd.to_numeric(agg['y'], errors='coerce'), subj=subj,
                     trial_type=stim.to_numpy())
    codes = {s: i + 1 for i, s in enumerate(dict.fromkeys(out['trial_type']))}
    out['trial_id'] = out['trial_type'].map(codes)
    return _finish_fix_table(out)


def pupil_core_surface_to_raw(path: str, screen_w: int, screen_h: int,
                              min_confidence: float = 0.6) -> pd.DataFrame:
    """Read Pupil Core gaze mapped onto a surface (experimental).

    Reads ``gaze_positions_on_surface_<name>.csv`` exported by Pupil Player's
    Surface Tracker, where the surface is the screen showing the text.
    Surface coordinates are normalised with the origin at the bottom left;
    they are converted to screen pixels with the origin at the top left.
    Samples that are off the surface or below ``min_confidence`` are set to
    missing.

    Note that the raw gaze export ``gaze_positions.csv`` is in scene-camera
    coordinates and cannot be used for reading analyses.
    """
    df = _sniff_table(path, sep=',')
    out = pd.DataFrame({'time': _to_ms(df['gaze_timestamp'], 's'),
                        'x': df['x_norm'].astype(float) * screen_w,
                        'y': (1.0 - df['y_norm'].astype(float)) * screen_h})
    bad = ~df['on_surf'].astype(str).str.lower().isin(['true', '1'])
    if 'confidence' in df:
        bad |= df['confidence'].astype(float) < min_confidence
    out.loc[bad.to_numpy(), ['x', 'y']] = np.nan
    out['pupil'] = np.nan
    out['eye'] = 'C'
    out['trial_id'] = 1
    return out


def pupil_core_surface_fixations(path: str, screen_w: int, screen_h: int,
                                 subj: str = '', trial_type: str = '') -> pd.DataFrame:
    """Read Pupil Core fixations mapped onto a surface (experimental).

    Reads ``fixations_on_surface_<name>.csv`` from Pupil Player's Surface
    Tracker. The file has one row per scene-camera frame of each fixation;
    rows are grouped by ``fixation_id`` and only on-surface positions are
    averaged.
    """
    df = _sniff_table(path, sep=',')
    on = df['on_surf'].astype(str).str.lower().isin(['true', '1'])
    g = df[on].groupby('fixation_id', sort=True)
    agg = g.agg(start=('start_timestamp', 'first'), duration=('duration', 'first'),
                x=('norm_pos_x', 'mean'), y=('norm_pos_y', 'mean'))
    start = _to_ms(agg['start'], 's')
    out = _fix_table(start, start + agg['duration'].to_numpy(float),
                     agg['x'].to_numpy(float) * screen_w,
                     (1.0 - agg['y'].to_numpy(float)) * screen_h,
                     subj=subj, trial_type=trial_type, eye='C')
    return _finish_fix_table(out)


def neon_surface_to_raw(path: str, screen_w: int, screen_h: int,
                        origin: str = 'top-left') -> pd.DataFrame:
    """Read Pupil Labs Neon gaze mapped onto a surface (experimental).

    Reads ``gaze.csv`` from a Pupil Cloud Marker Mapper enrichment export
    (columns ``timestamp [ns]``, ``gaze detected on surface``,
    ``gaze position on surface x/y [normalized]``).

    Parameters
    ----------
    origin : {'top-left', 'bottom-left'}
        Origin of the normalised surface coordinates in your export; check
        the documentation of the Pupil Cloud version you used.
    """
    df = _sniff_table(path, sep=',')
    t = _find_col(df, r'timestamp \[ns\]')
    on = _find_col(df, r'gaze detected on surface')
    gx = _find_col(df, r'gaze position on surface x.*')
    gy = _find_col(df, r'gaze position on surface y.*')
    y = df[gy].astype(float)
    out = pd.DataFrame({'time': _to_ms(df[t], 'ns'), 'x': df[gx].astype(float) * screen_w,
                        'y': (y if origin == 'top-left' else 1.0 - y) * screen_h})
    out.loc[~df[on].astype(str).str.lower().isin(['true', '1']).to_numpy(), ['x', 'y']] = np.nan
    out['pupil'] = np.nan
    out['eye'] = 'C'
    out['trial_id'] = 1
    return out


def neon_surface_fixations(path: str, screen_w: int, screen_h: int, subj: str = '',
                           trial_type: str = '', origin: str = 'top-left') -> pd.DataFrame:
    """Read Pupil Labs Neon fixations mapped onto a surface (experimental).

    Reads ``fixations.csv`` from a Pupil Cloud Marker Mapper enrichment
    export (columns ``start timestamp [ns]``, ``end timestamp [ns]``,
    ``fixation detected on surface``, ``fixation x/y [normalized]``).
    The fixations are those of Pupil Labs' detector, which compensates for
    head movement (Drews & Dierkes, 2024). See :func:`neon_surface_to_raw`
    for ``origin``.
    """
    df = _sniff_table(path, sep=',')
    st = _find_col(df, r'start timestamp \[ns\]')
    en = _find_col(df, r'end timestamp \[ns\]')
    on = _find_col(df, r'fixation detected on surface')
    fx = _find_col(df, r'fixation x.*')
    fy = _find_col(df, r'fixation y.*')
    df = df[df[on].astype(str).str.lower().isin(['true', '1'])]
    y = df[fy].astype(float).to_numpy()
    out = _fix_table(_to_ms(df[st], 'ns'), _to_ms(df[en], 'ns'),
                     df[fx].astype(float).to_numpy() * screen_w,
                     (y if origin == 'top-left' else 1.0 - y) * screen_h,
                     subj=subj, trial_type=trial_type, eye='C')
    return _finish_fix_table(out)
