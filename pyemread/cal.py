# -*- coding: utf-8 -*-
"""
Word-level (and optional sentence-level) eye-movement measures for reading.

The functions in this module score fixations that have already been
assigned to text lines and words (by :func:`pyemread.ext.process_subject`,
:func:`pyemread.ext.classify_fixations` or
:func:`pyemread.ext_generic.process_fixations`) against the word areas of
interest of a region file, and return one row per word.

Measures
--------
Whole-trial measures (the same value on every row of a trial):

``tffixos``
    Sum, over all words that received a first-pass fixation, of the
    character position of the word's first first-pass fixation counted from
    the beginning of the text (not of a sentence).
``tffixurt``
    Total first-pass reading time: the sum of ``fpurt`` over all words (ms).
``tfixcnt``
    Number of valid fixations in the trial.
``tregrcnt``
    Number of regressive saccades in the trial: saccades that move leftward
    within a line or to an earlier line. When no saccade table is available
    it is computed from consecutive fixations instead.

Word measures. A word's *first pass* starts with the first fixation on the
word, provided that the eye has not yet moved past it, that is, no word
further along in the text has been fixated before (one or two fixations
further along that are followed by a return to the line being read, such
as a brief glance at a later line, do not count; neither do one or two
fixations on a later line at the start of a trial, e.g. at a central
fixation target, that precede reading). It ends when the eye leaves the
word. A word without a first pass was skipped during first-pass
reading.

``fpurt``
    First-pass reading time (gaze duration): the summed duration of the
    first-pass fixations (ms). ``NaN`` if the word was skipped.
``fpcount``
    Number of first-pass fixations.
``fpregres``
    1 if the eye left the word during first pass with a regression (to an
    earlier part of the same line or to an earlier line), else 0.
``fpregreg``
    ``WordID`` of the word on which that regression landed; 0 if there was
    no regression.
``fpregchr``
    Character position (from the beginning of the text) at which that
    regression landed; the total number of characters if there was no
    regression.
``ffos``
    Character position, within the word's area of interest, of the first
    first-pass fixation (0 = first position). The area of interest of a
    word that follows a space includes that space, so a fixation on its
    first letter has position 1; fixations in the extension at a line edge
    count as position 0.
``ffixurt``
    First-fixation duration: duration of the first first-pass fixation (ms).
``spilover``
    Duration of the fixation that immediately follows first-pass reading of
    the word, wherever it lands (ms).
``rpurt``
    Regression-path (go-past) duration: summed duration of all fixations
    from the first first-pass fixation on the word until the eye first moves
    beyond the word's right edge (ms). Equal to ``fpurt`` if there was no
    first-pass regression.
``rpcount``
    Number of fixations in the regression path, including the first-pass
    fixations.
``rpregreg``
    Smallest ``WordID`` visited during the regression path; 0 if there was
    no first-pass regression.
``rpregchr``
    Character position (from the beginning of the text) of the leftmost
    fixation of the regression path; the total number of characters if
    there was no first-pass regression.
``spurt``
    Second-pass (re-reading) time: summed duration of all fixations on the
    word after its first pass, or of all fixations on the word if it was
    skipped during first pass (ms). ``NaN`` if there were none.
``spcount``
    Number of second-pass fixations.
``tfurt``
    Total fixation time on the word (ms); ``NaN`` if never fixated.
``tfcount``
    Total number of fixations on the word.

Only fixations marked valid (``valid == "yes"``) are scored (see
:func:`pyemread.ext.apply_fixation_validity`). Sentence-level first pass
does not ignore glances. Fixations
that were not assigned to any line (``line_no`` is ``NaN``) are ignored.
Fixations assigned to a line but not to a word end a first pass, and they
end a regression path if they lie beyond the word's right edge.

Examples
--------
>>> from pyemread import ext, cal
>>> regions = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]
>>> ext.process_subject("data", "1950138", regions)        # doctest: +SKIP
>>> cal.cal_write_em("data", "1950138", regions)          # doctest: +SKIP
"""

from __future__ import annotations

__author__ = "Tao Gong"
__copyright__ = "Copyright 2017-2026, Tao Gong"
__credits__ = ["David Braze", "Jonathan Gordils", "Hosung Nam"]
__license__ = "MIT"
__maintainer__ = "Tao Gong"
__email__ = "gtojty@gmail.com"

import fnmatch
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

TRIAL_COLUMNS = ['subj', 'trial_id', 'trial_type', 'trialstart', 'trialend', 'tdur',
                 'recstart', 'recend', 'blinks', 'eye']
WORD_MEASURES = ['fpurt', 'fpcount', 'fpregres', 'fpregreg', 'fpregchr', 'ffos',
                 'ffixurt', 'spilover', 'rpurt', 'rpcount', 'rpregreg', 'rpregchr',
                 'spurt', 'spcount', 'tfurt', 'tfcount']
"""Word-level measures, in output order."""
TRIAL_MEASURES = ['tffixos', 'tffixurt', 'tfixcnt', 'tregrcnt']
"""Whole-trial measures, in output order."""
EM_COLUMNS = (TRIAL_COLUMNS + TRIAL_MEASURES +
              ['region', 'reglen', 'word', 'line_no', 'x1_pos', 'x2_pos', 'mod_x1', 'mod_x2'] +
              WORD_MEASURES)
"""Columns of the per-word output table (``<subj>_EM_<trial_type>_<eye>.csv``)."""


# =============================================================================
# File helpers
# =============================================================================

def _get_reg_df(regfile_dic: Dict[str, str], trial_type: str) -> pd.DataFrame:
    """Region table for a trial type (``<trial_type>.region.csv``)."""
    for name, path in regfile_dic.items():
        if os.path.basename(name) == f'{trial_type}.region.csv':
            return pd.read_csv(path, sep=',', encoding='utf-8')
    raise KeyError(f"No region file for trial type {trial_type!r}; "
                   f"available: {sorted(regfile_dic)}")


def _crt_csv_dic(sit: int, direct: str, subj_id: str,
                 csv_file_type: str) -> Tuple[bool, Dict[str, str]]:
    """Locate ``<subj><csv_file_type>.csv`` files (sit=0: one subject; 1: all)."""
    target = csv_file_type + '.csv'
    dic = {}
    if sit == 0:
        path = os.path.join(direct, subj_id, subj_id + target)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. Create it with ext.process_subject() "
                f"(EyeLink data) or ext_generic.process_fixations() (other data).")
        dic[subj_id] = path
    else:
        for root, _, files in os.walk(direct):
            for name in files:
                if name.endswith(target):
                    dic[name[:-len(target)]] = os.path.join(root, name)
        if not dic:
            raise FileNotFoundError(f"No *{target} files found under {direct}.")
    return True, dic


def _crt_region_dic(direct: str, regfile_name_list: List[str]) -> Tuple[bool, Dict[str, str]]:
    """Locate region files (an empty list means all ``*.region.csv`` in ``direct``)."""
    dic = {}
    if not regfile_name_list:
        for f in os.listdir(direct):
            if fnmatch.fnmatch(f, '*.region.csv'):
                dic[f] = os.path.join(direct, f)
        if not dic:
            raise FileNotFoundError(f"No *.region.csv file found in {direct}.")
    else:
        for f in regfile_name_list:
            path = os.path.join(direct, f)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Region file not found: {path}")
            dic[f] = path
    return True, dic


# =============================================================================
# Core computation
# =============================================================================

def _padded_bounds(reg: pd.DataFrame, add_char_sp: float) -> Tuple[np.ndarray, np.ndarray]:
    """Word bounds widened by ``add_char_sp`` characters at line edges."""
    x1 = reg['x1_pos'].to_numpy(float).copy()
    x2 = reg['x2_pos'].to_numpy(float).copy()
    char_w = (x2[0] - x1[0]) / max(float(reg['length'].iloc[0]), 1.0)
    pad = add_char_sp * char_w
    ln = reg['line_no'].to_numpy()
    first = np.r_[True, ln[1:] != ln[:-1]]
    last = np.r_[ln[1:] != ln[:-1], True]
    x1[first] -= pad
    x2[last] += pad
    return x1, x2


def _regressions_from_fixations(lines: np.ndarray, xs: np.ndarray) -> int:
    ok = np.isfinite(lines)
    l, x = lines[ok], xs[ok]
    return int(np.sum((l[1:] < l[:-1]) | ((l[1:] == l[:-1]) & (x[1:] < x[:-1]))))


def _tregrcnt(sac_df: Optional[pd.DataFrame]) -> Optional[int]:
    """Regressive saccades from the saccade table (None if unavailable)."""
    if sac_df is None or len(sac_df) == 0 or 'line_no' not in sac_df:
        return None
    tot = 0
    for lab, x1, x2 in zip(sac_df['line_no'], sac_df['x1_pos'], sac_df['x2_pos']):
        if isinstance(lab, str) and '_' in lab:
            a, b = lab.split('_')
            tot += int(float(a) > float(b))
        elif not pd.isna(lab):
            tot += int(x1 > x2)
    return tot


def compute_word_measures(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                          sac_df: Optional[pd.DataFrame] = None,
                          add_char_sp: float = 1, subj_id: Optional[str] = None,
                          strict_first_pass: bool = True,
                          ignore_glances: bool = True, max_glance: int = 2) -> pd.DataFrame:
    """Compute word-level reading measures for one trial and one eye.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations of one trial and one eye in temporal order, with
        ``duration``, ``x_pos``, ``line_no`` and ``region_no`` (as produced by
        the line-assignment step) and, optionally, ``valid``.
    reg_df : pandas.DataFrame
        Region table of the stimulus (``WordID``, ``Word``, ``length``,
        ``line_no``, ``x1_pos``, ``x2_pos``).
    sac_df : pandas.DataFrame, optional
        Saccades of the same trial and eye, used for ``tregrcnt``.
    add_char_sp : float
        Character widths added at line edges (default 1).
    subj_id : str, optional
        Participant identifier written to the ``subj`` column.
    strict_first_pass : bool
        If True (default), a word has a first pass only if it is fixated
        before any word further along in the text. If False, the first
        fixation on a word always starts its first pass, as in versions of
        pyemread before 2.2.
    ignore_glances : bool
        If True (default), a brief excursion to a later line (at most
        ``max_glance`` fixations) after which the eye returns to the line it
        was reading does not count as having moved past the words in
        between. Such glances are common at the start of a trial and in
        oral reading. If False, any earlier fixation further along in the
        text does count.
    max_glance : int
        Longest excursion, in fixations, treated as a glance (default 2).

    Returns
    -------
    pandas.DataFrame
        One row per word with the columns in :data:`EM_COLUMNS`.
    """
    if 'line_no' not in fix_df or fix_df['line_no'].notna().sum() == 0:
        raise ValueError(
            "None of the fixations has been assigned to a text line. Assign lines "
            "first, e.g. with ext.process_subject() or ext.cal_crl_sac_fix(..., write=True).")
    reg = reg_df.reset_index(drop=True)
    mod_x1, mod_x2 = _padded_bounds(reg, add_char_sp)
    w_line = reg['line_no'].to_numpy(float)
    w_id = reg['WordID'].to_numpy(float)
    w_len = reg['length'].to_numpy(float)
    chars_before = np.r_[0.0, np.cumsum(w_len)[:-1]]
    total_chars = float(w_len.sum())
    id_to_idx = {wid: j for j, wid in enumerate(w_id)}

    fx = fix_df.reset_index(drop=True)
    if 'valid' in fx:
        fx = fx[fx['valid'].astype(str).str.lower().isin(['yes', '1', 'true', '1.0'])]
    fx = fx[fx['line_no'].notna()].reset_index(drop=True)
    L = fx['line_no'].to_numpy(float)
    X = fx['x_pos'].to_numpy(float)
    R = fx['region_no'].to_numpy(float)
    D = fx['duration'].to_numpy(float)
    n = len(fx)

    def charpos(i: int) -> float:
        if not np.isfinite(R[i]) or R[i] not in id_to_idx:
            return np.nan
        j = id_to_idx[R[i]]
        width = mod_x2[j] - mod_x1[j]
        within = max(np.ceil((X[i] - mod_x1[j]) / width * w_len[j]) - 1, 0) if width > 0 else 0
        return chars_before[j] + within

    # position of each fixation in reading order: (line, x)
    out = {m: np.full(len(reg), np.nan) for m in WORD_MEASURES}
    # furthest point reached so far (line, x) before each fixation
    reached_line = np.full(n, -np.inf)
    reached_x = np.full(n, -np.inf)
    cur_l, cur_x = -np.inf, -np.inf
    ahead = lambda i, l0, x0: (L[i] > l0) or (L[i] == l0 and X[i] > x0)
    i = 0
    if ignore_glances and n > 1:
        # A trial may begin with a glance at a later line (e.g. at a central
        # fixation target) before reading starts on an earlier line; such
        # initial fixations do not move the reading position either.
        for k in range(1, min(max_glance, n - 1) + 1):
            if np.all(L[:k] > L[k]):
                i = k
                break
    while i < n:
        reached_line[i], reached_x[i] = cur_l, cur_x
        if ignore_glances and np.isfinite(cur_l) and L[i] > cur_l:
            # Excursion to a later line: if the eye comes back to the current
            # line (or an earlier one) within `max_glance` fixations, the
            # excursion is a glance and does not move the reading position.
            k = i
            while k < n and k - i < max_glance and L[k] > cur_l:
                k += 1
            if k < n and k - i <= max_glance and L[k] <= cur_l:
                for m in range(i, k):
                    reached_line[m], reached_x[m] = cur_l, cur_x
                i = k
                continue
        if ahead(i, cur_l, cur_x):
            cur_l, cur_x = L[i], X[i]
        i += 1

    for j in range(len(reg)):
        wid, wl, a, b = w_id[j], w_line[j], mod_x1[j], mod_x2[j]
        on = np.where(R == wid)[0]
        if len(on):
            out['tfurt'][j] = D[on].sum()
            out['tfcount'][j] = len(on)
        if len(on) == 0:
            continue
        f0 = on[0]
        passed = (reached_line[f0] > wl) or (reached_line[f0] == wl and reached_x[f0] > b)
        if strict_first_pass and passed:
            sp = on
            out['spurt'][j] = D[sp].sum()
            out['spcount'][j] = len(sp)
            continue
        k = f0
        while k + 1 < n and R[k + 1] == wid:
            k += 1
        run = np.arange(f0, k + 1)
        end = k + 1 if k + 1 < n else None
        out['fpurt'][j] = D[run].sum()
        out['fpcount'][j] = len(run)
        out['ffixurt'][j] = D[f0]
        width = b - a
        out['ffos'][j] = (max(np.ceil((X[f0] - a) / width * w_len[j]) - 1, 0)
                          if width > 0 else 0)
        if end is not None:
            out['spilover'][j] = D[end]
        regress = end is not None and ((L[end] < wl) or (L[end] == wl and X[end] < a))
        if regress:
            out['fpregres'][j] = 1
            out['fpregreg'][j] = R[end] if np.isfinite(R[end]) else np.nan
            out['fpregchr'][j] = charpos(end)
            m = end
            while m < n and not ((L[m] > wl) or (L[m] == wl and X[m] > b)):
                m += 1
            path = np.arange(f0, m)
            out['rpurt'][j] = D[path].sum()
            out['rpcount'][j] = len(path)
            regs = R[path]
            if np.isfinite(regs).any():
                left = path[np.nanargmin(np.where(np.isfinite(regs), regs, np.inf))]
                out['rpregreg'][j] = R[left]
                out['rpregchr'][j] = charpos(left)
        else:
            out['fpregres'][j] = 0
            out['fpregreg'][j] = 0
            out['fpregchr'][j] = total_chars
            out['rpurt'][j] = out['fpurt'][j]
            out['rpcount'][j] = out['fpcount'][j]
            out['rpregreg'][j] = 0
            out['rpregchr'][j] = total_chars
        later = on[on > k]
        if len(later):
            out['spurt'][j] = D[later].sum()
            out['spcount'][j] = len(later)

    em = pd.DataFrame({
        'region': reg['WordID'], 'reglen': reg['length'], 'word': reg['Word'],
        'line_no': reg['line_no'], 'x1_pos': reg['x1_pos'], 'x2_pos': reg['x2_pos'],
        'mod_x1': mod_x1, 'mod_x2': mod_x2,
    })
    for m in WORD_MEASURES:
        em[m] = out[m]
    src = fix_df.iloc[0] if len(fix_df) else pd.Series(dtype=object)
    for c in TRIAL_COLUMNS:
        em[c] = src.get(c, np.nan) if c != 'subj' else (subj_id if subj_id is not None
                                                         else src.get('subj', np.nan))
    ffos_ok = np.isfinite(out['ffos'])
    em['tffixos'] = float(np.sum(chars_before[ffos_ok] + out['ffos'][ffos_ok]))
    em['tffixurt'] = float(np.nansum(out['fpurt']))
    em['tfixcnt'] = int(n)
    t = _tregrcnt(sac_df)
    em['tregrcnt'] = t if t is not None else _regressions_from_fixations(L, X)
    return em[EM_COLUMNS]


# =============================================================================
# Sentence-level measures
# =============================================================================

_SENT_END = re.compile(r'[.!?。！？]["\'”’)\]]*\s*$')


def sentence_ids(reg_df: pd.DataFrame) -> np.ndarray:
    """Sentence number of each word.

    Uses the region table's ``sentence_id`` column when it exists; otherwise
    a new sentence starts after every word ending in ``.``, ``!``, ``?`` (or
    the full-width forms), optionally followed by closing quotes or
    brackets.
    """
    if 'sentence_id' in reg_df:
        return reg_df['sentence_id'].to_numpy()
    ids, cur = [], 1
    for w in reg_df['Word'].astype(str):
        ids.append(cur)
        if _SENT_END.search(w.strip()):
            cur += 1
    return np.array(ids)


def compute_sentence_measures(fix_df: pd.DataFrame, reg_df: pd.DataFrame) -> pd.DataFrame:
    """Sentence-level reading measures for one trial and one eye.

    Each sentence is treated as a single area of interest made up of its
    words. Fixations that are not on a word are ignored.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations with ``region_no`` assigned (one trial, one eye).
    reg_df : pandas.DataFrame
        Region table; an optional ``sentence_id`` column defines the
        sentences (see :func:`sentence_ids`).

    Returns
    -------
    pandas.DataFrame
        One row per sentence: ``sentence_id``, ``first_word``, ``last_word``,
        ``n_words``, ``fpurt`` (first-pass reading time), ``fpcount``,
        ``fpregres`` (first pass ended with a regression), ``rpurt``
        (regression-path duration), ``spurt`` (re-reading time),
        ``tfurt`` (total reading time) and ``tfcount``. First-pass measures
        are ``NaN`` for a sentence that was skipped during first pass.
    """
    reg = reg_df.reset_index(drop=True)
    sid = sentence_ids(reg)
    w2s = dict(zip(reg['WordID'].astype(float), sid))
    order = {s: k for k, s in enumerate(dict.fromkeys(sid))}
    fx = fix_df.reset_index(drop=True)
    if 'valid' in fx:
        fx = fx[fx['valid'].astype(str).str.lower().isin(['yes', '1', 'true', '1.0'])]
    fx = fx[fx['region_no'].notna()].reset_index(drop=True)
    S = np.array([order[w2s[r]] if r in w2s else -1 for r in fx['region_no'].astype(float)])
    D = fx['duration'].to_numpy(float)
    rows = []
    furthest = np.maximum.accumulate(np.r_[-1, S[:-1]]) if len(S) else S
    for s, k in order.items():
        words = reg['WordID'][sid == s]
        on = np.where(S == k)[0]
        row = {'sentence_id': s, 'first_word': int(words.min()), 'last_word': int(words.max()),
               'n_words': int(len(words)), 'fpurt': np.nan, 'fpcount': np.nan,
               'fpregres': np.nan, 'rpurt': np.nan, 'spurt': np.nan,
               'tfurt': float(D[on].sum()) if len(on) else np.nan, 'tfcount': int(len(on))}
        if len(on):
            f0 = on[0]
            if furthest[f0] <= k:
                e = f0
                while e + 1 < len(S) and S[e + 1] == k:
                    e += 1
                row['fpurt'] = float(D[f0:e + 1].sum())
                row['fpcount'] = int(e + 1 - f0)
                nxt = e + 1 if e + 1 < len(S) else None
                row['fpregres'] = int(nxt is not None and S[nxt] < k)
                m = e + 1
                while m < len(S) and S[m] <= k:
                    m += 1
                row['rpurt'] = float(D[f0:m].sum())
                later = on[on > e]
                row['spurt'] = float(D[later].sum()) if len(later) else np.nan
            else:
                row['spurt'] = float(D[on].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# User functions
# =============================================================================

def _iter_trials(fix_df: pd.DataFrame):
    for trial_id in list(dict.fromkeys(fix_df['trial_id'])):
        f_trial = fix_df[fix_df['trial_id'] == trial_id]
        for eye in list(dict.fromkeys(f_trial['eye'])):
            yield trial_id, eye, f_trial[f_trial['eye'] == eye].reset_index(drop=True)


def cal_write_em(direct: str, subj_id: str, regfile_name_list: List[str],
                 add_char_sp: int = 1, fix_df: Optional[pd.DataFrame] = None,
                 sac_df: Optional[pd.DataFrame] = None, write: bool = True,
                 strict_first_pass: bool = True, ignore_glances: bool = True) -> pd.DataFrame:
    """
    Calculate (and write) word-level eye-movement measures for a participant.

    Parameters
    ----------
    direct : str
        Data directory: region files are read from it, and participant files
        from (and results written to) its sub-folder ``subj_id``.
    subj_id : str
        Participant identifier (e.g. ``"1950138"``).
    regfile_name_list : list of str
        Region files, one per stimulus (trial type).
    add_char_sp : int
        Character widths added at line edges to catch overshoot fixations
        (default 1).
    fix_df, sac_df : pandas.DataFrame, optional
        Classified fixation and saccade tables. If omitted, they are read
        from ``<subj_id>_Fix.csv`` and ``<subj_id>_Sac.csv``, which must have
        been written after line assignment (e.g. by
        :func:`pyemread.ext.process_subject`).
    write : bool
        Write one CSV per trial and eye,
        ``<subj_id>_EM_<trial_type>_<eye>.csv`` (default True).
    strict_first_pass, ignore_glances : bool
        See :func:`compute_word_measures`.

    Returns
    -------
    pandas.DataFrame
        All per-word rows of the participant (all trials and eyes).

    Raises
    ------
    ValueError
        If the fixations have not been assigned to lines.
    """
    _, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    if fix_df is None:
        _, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
        fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',', encoding='utf-8')
        if sac_df is None:
            try:
                _, sac_file_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
                sac_df = pd.read_csv(sac_file_dic[subj_id], sep=',', encoding='utf-8')
            except FileNotFoundError:
                sac_df = None
    if fix_df['line_no'].notna().sum() == 0:
        raise ValueError(
            f"The fixations of participant {subj_id} have not been assigned to text lines, "
            f"so every measure would be empty. Run ext.process_subject() (or "
            f"ext.cal_crl_sac_fix(..., write=True)) before cal_write_em(), or pass the "
            f"classified tables with fix_df= and sac_df=.")
    results = []
    for trial_id, eye, fe in _iter_trials(fix_df):
        trial_type = str(fe['trial_type'].iloc[0])
        reg_df = _get_reg_df(regfile_dic, trial_type)
        se = None
        if sac_df is not None and len(sac_df):
            se = sac_df[(sac_df['trial_id'] == trial_id) & (sac_df['eye'] == eye)]
        _log.info(f'Cal EM measures: Subj {subj_id}, trial {trial_id}, eye {eye}')
        em = compute_word_measures(fe, reg_df, se, add_char_sp, subj_id, strict_first_pass,
                                   ignore_glances)
        if write:
            os.makedirs(os.path.join(direct, subj_id), exist_ok=True)
            em.to_csv(os.path.join(direct, subj_id, f'{subj_id}_EM_{trial_type}_{eye}.csv'),
                      index=False, encoding='utf-8')
        results.append(em)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=EM_COLUMNS)


def cal_write_sm(direct: str, subj_id: str, regfile_name_list: List[str],
                 fix_df: Optional[pd.DataFrame] = None, write: bool = True) -> pd.DataFrame:
    """
    Calculate (and write) sentence-level measures for a participant.

    Sentences are taken from an optional ``sentence_id`` column of each
    region file or, if absent, segmented at sentence-final punctuation
    (see :func:`sentence_ids`). Results are written to
    ``<subj_id>_SM_<trial_type>_<eye>.csv``.

    Parameters
    ----------
    direct, subj_id, regfile_name_list
        As in :func:`cal_write_em`.
    fix_df : pandas.DataFrame, optional
        Classified fixations; read from ``<subj_id>_Fix.csv`` if omitted.
    write : bool
        Write the CSV files (default True).

    Returns
    -------
    pandas.DataFrame
        One row per sentence, trial and eye.
    """
    _, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    if fix_df is None:
        _, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
        fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',', encoding='utf-8')
    if fix_df['region_no'].notna().sum() == 0:
        raise ValueError("No fixation has been assigned to a word; assign lines and words first.")
    results = []
    for trial_id, eye, fe in _iter_trials(fix_df):
        trial_type = str(fe['trial_type'].iloc[0])
        sm = compute_sentence_measures(fe, _get_reg_df(regfile_dic, trial_type))
        sm.insert(0, 'eye', eye)
        sm.insert(0, 'trial_type', trial_type)
        sm.insert(0, 'trial_id', trial_id)
        sm.insert(0, 'subj', subj_id)
        if write:
            sm.to_csv(os.path.join(direct, subj_id, f'{subj_id}_SM_{trial_type}_{eye}.csv'),
                      index=False, encoding='utf-8')
        results.append(sm)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def cal_write_em_b(direct: str, regfile_name_list: List[str], add_char_sp: int = 1) -> None:
    """
    Calculate and write word-level measures for every participant in ``direct``.

    Every sub-folder containing a ``*_Fix.csv`` file is processed with
    :func:`cal_write_em`.
    """
    _, fix_file_dic = _crt_csv_dic(1, direct, '', '_Fix')
    for subj_id in fix_file_dic:
        cal_write_em(direct, subj_id, regfile_name_list, add_char_sp)


def merge_csv(
    direct: str,
    regfile_name_list: List[str],
    subj_id: str
) -> None:
    """
    Merge eye-movement timestamp data with audio CSV file.
    
    Parameters
    ----------
    direct : str
        Directory containing results (each subject in subfolder).
    regfile_name_list : list
        List of region file names.
    subj_id : str
        Subject ID.
    
    Output
    ------
    Creates {subj_id}_Merge.csv
    """
    stamp_file_exist, stamp_file_dic = _crt_csv_dic(0, direct, subj_id, '_Stamp')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if not (stamp_file_exist and regfile_exist):
        return
    
    _log.info(f'SubjID: {subj_id}')
    merge_dfs = []
    
    emdf = pd.read_csv(stamp_file_dic[subj_id], sep=',', encoding='utf-8')
    emdf['gaze_time'] = emdf.time - emdf.recstart
    emdf['audio_time'] = np.nan
    emdf['audio_label'] = np.nan
    emdf['audio_region_no'] = np.nan
    
    trial_list = list(np.unique(emdf.trial_type))
    
    for trial in trial_list:
        _log.info(f'Processing Trial: {trial}')
        
        # Get region file
        reg_df = pd.read_csv(regfile_dic[trial + '.region.csv'], encoding='utf-8')
        
        # Get ETime file
        aufile = os.path.join(direct, subj_id, f'{subj_id}-{trial}_ETime.csv')
        if not os.path.isfile(aufile):
            _log.info(f'{aufile} does not exist!')
            continue
        
        audf = pd.read_csv(aufile, sep=',', header=None, encoding='utf-8')
        audf.columns = ['audio_label', 'audio_time']
        audf.audio_label = audf.audio_label.str.lower()
        audf.loc[audf.audio_label == 'sp', 'audio_label'] = np.nan
        audf.audio_time = audf.audio_time.astype(float) * 1000
        
        # Find merge point
        emdf_temp = emdf[emdf.trial_type == trial].reset_index(drop=True)
        
        for ind in range(len(emdf_temp) - 1):
            if (emdf_temp.gaze_time[ind] < audf.audio_time[0] and 
                emdf_temp.gaze_time[ind + 1] >= audf.audio_time[0]):
                for ind2 in range(ind + 1, min(ind + 1 + len(audf), len(emdf_temp))):
                    emdf_temp.loc[ind2, 'audio_time'] = audf.audio_time[ind2 - ind - 1]
                    emdf_temp.loc[ind2, 'audio_label'] = audf.audio_label[ind2 - ind - 1]
                break
        
        # Add audio_region_no (only for error_free trials)
        if emdf_temp.error_free.iloc[0] == 1:
            cur_region = 1
            cur_label = list(reg_df.Word[reg_df.WordID == cur_region])[0]
            ind = 0
            
            while ind < len(emdf_temp):
                while ind < len(emdf_temp) and emdf_temp.loc[ind, 'audio_label'] != cur_label:
                    ind += 1
                while ind < len(emdf_temp) and emdf_temp.loc[ind, 'audio_label'] == cur_label:
                    emdf_temp.loc[ind, 'audio_region_no'] = cur_region
                    ind += 1
                cur_region += 1
                if cur_region in list(reg_df.WordID):
                    cur_label = list(reg_df.Word[reg_df.WordID == cur_region])[0]
        
        merge_dfs.append(emdf_temp)
    
    # Combine all trials
    if merge_dfs:
        merge_df = pd.concat(merge_dfs, ignore_index=True)
        merge_df = merge_df.sort_values(by=['trial_id', 'time'], ascending=True)
        merge_file_name = os.path.join(direct, subj_id, f'{subj_id}_Merge.csv')
        merge_df.to_csv(merge_file_name, index=False, encoding='utf-8')


def merge_csv_b(direct: str, regfile_name_list: List[str]) -> None:
    """
    Batch merge CSV files for all subjects.
    
    Parameters
    ----------
    direct : str
        Directory containing results.
    regfile_name_list : list
        List of region file names.
    
    Output
    ------
    Creates {subj_id}_Merge.csv for each subject.
    """
    stamp_file_exist, stamp_file_dic = _crt_csv_dic(1, direct, '', '_Stamp')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if stamp_file_exist and regfile_exist:
        for subj_id in stamp_file_dic.keys():
            merge_csv(direct, regfile_name_list, subj_id)

# =============================================================================
# Names used in earlier versions
# =============================================================================

cal_write_EM = cal_write_em
cal_write_EM_b = cal_write_em_b
mergeCSV = merge_csv
mergeCSV_b = merge_csv_b

__all__ = [
    'compute_word_measures', 'compute_sentence_measures', 'sentence_ids',
    'cal_write_em', 'cal_write_em_b', 'cal_write_sm', 'merge_csv', 'merge_csv_b',
    'WORD_MEASURES', 'TRIAL_MEASURES', 'EM_COLUMNS',
    'cal_write_EM', 'cal_write_EM_b', 'mergeCSV', 'mergeCSV_b',
]
