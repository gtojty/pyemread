# -*- coding: utf-8 -*-
"""
EyeLink (SR Research) ASC parsing and fixation-to-line classification.

This module reads the ASC files produced by SR Research's ``edf2asc``
utility, extracts per-trial fixations, saccades and samples, assigns
fixations to text lines and words, and writes the results as CSV tables
that :mod:`pyemread.cal` turns into word-level reading measures.

The recommended entry point is :func:`process_subject`, which reads one
participant's ASC file, classifies every trial and writes the tables in a
single call. The lower-level functions are retained for compatibility
with earlier versions; their camelCase names (e.g. ``cal_crlSacFix``) are
kept as aliases of the snake_case names.

Examples
--------
>>> from pyemread import ext, cal
>>> ext.process_subject("data", "1950138",
...                     ["story01.region.csv", "story02.region.csv",
...                      "story03.region.csv"])          # doctest: +SKIP
>>> cal.cal_write_em("data", "1950138",
...                  ["story01.region.csv", "story02.region.csv",
...                   "story03.region.csv"])           # doctest: +SKIP
"""

__author__ = "Tao Gong"
__copyright__ = "Copyright 2017-2026, Tao Gong"
__credits__ = ["David Braze", "Jonathan Gordils", "Hosung Nam"]
__license__ = "MIT"
__maintainer__ = "Tao Gong"
__email__ = "gtojty@gmail.com"

import logging
import os
import sys
import fnmatch
import re
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import pandas as pd
import numpy as np

from . import lines as _lines

_log = logging.getLogger(__name__)

TRIAL_TYPE_VAR = 'picture_name'
"""Name of the EyeLink trial variable (``!V TRIAL_VAR <name> <value>``) whose
value, without file extension, identifies the stimulus of a trial and hence
its region file (``<value>.region.csv``). Change it if your experiment uses
another variable, e.g. ``ext.TRIAL_TYPE_VAR = "stimulus"``."""


# -----------------------------------------------------------------------------
# Helper functions for obtaining basic information from data files
# -----------------------------------------------------------------------------

def _get_header(lines: List[str]) -> Tuple[str, str, str]:
    """
    Extract header information from data lines.
    
    Args:
        lines: Data lines from ASC file
        
    Returns:
        Tuple of (script, session_date, source_file)
    """
    header = []
    script = ""
    sessdate = ""
    srcfile = ""
    
    for line in lines:
        line = line.rstrip()
        if re.search(r'^[*][*] ', line):
            header.append(line)
    
    for line in header:
        if re.search('RECORDED BY', line):
            parts = line.split(' ')
            if len(parts) > 3:
                script = parts[3]
        if re.search('DATE:', line):
            parts = line.split(': ')
            if len(parts) > 1:
                sessdate = parts[1]
        if re.search('CONVERTED FROM', line):
            m = re.search(r' FROM (.+?) using', line)
            if m:
                srcfile = m.group(1).split('\\')[-1]
    
    return script, sessdate, srcfile


def _get_line_info(fix_rep_cur: pd.DataFrame) -> Tuple[List[int], List[List[float]]]:
    """
    Get time information from fixation report lines.
    
    Args:
        fix_rep_cur: Current trial's fix report DataFrame
        
    Returns:
        Tuple of (line_indices, line_times)
    """
    totlines = int(max(fix_rep_cur.CURRENT_FIX_LABEL))
    line_idx = []
    line_time = []
    
    for cur in range(1, totlines):
        line_idx.append(cur)
        sub_fix_rep = fix_rep_cur[fix_rep_cur.CURRENT_FIX_LABEL == cur].reset_index()
        line_time.append([
            sub_fix_rep.loc[0, 'CURRENT_FIX_START'],
            sub_fix_rep.loc[len(sub_fix_rep) - 1, 'CURRENT_FIX_END']
        ])
    
    return line_idx, line_time


def _get_trial_reg(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the region (line range) of each trial.
    
    Args:
        lines: Data lines from ASC file
        
    Returns:
        Tuple of (trial_indices, trial_line_numbers)
    """
    trial_start = []
    trial_start_lines = []
    trial_end = []
    trial_end_lines = []
    
    for cur, line in enumerate(lines):
        if re.search('TRIALID', line):
            trial_start.append(line)
            trial_start_lines.append(cur)
        if re.search('TRIAL_RESULT', line):
            trial_end.append(line)
            trial_end_lines.append(cur)
    
    if len(trial_start) != len(trial_end):
        raise ValueError("Trial starting and ending mismatch!")
    
    T_idx = np.column_stack((trial_start, trial_end))
    T_lines = np.column_stack((trial_start_lines, trial_end_lines))
    
    return T_idx, T_lines


def _get_blink_fix_sac_sampfreq_eyerec(triallines: List[str], 
                                       datatype: int) -> Union[
                                           Tuple[List, List, List, int, str],
                                           Tuple[List, List, int, str]]:
    """
    Get split blink, fixation, and saccade data lines, sampling frequency and eye recorded.
    
    Args:
        triallines: Data lines of a trial
        datatype: 0 for fixation/saccade, 1 for time-stamped data
        
    Returns:
        For datatype=0: (blink_lines, fix_lines, sac_lines, sampfreq, eyerec)
        For datatype=1: (blink_lines, stamp_lines, sampfreq, eyerec)
    """
    blinklines = []
    sampfreq = 250  # Default
    eyerec = 'R'    # Default
    
    if datatype == 0:
        fixlines = []
        saclines = []
        for line in triallines:
            if re.search('^EBLINK', line):
                blinklines.append(line.split())
            if re.search('^EFIX', line):
                fixlines.append(line.split())
            if re.search('^ESACC', line):
                saclines.append(line.split())
            if re.search('!MODE RECORD', line):
                parts = line.split()
                sampfreq = int(parts[5])
                eyerec = parts[-1]
        return blinklines, fixlines, saclines, sampfreq, eyerec
    
    elif datatype == 1:
        stamplines = []
        for line in triallines:
            if re.search('^EBLINK', line):
                blinklines.append(line.split())
            if re.search(r'^[0-9]', line):
                stamplines.append(line.split())
            if re.search('!MODE RECORD', line):
                parts = line.split()
                sampfreq = int(parts[5])
                eyerec = parts[-1]
        return blinklines, stamplines, sampfreq, eyerec


def _get_tdur(triallines: List[str]) -> Tuple[Any, int, int, int, int, int]:
    """
    Get estimated trial duration.
    
    Args:
        triallines: Data lines of a trial
        
    Returns:
        Tuple of (trial_type, trialstart, trialend, tdur, recstart, recend)
    """
    trial_type = np.nan
    trialstart = 0
    trialend = 0
    tdur = 0
    recstart = 0
    recend = 0
    
    for line in triallines:
        if re.search(rf'!V TRIAL_VAR {TRIAL_TYPE_VAR}\b', line):
            trial_type = line.split()[-1].split('.')[0]
        if re.search('^START', line):
            trialstart = int(line.split()[1])
        if re.search('^END', line):
            trialend = int(line.split()[1])
        if re.search('ARECSTART', line):
            parts = line.split()
            recstart = int(parts[1]) - int(parts[2])
        if re.search('ARECSTOP', line):
            parts = line.split()
            recend = int(parts[1]) - int(parts[2])
    
    tdur = trialend - trialstart
    return trial_type, trialstart, trialend, tdur, recstart, recend


def _get_error_free(etran_df: pd.DataFrame, subj_id: str, trial_type: str) -> int:
    """
    Get trial_type's error_free status.
    
    Args:
        etran_df: ETRAN DataFrame
        subj_id: Subject ID
        trial_type: Trial type
        
    Returns:
        Error-free status (1 or 0)
    """
    col_name = f'etRan_{trial_type}ErrorFree'
    mask = etran_df.SubjectID == f'a{subj_id}'
    if col_name in etran_df.columns:
        return int(etran_df.loc[mask, col_name].iloc[0])
    return 1


def _get_reg_df(regfile_dic: Dict[str, str], trial_type: str) -> pd.DataFrame:
    """
    Get the region file DataFrame based on trial type.
    
    Args:
        regfile_dic: Dictionary of region file names and paths
        trial_type: Current trial type
        
    Returns:
        Region file DataFrame
        
    Raises:
        ValueError: If trial_type is invalid
    """
    regfile_name = f'{trial_type}.region.csv'
    if regfile_name not in regfile_dic:
        raise ValueError(f"Invalid trial_type: {trial_type}")
    return pd.read_csv(regfile_dic[regfile_name], sep=',')


# -----------------------------------------------------------------------------
# Helper functions for crossline information based on region files
# -----------------------------------------------------------------------------

def _get_crossline_info(reg_df: pd.DataFrame) -> List[Dict]:
    """
    Get cross line information from region file.
    
    Args:
        reg_df: Region file DataFrame
        
    Returns:
        List of dictionaries marking the cross line information
    """
    crossline_info = []
    
    for ind in range(len(reg_df) - 1):
        if reg_df.line_no[ind] + 1 == reg_df.line_no[ind + 1]:
            dic = {
                'p': reg_df.loc[ind, 'line_no'],
                'p_x': (reg_df.loc[ind, 'x1_pos'] + reg_df.loc[ind, 'x2_pos']) / 2.0,
                'p_y': (reg_df.loc[ind, 'y1_pos'] + reg_df.loc[ind, 'y2_pos']) / 2.0,
                'n': reg_df.loc[ind + 1, 'line_no'],
                'n_x': (reg_df.loc[ind + 1, 'x1_pos'] + reg_df.loc[ind + 1, 'x2_pos']) / 2.0,
                'n_y': (reg_df.loc[ind + 1, 'y1_pos'] + reg_df.loc[ind + 1, 'y2_pos']) / 2.0
            }
            crossline_info.append(dic)
    
    return crossline_info


# -----------------------------------------------------------------------------
# Helper functions for lumping short fixations (< 50ms)
# -----------------------------------------------------------------------------

def _lump_two_fix(df: pd.DataFrame, ind1: int, ind2: int, direc: int, addtime: float) -> None:
    """
    Lump two adjacent fixation data lines.
    
    Args:
        df: Fixation DataFrame (modified in place)
        ind1: First fixation index
        ind2: Second fixation index
        direc: Direction (1 for next, -1 for previous)
        addtime: Adjusting time for duration
    """
    if direc == 1:
        if ind1 >= ind2:
            raise ValueError('Wrong direction in lumping!')
        df.loc[ind1, 'end_time'] = df.loc[ind2, 'end_time']
        df.loc[ind1, 'duration'] = df.loc[ind1, 'end_time'] - df.loc[ind1, 'start_time'] + addtime
        df.loc[ind1, 'x_pos'] = (df.loc[ind1, 'x_pos'] + df.loc[ind2, 'x_pos']) / 2.0
        df.loc[ind1, 'y_pos'] = (df.loc[ind1, 'y_pos'] + df.loc[ind2, 'y_pos']) / 2.0
        df.loc[ind1, 'pup_size'] = (df.loc[ind1, 'pup_size'] + df.loc[ind2, 'pup_size']) / 2.0
    elif direc == -1:
        if ind1 <= ind2:
            raise ValueError('Wrong direction in lumping!')
        df.loc[ind1, 'start_time'] = df.loc[ind2, 'start_time']
        df.loc[ind1, 'duration'] = df.loc[ind1, 'end_time'] - df.loc[ind1, 'start_time'] + addtime
        df.loc[ind1, 'x_pos'] = (df.loc[ind1, 'x_pos'] + df.loc[ind2, 'x_pos']) / 2.0
        df.loc[ind1, 'y_pos'] = (df.loc[ind1, 'y_pos'] + df.loc[ind2, 'y_pos']) / 2.0
        df.loc[ind1, 'pup_size'] = (df.loc[ind1, 'pup_size'] + df.loc[ind2, 'pup_size']) / 2.0


def _lump_more_fix(df: pd.DataFrame, ind: int, ind_list: List[int], addtime: float) -> None:
    """
    Lump multiple fixations.
    
    Args:
        df: Fixation DataFrame (modified in place)
        ind: Target fixation index
        ind_list: List of fixation indices to lump
        addtime: Adjusting time for duration
    """
    df.loc[ind, 'end_time'] = df.loc[ind_list[-1], 'end_time']
    df.loc[ind, 'duration'] = df.loc[ind, 'end_time'] - df.loc[ind, 'start_time'] + addtime
    
    for item in ind_list:
        df.loc[ind, 'x_pos'] += df.loc[item, 'x_pos']
        df.loc[ind, 'y_pos'] += df.loc[item, 'y_pos']
        df.loc[ind, 'pup_size'] += df.loc[item, 'pup_size']
    
    df.loc[ind, 'x_pos'] /= float(len(ind_list) + 1)
    df.loc[ind, 'y_pos'] /= float(len(ind_list) + 1)
    df.loc[ind, 'pup_size'] /= float(len(ind_list) + 1)


def _lump_fix(df: pd.DataFrame, endindex: int, short_index: List[int], 
              addtime: float, ln: int, zn: int) -> pd.DataFrame:
    """
    Lump fixations that are too short.
    
    Args:
        df: Fixation DataFrame
        endindex: Upper bound of searching range
        short_index: List of indices of fixations with short duration
        addtime: Adjusting time calculated from sampling frequency
        ln: Maximum duration of a fixation to "lump" (default 50)
        zn: Maximum distance between two fixations for "lumping" (default 50)
        
    Returns:
        Modified fixation DataFrame
    """
    droplist = []
    cur = 0
    
    while cur < len(short_index):
        if short_index[cur] == 0:
            # First fixation - check next
            next_list = []
            ind = cur + 1
            while (ind < len(short_index) and 
                   short_index[ind] == short_index[ind - 1] + 1 and 
                   abs(df.x_pos[short_index[ind]] - df.x_pos[short_index[cur]]) <= zn):
                next_list.append(short_index[ind])
                ind += 1
            
            if next_list:
                _lump_more_fix(df, short_index[cur], next_list, addtime)
                droplist.extend(next_list)
                
                if df.duration[short_index[cur]] <= ln:
                    if (next_list[-1] + 1 <= endindex and 
                        abs(df.x_pos[short_index[cur]] - df.x_pos[next_list[-1] + 1]) <= zn):
                        _lump_two_fix(df, short_index[cur], next_list[-1] + 1, 1, addtime)
                        droplist.append(next_list[-1] + 1)
                
                cur += len(next_list)
            else:
                if (short_index[cur] + 1 <= endindex and 
                    abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] + 1]) <= zn):
                    _lump_two_fix(df, short_index[cur], short_index[cur] + 1, 1, addtime)
                    droplist.append(short_index[cur] + 1)
        
        elif short_index[cur] == endindex:
            # Last fixation - check previous
            if (short_index[cur] - 1 not in droplist and 
                abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1]) <= zn):
                _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                droplist.append(short_index[cur] - 1)
        
        else:
            # Middle fixation - check both
            next_list = []
            ind = cur + 1
            while (ind < len(short_index) and 
                   short_index[ind] == short_index[ind - 1] + 1 and 
                   abs(df.x_pos[short_index[ind]] - df.x_pos[short_index[cur]]) <= zn):
                next_list.append(short_index[ind])
                ind += 1
            
            if next_list:
                _lump_more_fix(df, short_index[cur], next_list, addtime)
                droplist.extend(next_list)
                
                if df.duration[short_index[cur]] <= ln:
                    dist_next = 0.0
                    dist_prev = 0.0
                    
                    if (next_list[-1] + 1 <= endindex and 
                        next_list[-1] + 1 not in droplist and 
                        abs(df.x_pos[short_index[cur]] - df.x_pos[next_list[-1] + 1]) <= zn):
                        dist_next = abs(df.x_pos[short_index[cur]] - df.x_pos[next_list[-1] + 1])
                    
                    if (short_index[cur] - 1 not in droplist and 
                        abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1]) <= zn):
                        dist_prev = abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1])
                    
                    if dist_next != 0.0 and dist_prev == 0.0:
                        _lump_two_fix(df, short_index[cur], next_list[-1] + 1, 1, addtime)
                        droplist.append(next_list[-1] + 1)
                    elif dist_next == 0.0 and dist_prev != 0.0:
                        _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                        droplist.append(short_index[cur] - 1)
                    elif dist_next != 0.0 and dist_prev != 0.0:
                        if dist_next < dist_prev:
                            _lump_two_fix(df, short_index[cur], next_list[-1] + 1, 1, addtime)
                            droplist.append(next_list[-1] + 1)
                            if (df.duration[short_index[cur]] <= ln and 
                                abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1]) <= zn):
                                _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                                droplist.append(short_index[cur] - 1)
                        else:
                            _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                            droplist.append(short_index[cur] - 1)
                            if (df.duration[short_index[cur]] <= ln and 
                                abs(df.x_pos[short_index[cur]] - df.x_pos[next_list[-1] + 1]) <= zn):
                                _lump_two_fix(df, short_index[cur], next_list[-1] + 1, 1, addtime)
                                droplist.append(next_list[-1] + 1)
                
                cur += len(next_list)
            else:
                dist_next = 0.0
                dist_prev = 0.0
                
                if (short_index[cur] + 1 <= endindex and 
                    abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] + 1]) <= zn):
                    dist_next = abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] + 1])
                
                if (short_index[cur] - 1 not in droplist and 
                    abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1]) <= zn):
                    dist_prev = abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1])
                
                if dist_next != 0.0 and dist_prev == 0.0:
                    _lump_two_fix(df, short_index[cur], short_index[cur] + 1, 1, addtime)
                    droplist.append(short_index[cur] + 1)
                elif dist_next == 0.0 and dist_prev != 0.0:
                    _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                    droplist.append(short_index[cur] - 1)
                elif dist_next != 0.0 and dist_prev != 0.0:
                    if dist_next < dist_prev:
                        _lump_two_fix(df, short_index[cur], short_index[cur] + 1, 1, addtime)
                        droplist.append(short_index[cur] + 1)
                        if (df.duration[short_index[cur]] <= ln and 
                            abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] - 1]) <= zn):
                            _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                            droplist.append(short_index[cur] - 1)
                    else:
                        _lump_two_fix(df, short_index[cur], short_index[cur] - 1, -1, addtime)
                        droplist.append(short_index[cur] - 1)
                        if (df.duration[short_index[cur]] <= ln and 
                            abs(df.x_pos[short_index[cur]] - df.x_pos[short_index[cur] + 1]) <= zn):
                            _lump_two_fix(df, short_index[cur], short_index[cur] + 1, 1, addtime)
                            droplist.append(short_index[cur] + 1)
        
        if df.loc[short_index[cur], 'duration'] <= ln:
            droplist.append(short_index[cur])
        
        cur += 1
    
    if droplist:
        df = df.drop(droplist)
        df = df.reset_index(drop=True)
    
    return df


def _merge_fix_lines(startline: int, endline: int, df: pd.DataFrame) -> List[Tuple]:
    """
    Merge continuous rightward and leftward fixations.
    
    Args:
        startline: Search starting line
        endline: Search ending line
        df: Fixation DataFrame
        
    Returns:
        List of merged fixation tuples
    """
    mergelines = []
    ind = startline
    
    while ind < endline - 1:
        stl, edl = ind, ind + 1
        if df.loc[edl, 'x_pos'] - df.loc[stl, 'x_pos'] > 0:
            mergelines.append((stl, edl, df.loc[edl, 'x_pos'] - df.loc[stl, 'x_pos'], 0))
        else:
            nextl = edl + 1
            while nextl < endline and df.loc[nextl, 'x_pos'] - df.loc[edl, 'x_pos'] <= 0:
                edl = nextl
                nextl += 1
            mergelines.append((stl, edl, df.loc[edl, 'x_pos'] - df.loc[stl, 'x_pos'], 1))
        ind = edl
    
    return mergelines


def _get_crossline_fix(crossline_info: List[Dict], startline: int, endline: int,
                       df: pd.DataFrame, diff_ratio: float, 
                       frontrange_ratio: float) -> Tuple[List, int, bool]:
    """
    Collect all cross-line fixations.
    
    Args:
        crossline_info: Cross line information from region file
        startline: Search starting line
        endline: Search ending line
        df: Fixation DataFrame
        diff_ratio: Ratio of maximum distance for cross-line detection
        frontrange_ratio: Ratio to check backward crossline
        
    Returns:
        Tuple of (lines, curline, question_flag)
    """
    lines = []
    mergelines = _merge_fix_lines(startline, endline, df)
    curline, ind = 0, 0
    
    while ind < len(crossline_info):
        cur_cross = crossline_info[ind]
        fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
        
        if curline >= len(mergelines):
            break
            
        if (mergelines[curline][3] == 0 and 
            mergelines[curline][2] >= fix_dist_thres and 
            df.loc[mergelines[curline][0], 'x_pos'] <= cur_cross['n_x'] + frontrange_ratio * (cur_cross['p_x'] - cur_cross['n_x'])):
            if ind != 0:
                if ind > 0:
                    ind -= 1
                    cur_cross = crossline_info[ind]
                lines.append((-1, cur_cross['n'], cur_cross['p'], mergelines[curline][1]))
        
        if (mergelines[curline][3] == 1 and 
            mergelines[curline][2] <= -fix_dist_thres):
            find_one = False
            stl1 = mergelines[curline][0]
            
            for nextl in range(mergelines[curline][0] + 1, mergelines[curline][1] + 1):
                if df.loc[nextl, 'x_pos'] - df.loc[nextl - 1, 'x_pos'] <= -fix_dist_thres:
                    stl1 = nextl
                    find_one = True
                    break
            
            if find_one:
                lines.append((1, cur_cross['p'], cur_cross['n'], stl1))
            else:
                stl1 = mergelines[curline][0]
                big_x = 0
                for nextl in range(mergelines[curline][0] + 1, mergelines[curline][1] + 1):
                    if df.loc[nextl - 1, 'x_pos'] - df.loc[nextl, 'x_pos'] > big_x:
                        big_x = df.loc[nextl - 1, 'x_pos'] - df.loc[nextl, 'x_pos']
                        stl1 = nextl
                
                stl2 = mergelines[curline][0]
                big_y = 0
                for nextl in range(mergelines[curline][0] + 1, mergelines[curline][1] + 1):
                    if df.loc[nextl, 'y_pos'] - df.loc[nextl - 1, 'y_pos'] > big_y:
                        big_y = df.loc[nextl, 'y_pos'] - df.loc[nextl - 1, 'y_pos']
                        stl2 = nextl
                
                lines.append((1, cur_cross['p'], cur_cross['n'], max(stl1, stl2)))
            
            if ind < len(crossline_info) - 1:
                ind += 1
            else:
                break
        
        curline += 1
    
    if curline < len(mergelines):
        curline = mergelines[curline][1]
    else:
        curline = mergelines[-1][1]
    
    question = False
    if lines and (lines[0][0] == -1 or lines[-1][0] == -1 or 
                  lines[-1][2] != crossline_info[-1]['n']):
        _log.debug('DIFF: first or last line change is irregular (crlFix).')
        question = True
    
    return lines, curline, question


def _get_fix_line(reg_df: pd.DataFrame, crl_sac: pd.DataFrame, fix_df: pd.DataFrame,
                  classify_method: str, diff_ratio: float, frontrange_ratio: float,
                  y_range: int) -> Tuple[List, bool]:
    """
    Add line information for each fixation.
    
    Args:
        reg_df: Region file DataFrame
        crl_sac: Cross-line saccade DataFrame
        fix_df: Fixation DataFrame (modified in place)
        classify_method: 'DIFF' or 'SAC'
        diff_ratio: Ratio for cross-line detection
        frontrange_ratio: Ratio for backward cross-line check
        y_range: Maximum y difference for line crossing
        
    Returns:
        Tuple of (lines, question_flag)
    """
    crossline_info = _get_crossline_info(reg_df)
    question = False
    
    if len(fix_df.eye.unique()) == 1 and fix_df.eye.iloc[0] in ['L', 'R']:
        # Single eye data
        if classify_method == 'DIFF':
            lines, curline, question = _get_crossline_fix(
                crossline_info, 0, len(fix_df), fix_df, diff_ratio, frontrange_ratio
            )
            endline = len(fix_df)
            
            if curline < len(fix_df):
                cur_cross = crossline_info[-1]
                fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
                nextline = curline + 1
                while (nextline < len(fix_df) and 
                       abs(fix_df.x_pos[curline] - fix_df.x_pos[nextline]) <= fix_dist_thres and 
                       fix_df.y_pos[curline] - fix_df.y_pos[nextline] <= y_range):
                    curline = nextline
                    nextline = curline + 1
                if nextline < len(fix_df):
                    endline = nextline
            
            curlow = 0
            for ind in range(len(lines)):
                curline_data = lines[ind]
                for line in range(curlow, curline_data[3]):
                    fix_df.loc[line, 'line_no'] = curline_data[1]
                curlow = curline_data[3]
            for line in range(curlow, endline):
                fix_df.loc[line, 'line_no'] = lines[-1][2]
                
        elif classify_method == 'SAC':
            lines = []
            curlow = 0
            for ind in range(len(crl_sac)):
                curup = curlow + 1
                while fix_df.end_time[curup] <= crl_sac.start_time[ind]:
                    curup += 1
                start = crl_sac.loc[ind, 'startline']
                end = crl_sac.loc[ind, 'endline']
                direction = 1 if start < end else -1
                lines.append([direction, start, end, curup])
                for line in range(curlow, curup):
                    fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'startline']
                curlow = curup
            for line in range(curlow, len(fix_df)):
                fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'endline']
    else:
        # Double eye data
        num_left = len(fix_df[fix_df.eye == 'L'])
        num_right = len(fix_df[fix_df.eye == 'R'])
        
        if classify_method == 'DIFF':
            # Process left eye
            lines_left, curline_left, ques1 = _get_crossline_fix(
                crossline_info, 0, num_left, fix_df, diff_ratio, frontrange_ratio
            )
            endline_left = num_left
            
            if curline_left < num_left:
                cur_cross = crossline_info[-1]
                fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
                nextline = curline_left + 1
                while (nextline < num_left and 
                       abs(fix_df.x_pos[curline_left] - fix_df.x_pos[nextline]) <= fix_dist_thres and 
                       fix_df.y_pos[curline_left] - fix_df.y_pos[nextline] <= y_range):
                    curline_left = nextline
                    nextline = curline_left + 1
                if nextline < num_left:
                    endline_left = nextline
            
            curlow = 0
            for ind in range(len(lines_left)):
                curline_data = lines_left[ind]
                for line in range(curlow, curline_data[3]):
                    fix_df.loc[line, 'line_no'] = curline_data[1]
                curlow = curline_data[3]
            for line in range(curlow, endline_left):
                fix_df.loc[line, 'line_no'] = lines_left[-1][2]
            
            # Process right eye
            lines_right, curline_right, ques2 = _get_crossline_fix(
                crossline_info, num_left, num_left + num_right, fix_df, diff_ratio, frontrange_ratio
            )
            endline_right = num_left + num_right
            
            if curline_right < num_left + num_right:
                cur_cross = crossline_info[-1]
                fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
                nextline = curline_right + 1
                while (nextline < num_left + num_right and 
                       abs(fix_df.x_pos[curline_right] - fix_df.x_pos[nextline]) <= fix_dist_thres and 
                       fix_df.y_pos[curline_right] - fix_df.y_pos[nextline] <= y_range):
                    curline_right = nextline
                    nextline = curline_right + 1
                if nextline < num_left + num_right:
                    endline_right = nextline
            
            curlow = num_left
            for ind in range(len(lines_right)):
                curline_data = lines_right[ind]
                for line in range(curlow, curline_data[3]):
                    fix_df.loc[line, 'line_no'] = curline_data[1]
                curlow = curline_data[3]
            for line in range(curlow, endline_right):
                fix_df.loc[line, 'line_no'] = lines_right[-1][2]
            
            lines = lines_left + lines_right
            if ques1 or ques2:
                question = True
                
        elif classify_method == 'SAC':
            lines = []
            curlow = 0
            for ind in range(len(crl_sac)):
                if crl_sac.eye[ind] == 'L':
                    curup = curlow + 1
                    while fix_df.eye[curup] == 'L' and fix_df.end_time[curup] <= crl_sac.start_time[ind]:
                        curup += 1
                    start = crl_sac.loc[ind, 'startline']
                    end = crl_sac.loc[ind, 'endline']
                    direction = 1 if start < end else -1
                    lines.append([direction, start, end, curup])
                    for line in range(curlow, curup):
                        fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'startline']
                    curlow = curup
            for line in range(curlow, num_left):
                fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'endline']
            
            curlow = num_left
            for ind in range(len(crl_sac)):
                if crl_sac.eye[ind] == 'R':
                    curup = curlow + 1
                    while fix_df.eye[curup] == 'R' and fix_df.end_time[curup] <= crl_sac.start_time[ind]:
                        curup += 1
                    start = crl_sac.loc[ind, 'startline']
                    end = crl_sac.loc[ind, 'endline']
                    direction = 1 if start < end else -1
                    lines.append([direction, start, end, curup])
                    for line in range(curlow, curup):
                        fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'startline']
                    curlow = curup
            for line in range(curlow, num_left + num_right):
                fix_df.loc[line, 'line_no'] = crl_sac.loc[ind, 'endline']
    
    return lines, question


CRL_FIX_COLUMNS = ['subj', 'trial_id', 'eye', 'startline', 'endline', 'FixlineIndex',
                   'start_time', 'end_time', 'duration', 'x_pos', 'y_pos', 'pup_size',
                   'valid']
"""Columns of the cross-line fixation table (``<subj>_crlFix.csv``)."""

CRL_SAC_COLUMNS = ['subj', 'trial_id', 'eye', 'startline', 'endline', 'SaclineIndex',
                   'start_time', 'end_time', 'duration', 'x1_pos', 'y1_pos', 'x2_pos',
                   'y2_pos', 'ampl', 'pk']
"""Columns of the cross-line saccade table (``<subj>_crlSac.csv``)."""


def _crl_fix_table(fix_df: pd.DataFrame, startlines, endlines, indices) -> pd.DataFrame:
    """Table of the fixations at which the reader's line changes."""
    rows = []
    for st, en, ind in zip(startlines, endlines, indices):
        cur_fix = fix_df.loc[ind]
        rows.append({'subj': cur_fix['subj'], 'trial_id': cur_fix['trial_id'],
                     'eye': cur_fix['eye'], 'startline': st, 'endline': en,
                     'FixlineIndex': ind,
                     **{c: cur_fix.get(c, np.nan) for c in ['start_time', 'end_time',
                                                           'duration', 'x_pos', 'y_pos',
                                                           'pup_size', 'valid']}})
    return pd.DataFrame(rows, columns=CRL_FIX_COLUMNS)


def _crl_fix_from_lines(fix_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the cross-line fixation table from assigned ``line_no`` values.

    A row is created for every fixation whose line differs from the line
    of the preceding assigned fixation (forward return sweeps have
    ``endline > startline``; regressions to earlier lines have
    ``endline < startline``).
    """
    ln = fix_df['line_no'].to_numpy(float)
    idx = [i for i in range(len(ln)) if np.isfinite(ln[i])]
    st, en, at = [], [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if ln[b] != ln[a]:
            st.append(ln[a]); en.append(ln[b]); at.append(b)
    return _crl_fix_table(fix_df, st, en, at)


def _get_crl_fix(reg_df: pd.DataFrame, crl_sac: pd.DataFrame, fix_df: pd.DataFrame,
                 classify_method: str, diff_ratio: float, frontrange_ratio: float,
                 y_range: int) -> Tuple[pd.DataFrame, bool]:
    """
    Get cross-line fixations.
    
    Args:
        reg_df: Region file DataFrame
        crl_sac: Cross-line saccade DataFrame
        fix_df: Fixation DataFrame (modified in place)
        classify_method: 'DIFF' or 'SAC'
        diff_ratio: Ratio for cross-line detection
        frontrange_ratio: Ratio for backward cross-line check
        y_range: Maximum y difference for line crossing
        
    Returns:
        Tuple of (crl_fix_df, question_flag)
    """
    lines, question = _get_fix_line(reg_df, crl_sac, fix_df, classify_method, 
                                     diff_ratio, frontrange_ratio, y_range)
    
    crl_fix = _crl_fix_table(fix_df, [item[1] for item in lines],
                             [item[2] for item in lines], [item[3] for item in lines])
    
    return crl_fix, question


def _rec_timestamp(exp_type: str, trial_id: int, blinklines: List, stamplines: List,
                   sampfreq: int, eyerec: str, script: str, sessdate: str, srcfile: str,
                   trial_type: str, trialstart: int, trialend: int, tdur: int,
                   recstart: int, recend: int, error_free: int = 1) -> pd.DataFrame:
    """
    Get timestamp data from trials.
    
    Args:
        exp_type: Type of experiment ('RAN', 'RP')
        trial_id: Trial ID
        blinklines: Blink data lines
        stamplines: Timestamp data lines
        sampfreq: Sampling frequency
        eyerec: Eye recorded ('R', 'L', or 'LR')
        script: Script file
        sessdate: Session date
        srcfile: Source file
        trial_type: Trial type
        trialstart: Trial start time
        trialend: Trial end time
        tdur: Trial duration
        recstart: Recording start time
        recend: Recording end time
        error_free: Error-free status (default 1)
        
    Returns:
        Timestamp DataFrame
    """
    blink_number = len(blinklines)
    stamp_number = len(stamplines)
    
    stamp_df = pd.DataFrame(np.zeros((stamp_number, 26)), dtype=object,
                            columns=['subj', 'trial_id', 'trial_type', 'sampfreq', 'script',
                                    'sessdate', 'srcfile', 'trialstart', 'trialend', 'tdur',
                                    'recstart', 'recend', 'blinks', 'eye', 'time', 'x_pos1',
                                    'y_pos1', 'pup_size1', 'x_pos2', 'y_pos2', 'pup_size2',
                                    'line_no', 'gaze_region_no', 'label', 'error_free', 'Fix_Sac'])
    
    stamp_df['subj'] = srcfile.split('.')[0]
    stamp_df['trial_id'] = int(trial_id)
    stamp_df['trial_type'] = trial_type
    stamp_df['sampfreq'] = int(sampfreq)
    stamp_df['script'] = script
    stamp_df['sessdate'] = sessdate
    stamp_df['srcfile'] = srcfile
    stamp_df['trialstart'] = trialstart
    stamp_df['trialend'] = trialend
    stamp_df['tdur'] = tdur
    stamp_df['recstart'] = recstart
    stamp_df['recend'] = recend
    stamp_df['blinks'] = int(blink_number)
    stamp_df['eye'] = eyerec
    stamp_df['time'] = [int(line[0]) for line in stamplines]
    
    if eyerec in ['L', 'R']:
        x_pos1 = [line[1] for line in stamplines]
        stamp_df['x_pos1'] = pd.Series(x_pos1).replace('.', np.nan).astype(float)
        y_pos1 = [line[2] for line in stamplines]
        stamp_df['y_pos1'] = pd.Series(y_pos1).replace('.', np.nan).astype(float)
        pup1 = [line[3] for line in stamplines]
        stamp_df['pup_size1'] = pd.Series(pup1).replace('.', np.nan).astype(float)
        stamp_df['x_pos2'] = np.nan
        stamp_df['y_pos2'] = np.nan
        stamp_df['pup_size2'] = np.nan
    elif eyerec == 'LR':
        x_pos1 = [line[1] for line in stamplines]
        stamp_df['x_pos1'] = pd.Series(x_pos1).replace('.', np.nan).astype(float)
        y_pos1 = [line[2] for line in stamplines]
        stamp_df['y_pos1'] = pd.Series(y_pos1).replace('.', np.nan).astype(float)
        pup1 = [line[3] for line in stamplines]
        stamp_df['pup_size1'] = pd.Series(pup1).replace('.', np.nan).astype(float)
        x_pos2 = [line[4] for line in stamplines]
        stamp_df['x_pos2'] = pd.Series(x_pos2).replace('.', np.nan).astype(float)
        y_pos2 = [line[5] for line in stamplines]
        stamp_df['y_pos2'] = pd.Series(y_pos2).replace('.', np.nan).astype(float)
        pup2 = [line[6] for line in stamplines]
        stamp_df['pup_size2'] = pd.Series(pup2).replace('.', np.nan).astype(float)
    
    stamp_df['line_no'] = pd.Series([None] * stamp_number, dtype=object)
    stamp_df['gaze_region_no'] = np.nan
    stamp_df['label'] = pd.Series([None] * stamp_number, dtype=object)
    stamp_df['error_free'] = error_free
    stamp_df['Fix_Sac'] = pd.Series([None] * stamp_number, dtype=object)
    
    return stamp_df


FIX_COLUMNS = ['subj', 'trial_id', 'trial_type', 'sampfreq', 'script', 'sessdate',
               'srcfile', 'trialstart', 'trialend', 'tdur', 'recstart', 'recend',
               'blinks', 'eye', 'start_time', 'end_time', 'duration', 'x_pos',
               'y_pos', 'pup_size', 'valid', 'line_no', 'region_no']
"""Columns of the fixation table written to ``<subj>_Fix.csv``."""

SAC_COLUMNS = ['subj', 'trial_id', 'trial_type', 'sampfreq', 'script', 'sessdate',
               'srcfile', 'trialstart', 'trialend', 'tdur', 'recstart', 'recend',
               'blinks', 'eye', 'start_time', 'end_time', 'duration', 'x1_pos',
               'y1_pos', 'x2_pos', 'y2_pos', 'ampl', 'pk', 'line_no']
"""Columns of the saccade table written to ``<subj>_Sac.csv``."""


def _num(v: str) -> float:
    """Convert an ASC field to float; '.' (missing) becomes NaN."""
    try:
        return float(v)
    except ValueError:
        return np.nan


def _rec_fix(exp_type: str, trial_id: int, blinklines: List, fixlines: List,
             sampfreq: int, eyerec: str, script: str, sessdate: str, srcfile: str,
             trial_type: str, trialstart: int, trialend: int, tdur: int,
             recstart: int, recend: int, rec_last_fix: bool, lump_fix: bool,
             ln: int, zn: int, mn: int) -> pd.DataFrame:
    """
    Build the fixation table of one trial from its EFIX lines.

    Fixations are grouped by eye (left eye first). If ``rec_last_fix`` is
    False, the final fixation of each eye is marked invalid, because it is
    usually cut short by the end of the trial. If ``lump_fix`` is True,
    fixations no longer than ``ln`` ms are merged with a neighbour within
    ``zn`` pixels, and fixations still shorter than ``mn`` ms are marked
    invalid.
    """
    blink_number = len(blinklines)
    addtime = 1 / float(sampfreq) * 1000
    common = {'subj': srcfile.split('.')[0], 'trial_id': int(trial_id),
              'trial_type': trial_type, 'sampfreq': int(sampfreq),
              'script': script, 'sessdate': sessdate, 'srcfile': srcfile,
              'trialstart': trialstart, 'trialend': trialend, 'tdur': tdur,
              'recstart': recstart, 'recend': recend, 'blinks': int(blink_number)}
    eyes = [eyerec] if eyerec in ('L', 'R') else ['L', 'R']
    parts = []
    for eye in eyes:
        rows = [dict(common, eye=line[1], start_time=_num(line[2]), end_time=_num(line[3]),
                     duration=_num(line[4]), x_pos=_num(line[5]), y_pos=_num(line[6]),
                     pup_size=_num(line[7]), valid='yes', line_no=np.nan, region_no=np.nan)
                for line in fixlines if line[1] == eye]
        if not rows:
            if eyerec == 'LR':
                _log.warning(f'Trial {trial_id}: binocular recording without {eye}-eye fixations.')
            continue
        df = pd.DataFrame(rows, columns=FIX_COLUMNS)
        n = len(df)
        df = _apply_validity(df, addtime, rec_last_fix, lump_fix, ln, zn, mn)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=FIX_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _apply_validity(df: pd.DataFrame, addtime: float, rec_last_fix: bool = False,
                    lump_fix: bool = True, ln: int = 50, zn: int = 50,
                    mn: int = 50) -> pd.DataFrame:
    """Validity rules for the fixations of one trial and one eye.

    Shared by the EyeLink route (:func:`read_srr_asc`) and the generic route
    (:func:`apply_fixation_validity`). If ``rec_last_fix`` is False, the last
    fixation is marked invalid. If ``lump_fix`` is True, fixations no longer
    than ``ln`` ms are merged with a neighbour within ``zn`` pixels, and
    fixations still shorter than ``mn`` ms are marked invalid.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return df
    if not rec_last_fix:
        df.loc[n - 1, 'valid'] = 'no'
    if lump_fix:
        short_index = [i for i in range(n)
                       if df.loc[i, 'duration'] <= ln and df.loc[i, 'valid'] == 'yes']
        endindex = n - 2 if not rec_last_fix else n - 1
        df = _lump_fix(df, endindex, short_index, addtime, ln, zn)
        df.loc[df['duration'] < mn, 'valid'] = 'no'
    return df


def apply_fixation_validity(fix_df: pd.DataFrame, rec_last_fix: bool = False,
                            lump_fix: bool = True, ln: int = 50, zn: int = 50,
                            mn: int = 50, sampfreq: Optional[float] = None) -> pd.DataFrame:
    """Apply the EyeLink-route validity rules to a fixation table from any source.

    The rules are those of :func:`read_srr_asc`: per trial and eye, the last
    fixation is marked invalid (unless ``rec_last_fix``), fixations no longer
    than ``ln`` ms are merged with a neighbour within ``zn`` pixels, and
    fixations still shorter than ``mn`` ms are marked invalid. Applying
    them makes measures computed from other sources comparable with those
    computed from EyeLink ASC files.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations in the pyemread schema, in temporal order within trials.
    rec_last_fix, lump_fix : bool
        See above (defaults False and True, as for EyeLink data).
    ln, zn, mn : int
        Duration (ms) and distance (px) limits for merging, and the minimum
        valid duration (ms); defaults 50, 50 and 50.
    sampfreq : float, optional
        Sampling rate (Hz); taken from a ``sampfreq`` column if absent.
        Merged fixations are extended by one sample interval.

    Returns
    -------
    pandas.DataFrame
        The fixation table with ``valid`` updated (``'yes'``/``'no'``).
    """
    parts = []
    df = fix_df.copy()
    if 'valid' not in df:
        df['valid'] = 'yes'
    added = [c for c in ('pup_size', 'line_no', 'region_no') if c not in df]
    for c in added:
        df[c] = np.nan
    for (_, _), g in df.groupby(['trial_id', 'eye'], sort=False):
        sf = sampfreq
        if sf is None and 'sampfreq' in g and pd.notna(g['sampfreq']).any():
            sf = float(pd.to_numeric(g['sampfreq'], errors='coerce').dropna().iloc[0])
        addtime = 1000.0 / sf if sf else 0.0
        parts.append(_apply_validity(g, addtime, rec_last_fix, lump_fix, ln, zn, mn))
    if not parts:
        return df.drop(columns=added)
    return pd.concat(parts, ignore_index=True).drop(columns=added)


def _merge_sac_lines(startline: int, endline: int, df: pd.DataFrame) -> List[Tuple]:
    """
    Merge continuous rightward and leftward saccades.
    
    Args:
        startline: Search starting line
        endline: Search ending line
        df: Saccade DataFrame
        
    Returns:
        List of merged saccade tuples
    """
    mergelines = []
    ind = startline
    
    while ind < endline:
        if df.loc[ind, 'x2_pos'] - df.loc[ind, 'x1_pos'] > 0:
            mergelines.append((ind, ind, df.loc[ind, 'x2_pos'] - df.loc[ind, 'x1_pos'], 0))
            nextl = ind + 1
        else:
            nextl = ind + 1
            edl = nextl - 1
            while nextl < endline and df.loc[nextl, 'x2_pos'] - df.loc[nextl, 'x1_pos'] <= 0:
                edl = nextl
                nextl += 1
            mergelines.append((ind, edl, df.loc[edl, 'x2_pos'] - df.loc[ind, 'x1_pos'], 1))
        ind = nextl
    
    return mergelines


def _get_crossline_sac(crossline_info: List[Dict], startline: int, endline: int,
                       df: pd.DataFrame, diff_ratio: float, 
                       frontrange_ratio: float) -> Tuple[List, int, bool]:
    """
    Collect all cross-line saccades.
    
    Args:
        crossline_info: Cross line information from region file
        startline: Search starting line
        endline: Search ending line
        df: Saccade DataFrame
        diff_ratio: Ratio for cross-line detection
        frontrange_ratio: Ratio for backward cross-line check
        
    Returns:
        Tuple of (lines, curline, question_flag)
    """
    lines = []
    mergelines = _merge_sac_lines(startline, endline, df)
    curline, ind = 0, 0
    
    while ind < len(crossline_info):
        if curline >= len(mergelines):
            break
            
        cur_cross = crossline_info[ind]
        fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
        
        if (mergelines[curline][3] == 0 and 
            mergelines[curline][2] >= fix_dist_thres and 
            df.loc[mergelines[curline][0], 'x1_pos'] <= cur_cross['n_x'] + frontrange_ratio * (cur_cross['p_x'] - cur_cross['n_x'])):
            if ind != 0:
                if ind > 0:
                    ind -= 1
                    cur_cross = crossline_info[ind]
                lines.append((-1, cur_cross['n'], cur_cross['p'], mergelines[curline][1]))
        
        if (mergelines[curline][3] == 1 and 
            mergelines[curline][2] <= -fix_dist_thres):
            find_one = False
            stl1 = mergelines[curline][0]
            
            for nextl in range(mergelines[curline][0], mergelines[curline][1] + 1):
                if df.loc[nextl, 'x2_pos'] - df.loc[nextl, 'x1_pos'] <= -fix_dist_thres:
                    stl1 = nextl
                    find_one = True
                    break
            
            if find_one:
                lines.append((1, cur_cross['p'], cur_cross['n'], stl1))
            else:
                stl1 = mergelines[curline][0]
                big_x = 0
                for nextl in range(mergelines[curline][0], mergelines[curline][1] + 1):
                    if df.loc[nextl, 'x1_pos'] - df.loc[nextl, 'x2_pos'] > big_x:
                        big_x = df.loc[nextl, 'x1_pos'] - df.loc[nextl, 'x2_pos']
                        stl1 = nextl
                
                stl2 = mergelines[curline][0]
                big_y = 0
                for nextl in range(mergelines[curline][0], mergelines[curline][1] + 1):
                    if df.loc[nextl, 'y2_pos'] - df.loc[nextl, 'y1_pos'] > big_y:
                        big_y = df.loc[nextl, 'y2_pos'] - df.loc[nextl, 'y1_pos']
                        stl2 = nextl
                
                lines.append((1, cur_cross['p'], cur_cross['n'], max(stl1, stl2)))
            
            if ind < len(crossline_info) - 1:
                ind += 1
            else:
                break
        
        curline += 1
    
    if curline < len(mergelines):
        curline = mergelines[curline][1]
    else:
        curline = mergelines[-1][1]
    
    question = False
    if lines and (lines[0][0] == -1 or lines[-1][0] == -1 or 
                  lines[-1][2] != crossline_info[-1]['n']):
        _log.debug('DIFF: first or last line change is irregular (crlSac).')
        question = True
    
    return lines, curline, question


def _get_sac_line(reg_df: pd.DataFrame, sac_df: pd.DataFrame, diff_ratio: float,
                  frontrange_ratio: float, y_range: int) -> Tuple[List, bool]:
    """
    Add line information for each saccade.
    
    Args:
        reg_df: Region file DataFrame
        sac_df: Saccade DataFrame (modified in place)
        diff_ratio: Ratio for cross-line detection
        frontrange_ratio: Ratio for backward cross-line check
        y_range: Maximum y difference for line crossing
        
    Returns:
        Tuple of (lines, question_flag)
    """
    crossline_info = _get_crossline_info(reg_df)
    question = False
    
    if len(sac_df.eye.unique()) == 1 and sac_df.eye.iloc[0] in ['L', 'R']:
        # Single eye data
        lines, curline, question = _get_crossline_sac(
            crossline_info, 0, len(sac_df), sac_df, diff_ratio, frontrange_ratio
        )
        endline = len(sac_df)
        
        if curline < len(sac_df):
            cur_cross = crossline_info[-1]
            fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
            curline += 1
            while (curline < len(sac_df) and 
                   abs(sac_df.x2_pos[curline] - sac_df.x1_pos[curline]) <= fix_dist_thres and 
                   sac_df.y1_pos[curline] - sac_df.y2_pos[curline] <= y_range):
                curline += 1
            if curline < len(sac_df):
                endline = curline
        
        curlow = 0
        for ind in range(len(lines)):
            curline_data = lines[ind]
            for line in range(curlow, curline_data[3]):
                sac_df.loc[line, 'line_no'] = curline_data[1]
            sac_df.loc[curline_data[3], 'line_no'] = f"{curline_data[1]}_{curline_data[2]}"
            curlow = curline_data[3] + 1
        for line in range(curlow, endline):
            sac_df.loc[line, 'line_no'] = lines[-1][2]
    else:
        # Double eye data
        num_left = len(sac_df[sac_df.eye == 'L'])
        num_right = len(sac_df[sac_df.eye == 'R'])
        
        # Process left eye
        lines_left, curline_left, ques1 = _get_crossline_sac(
            crossline_info, 0, num_left, sac_df, diff_ratio, frontrange_ratio
        )
        endline_left = num_left
        
        if curline_left < num_left:
            cur_cross = crossline_info[-1]
            fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
            curline_left += 1
            while (curline_left < num_left and 
                   abs(sac_df.x2_pos[curline_left] - sac_df.x1_pos[curline_left]) <= fix_dist_thres and 
                   sac_df.y1_pos[curline_left] - sac_df.y2_pos[curline_left] <= y_range):
                curline_left += 1
            if curline_left < num_left:
                endline_left = curline_left
        
        curlow = 0
        for ind in range(len(lines_left)):
            curline_data = lines_left[ind]
            for line in range(curlow, curline_data[3]):
                sac_df.loc[line, 'line_no'] = curline_data[1]
            sac_df.loc[curline_data[3], 'line_no'] = f"{curline_data[1]}_{curline_data[2]}"
            curlow = curline_data[3] + 1
        for line in range(curlow, endline_left):
            sac_df.loc[line, 'line_no'] = lines_left[-1][2]
        
        # Process right eye
        lines_right, curline_right, ques2 = _get_crossline_sac(
            crossline_info, num_left, num_left + num_right, sac_df, diff_ratio, frontrange_ratio
        )
        endline_right = num_left + num_right
        
        if curline_right < num_left + num_right:
            cur_cross = crossline_info[-1]
            fix_dist_thres = diff_ratio * (cur_cross['p_x'] - cur_cross['n_x'])
            curline_right += 1
            while (curline_right < num_left + num_right and 
                   abs(sac_df.x2_pos[curline_right] - sac_df.x1_pos[curline_right]) <= fix_dist_thres and 
                   sac_df.y1_pos[curline_right] - sac_df.y2_pos[curline_right] <= y_range):
                curline_right += 1
            if curline_right < num_left + num_right:
                endline_right = curline_right
        
        curlow = num_left
        for ind in range(len(lines_right)):
            curline_data = lines_right[ind]
            for line in range(curlow, curline_data[3]):
                sac_df.loc[line, 'line_no'] = curline_data[1]
            sac_df.loc[curline_data[3], 'line_no'] = f"{curline_data[1]}_{curline_data[2]}"
            curlow = curline_data[3] + 1
        for line in range(curlow, endline_right):
            sac_df.loc[line, 'line_no'] = lines_right[-1][2]
        
        lines = lines_left + lines_right
        if ques1 or ques2:
            question = True
    
    return lines, question


def _get_crl_sac(reg_df: pd.DataFrame, sac_df: pd.DataFrame, diff_ratio: float,
                 frontrange_ratio: float, y_range: int) -> Tuple[pd.DataFrame, bool]:
    """
    Get cross-line saccades.
    
    Args:
        reg_df: Region file DataFrame
        sac_df: Saccade DataFrame (modified in place)
        diff_ratio: Ratio for cross-line detection
        frontrange_ratio: Ratio for backward cross-line check
        y_range: Maximum y difference for line crossing
        
    Returns:
        Tuple of (crl_sac_df, question_flag)
    """
    lines, question = _get_sac_line(reg_df, sac_df, diff_ratio, frontrange_ratio, y_range)
    
    rows = []
    for item in lines:
        cur_sac = sac_df.loc[item[3]]
        rows.append({'subj': cur_sac['subj'], 'trial_id': cur_sac['trial_id'],
                     'eye': cur_sac['eye'], 'startline': item[1], 'endline': item[2],
                     'SaclineIndex': item[3],
                     **{c: cur_sac[c] for c in ['start_time', 'end_time', 'duration',
                                                'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos',
                                                'ampl', 'pk']}})
    crl_sac = pd.DataFrame(rows, columns=CRL_SAC_COLUMNS)
    
    return crl_sac, question


def _rec_sac(exp_type: str, trial_id: int, blinklines: List, saclines: List,
             sampfreq: int, eyerec: str, script: str, sessdate: str, srcfile: str,
             trial_type: str, trialstart: int, trialend: int, tdur: int,
             recstart: int, recend: int) -> pd.DataFrame:
    """
    Build the saccade table of one trial from its ESACC lines.

    Saccades with a missing field (e.g. those interrupted by a blink) are
    dropped. Saccades are grouped by eye (left eye first).
    """
    blink_number = len(blinklines)
    saclines = [line for line in saclines
                if len(line) >= 11 and all(line[ind] != '.' for ind in range(2, 11))]
    common = {'subj': srcfile.split('.')[0], 'trial_id': int(trial_id),
              'trial_type': trial_type, 'sampfreq': int(sampfreq),
              'script': script, 'sessdate': sessdate, 'srcfile': srcfile,
              'trialstart': trialstart, 'trialend': trialend, 'tdur': tdur,
              'recstart': recstart, 'recend': recend, 'blinks': int(blink_number)}
    eyes = [eyerec] if eyerec in ('L', 'R') else ['L', 'R']
    rows = []
    for eye in eyes:
        sel = [line for line in saclines if line[1] == eye]
        if not sel and eyerec == 'LR':
            _log.warning(f'Trial {trial_id}: binocular recording without {eye}-eye saccades.')
        rows += [dict(common, eye=line[1], start_time=float(line[2]), end_time=float(line[3]),
                      duration=float(line[4]), x1_pos=float(line[5]), y1_pos=float(line[6]),
                      x2_pos=float(line[7]), y2_pos=float(line[8]), ampl=float(line[9]),
                      pk=float(line[10]), line_no=None)
                 for line in sel]
    df = pd.DataFrame(rows, columns=SAC_COLUMNS)
    df['line_no'] = df['line_no'].astype(object)
    return df


def _mod_reg_df(reg_df: pd.DataFrame, add_char_sp: int) -> None:
    """
    Modify region DataFrame boundaries for overshoot fixation capture.
    
    Args:
        reg_df: Region DataFrame (modified in place)
        add_char_sp: Number of single character spaces to add
    """
    reg_df['mod_x1'] = reg_df['x1_pos'].astype(float)
    reg_df['mod_x2'] = reg_df['x2_pos'].astype(float)
    add_dist = add_char_sp * (reg_df.loc[0, 'x2_pos'] - reg_df.loc[0, 'x1_pos']) / float(reg_df.loc[0, 'length'])
    
    for cur_em in range(len(reg_df)):
        if cur_em == 0:
            reg_df.loc[cur_em, 'mod_x1'] -= add_dist
        elif cur_em == len(reg_df) - 1:
            reg_df.loc[cur_em, 'mod_x2'] += add_dist
        else:
            if reg_df.loc[cur_em - 1, 'line_no'] == reg_df.loc[cur_em, 'line_no'] - 1:
                reg_df.loc[cur_em, 'mod_x1'] -= add_dist
            elif reg_df.loc[cur_em + 1, 'line_no'] == reg_df.loc[cur_em, 'line_no'] + 1:
                reg_df.loc[cur_em, 'mod_x2'] += add_dist


def _crt_asc_dic(sit: int, direct: str, subj_id: str) -> Tuple[bool, Dict[str, str]]:
    """
    Create dictionary of ASCII files.
    
    Args:
        sit: Situation (0 for specific subject, 1 for all subjects)
        direct: Root directory
        subj_id: Subject ID (for sit=0)
        
    Returns:
        Tuple of (file_exists, file_dictionary)
    """
    asc_file_exist = True
    asc_file_dic = {}
    
    if sit == 0:
        filename = os.path.join(direct, subj_id, f'{subj_id}.asc')
        if os.path.isfile(filename):
            asc_file_dic[subj_id] = filename
        else:
            raise FileNotFoundError(
                f"ASC file not found: {filename}. pyemread expects each "
                f"participant's file at <direct>/<subj_id>/<subj_id>.asc.")
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith('.asc'):
                    subj = name.split('.')[0]
                    asc_file_dic[subj] = os.path.join(root, name)
        if not asc_file_dic:
            raise FileNotFoundError(f"No .asc files found under {direct}.")
    
    return asc_file_exist, asc_file_dic


def _crt_csv_dic(sit: int, direct: str, subj_id: str, csv_filetype: str) -> Tuple[bool, Dict[str, str]]:
    """
    Create dictionary for different types of CSV files.
    
    Args:
        sit: Situation (0 for specific subject, 1 for all subjects)
        direct: Root directory
        subj_id: Subject ID (for sit=0)
        csv_filetype: File type suffix ('_Stamp', '_Sac', '_crlSac', '_Fix', '_crlFix')
        
    Returns:
        Tuple of (file_exists, file_dictionary)
    """
    csv_file_exist = True
    csv_file_dic = {}
    target_file_end = f'{csv_filetype}.csv'
    
    if sit == 0:
        filename = os.path.join(direct, subj_id, f'{subj_id}{target_file_end}')
        if os.path.isfile(filename):
            csv_file_dic[subj_id] = filename
        else:
            raise FileNotFoundError(
                f"{filename} not found. Run ext.read_write_srr_asc() or "
                f"ext.process_subject() first to create it.")
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(target_file_end):
                    subj = name.split(target_file_end)[0]
                    csv_file_dic[subj] = os.path.join(root, name)
        if not csv_file_dic:
            raise FileNotFoundError(f"No *{target_file_end} files found under {direct}.")
    
    return csv_file_exist, csv_file_dic


def _crt_region_dic(direct: str, regfile_name_list: List[str]) -> Tuple[bool, Dict[str, str]]:
    """
    Create region file dictionary.
    
    Args:
        direct: Root directory
        regfile_name_list: List of region file names
        
    Returns:
        Tuple of (file_exists, file_dictionary)
    """
    regfile_exist = True
    regfile_dic = {}
    target_file_end = '.region.csv'
    
    if not regfile_name_list:
        for file in os.listdir(direct):
            if fnmatch.fnmatch(file, f'*{target_file_end}'):
                regfile_dic[str(file)] = os.path.join(direct, str(file))
        if not regfile_dic:
            raise FileNotFoundError(f"No *.region.csv file found in {direct}.")
    else:
        for regfile in regfile_name_list:
            regfile_name = os.path.join(direct, regfile)
            if os.path.isfile(regfile_name):
                regfile_dic[regfile] = regfile_name
            else:
                raise FileNotFoundError(f"Region file not found: {regfile_name}")
    
    return regfile_exist, regfile_dic


def _crt_fix_rep_dic(sit: int, direct: str, subj_id: str) -> Tuple[bool, Dict[str, str]]:
    """
    Create dictionary for fixation report files.
    
    Args:
        sit: Situation (0 for specific subject, 1 for all subjects)
        direct: Root directory
        subj_id: Subject ID (for sit=0)
        
    Returns:
        Tuple of (file_exists, file_dictionary)
    """
    fix_rep_exist = True
    fix_rep_dic = {}
    fix_rep_name_end = '-FixReportLines.txt'
    
    if sit == 0:
        filename = os.path.join(direct, subj_id, f'{subj_id}{fix_rep_name_end}')
        if os.path.isfile(filename):
            fix_rep_dic[subj_id] = filename
        else:
            raise FileNotFoundError(f"Fixation report not found: {filename}")
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(fix_rep_name_end):
                    subj = name.split(fix_rep_name_end)[0]
                    fix_rep_dic[subj] = os.path.join(root, name)
        if not fix_rep_dic:
            raise FileNotFoundError(f"No *{fix_rep_name_end} files found under {direct}.")
    
    return fix_rep_exist, fix_rep_dic


def _cal_timestamp(align_method: str, trial_type: str, trialstart: int,
                   reg_df: pd.DataFrame, stamp_df_temp: pd.DataFrame,
                   fix_rep_df: Optional[pd.DataFrame], sac_df: Optional[pd.DataFrame],
                   fix_df: Optional[pd.DataFrame]) -> None:
    """
    Assign line_no based on FixRep or Fix_Sac.
    
    Args:
        align_method: 'FixRep' or 'Fix_Sac'
        trial_type: Trial type
        trialstart: Trial start time
        reg_df: Region DataFrame
        stamp_df_temp: Timestamp DataFrame (modified in place)
        fix_rep_df: Fixation report DataFrame (for FixRep method)
        sac_df: Saccade DataFrame (for Fix_Sac method)
        fix_df: Fixation DataFrame (for Fix_Sac method)
    """
    if align_method == 'FixRep' and fix_rep_df is not None:
        fix_rep_cur = fix_rep_df[fix_rep_df.trial == trial_type].reset_index()
        line_idx, line_time = _get_line_info(fix_rep_cur)
        for curind in range(len(line_idx)):
            mask = ((stamp_df_temp.time >= line_time[curind][0] + trialstart) & 
                    (stamp_df_temp.time <= line_time[curind][1] + trialstart))
            stamp_df_temp.loc[mask, 'line_no'] = line_idx[curind]
            stamp_df_temp.loc[mask, 'Fix_Sac'] = 'Fix'
    
    elif align_method == 'Fix_Sac' and sac_df is not None and fix_df is not None:
        sac_df_cur = sac_df[sac_df.trial_type == trial_type].reset_index()
        fix_df_cur = fix_df[fix_df.trial_type == trial_type].reset_index()
        
        for curind in range(len(sac_df_cur)):
            starttime = sac_df_cur.start_time[curind]
            endtime = sac_df_cur.end_time[curind]
            mask = (stamp_df_temp.time >= starttime) & (stamp_df_temp.time <= endtime)
            stamp_df_temp.loc[mask, 'line_no'] = sac_df_cur.line_no[curind]
            stamp_df_temp.loc[mask, 'Fix_Sac'] = 'Sac'
        
        for curind in range(len(fix_df_cur)):
            starttime = fix_df_cur.start_time[curind]
            endtime = fix_df_cur.end_time[curind]
            mask = (stamp_df_temp.time >= starttime) & (stamp_df_temp.time <= endtime)
            stamp_df_temp.loc[mask, 'line_no'] = fix_df_cur.line_no[curind]
            stamp_df_temp.loc[mask, 'Fix_Sac'] = 'Fix'
    
    # Assign region_no
    for cur_stamp in range(len(stamp_df_temp)):
        if (not pd.isna(stamp_df_temp.loc[cur_stamp, 'x_pos1']) and 
            stamp_df_temp.loc[cur_stamp, 'Fix_Sac'] == 'Fix' and 
            not pd.isna(stamp_df_temp.loc[cur_stamp, 'line_no'])):
            indlist = reg_df[
                (reg_df['line_no'] == stamp_df_temp.loc[cur_stamp, 'line_no']) & 
                (reg_df['mod_x1'] <= stamp_df_temp.loc[cur_stamp, 'x_pos1']) & 
                (reg_df['mod_x2'] >= stamp_df_temp.loc[cur_stamp, 'x_pos1'])
            ].index.tolist()
            if len(indlist) == 1:
                stamp_df_temp.loc[cur_stamp, 'gaze_region_no'] = int(reg_df.WordID[indlist[0]])
                stamp_df_temp.loc[cur_stamp, 'label'] = reg_df.Word[indlist[0]]
            else:
                stamp_df_temp.loc[cur_stamp, 'gaze_region_no'] = np.nan
        else:
            stamp_df_temp.loc[cur_stamp, 'gaze_region_no'] = np.nan


# =============================================================================
# User functions
# =============================================================================

def read_srr_asc(direct: str, subj_id: str, exp_type: str,
                 rec_last_fix: bool = False, lump_fix: bool = True,
                 ln: int = 50, zn: int = 50, mn: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read SR Research ASCII file and extract saccades and fixations.
    
    Args:
        direct: Directory containing output files
        subj_id: Subject ID
        exp_type: Type of experiment ('RAN', 'RP')
        rec_last_fix: Include last fixation of a trial (default False)
        lump_fix: Lump short fixations (default True)
        ln: Maximum duration for lumping (default 50)
        zn: Maximum distance for lumping (default 50)
        mn: Minimum legal fixation duration (default 50)
        
    Returns:
        Tuple of (SacDF, FixDF)
    """
    asc_file_exist, asc_file_dic = _crt_asc_dic(0, direct, subj_id)
    
    if not asc_file_exist:
        return pd.DataFrame(), pd.DataFrame()
    
    with open(asc_file_dic[subj_id], 'r', encoding='utf-8', errors='replace') as f:
        _log.info(f"Read ASC: {f.name}")
        lines = f.readlines()
    
    script, sessdate, srcfile = _get_header(lines)
    T_idx, T_lines = _get_trial_reg(lines)
    
    sac_df = pd.DataFrame()
    fix_df = pd.DataFrame()
    
    for ind in range(len(T_lines)):
        triallines = lines[T_lines[ind, 0] + 1:T_lines[ind, 1]]
        # Convert 0-indexed trial_id from ASC file to 1-indexed
        trial_id = int(T_idx[ind, 0].split(' ')[-1]) + 1
        blinklines, fixlines, saclines, sampfreq, eyerec = _get_blink_fix_sac_sampfreq_eyerec(triallines, 0)
        trial_type, trialstart, trialend, tdur, recstart, recend = _get_tdur(triallines)
        
        _log.info(f"Read Sac: Trial {trial_id}, Type {trial_type}")
        sac_df_temp = _rec_sac(exp_type, trial_id, blinklines, saclines, sampfreq, eyerec,
                               script, sessdate, srcfile, trial_type, trialstart, trialend,
                               tdur, recstart, recend)
        sac_df = pd.concat([sac_df, sac_df_temp], ignore_index=True)
        
        _log.info(f"Read Fix: Trial {trial_id}, Type {trial_type}")
        fix_df_temp = _rec_fix(exp_type, trial_id, blinklines, fixlines, sampfreq, eyerec,
                               script, sessdate, srcfile, trial_type, trialstart, trialend,
                               tdur, recstart, recend, rec_last_fix, lump_fix, ln, zn, mn)
        fix_df = pd.concat([fix_df, fix_df_temp], ignore_index=True)
    
    return sac_df, fix_df


def write_sac_report(direct: str, subj_id: str, sac_df: pd.DataFrame) -> None:
    """Write saccade DataFrame to CSV file."""
    sac_df.to_csv(os.path.join(direct, subj_id, f'{subj_id}_Sac.csv'), index=False)


def write_fix_report(direct: str, subj_id: str, fix_df: pd.DataFrame) -> None:
    """Write fixation DataFrame to CSV file."""
    fix_df.to_csv(os.path.join(direct, subj_id, f'{subj_id}_Fix.csv'), index=False)


def write_timestamp_report(direct: str, subj_id: str, stamp_df: pd.DataFrame) -> None:
    """Write timestamp DataFrame to CSV file."""
    stamp_df.to_csv(os.path.join(direct, subj_id, f'{subj_id}_Stamp.csv'), index=False)


def read_write_srr_asc(direct: str, subj_id: str, exp_type: str,
                       rec_last_fix: bool = False, lump_fix: bool = True,
                       ln: int = 50, zn: int = 50, mn: int = 50) -> None:
    """
    Read and process a subject's saccades and fixations, then write to CSV files.
    
    Args:
        direct: Directory containing the ASC file
        subj_id: Subject ID
        exp_type: Type of experiment ('RAN', 'RP')
        rec_last_fix: Include last fixation of a trial (default False)
        lump_fix: Lump short fixations (default True)
        ln: Maximum duration for lumping (default 50)
        zn: Maximum distance for lumping (default 50)
        mn: Minimum legal fixation duration (default 50)
    """
    sac_df, fix_df = read_srr_asc(direct, subj_id, exp_type, rec_last_fix, lump_fix, ln, zn, mn)
    write_sac_report(direct, subj_id, sac_df)
    write_fix_report(direct, subj_id, fix_df)


def read_write_srr_asc_b(direct: str, exp_type: str,
                         rec_last_fix: bool = False, lump_fix: bool = True,
                         ln: int = 50, zn: int = 50, mn: int = 50) -> None:
    """
    Batch process all subjects' saccades and fixations.
    
    Args:
        direct: Directory containing all ASC files
        exp_type: Type of experiment ('RAN', 'RP')
        rec_last_fix: Include last fixation of a trial (default False)
        lump_fix: Lump short fixations (default True)
        ln: Maximum duration for lumping (default 50)
        zn: Maximum distance for lumping (default 50)
        mn: Minimum legal fixation duration (default 50)
    """
    asc_file_exist, asc_file_dic = _crt_asc_dic(1, direct, '')
    if asc_file_exist:
        for subj_id in asc_file_dic:
            sac_df, fix_df = read_srr_asc(direct, subj_id, exp_type, rec_last_fix, lump_fix, ln, zn, mn)
            write_sac_report(direct, subj_id, sac_df)
            write_fix_report(direct, subj_id, fix_df)


def read_timestamp(direct: str, subj_id: str, exp_type: str) -> pd.DataFrame:
    """
    Read SR Research ASCII file and extract time-stamped eye movements.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        exp_type: Type of experiment ('RAN', 'RP')
        
    Returns:
        Timestamp DataFrame
    """
    asc_file_exist, asc_file_dic = _crt_asc_dic(0, direct, subj_id)
    
    if not asc_file_exist:
        return pd.DataFrame()
    
    with open(asc_file_dic[subj_id], 'r', encoding='utf-8', errors='replace') as f:
        _log.info(f"Read ASC: {f.name}")
        lines = f.readlines()
    
    script, sessdate, srcfile = _get_header(lines)
    T_idx, T_lines = _get_trial_reg(lines)
    
    stamp_df = pd.DataFrame()
    
    for ind in range(len(T_lines)):
        triallines = lines[T_lines[ind, 0] + 1:T_lines[ind, 1]]
        # Convert 0-indexed trial_id from ASC file to 1-indexed
        trial_id = int(T_idx[ind, 0].split(' ')[-1]) + 1
        blinklines, stamplines, sampfreq, eyerec = _get_blink_fix_sac_sampfreq_eyerec(triallines, 1)
        trial_type, trialstart, trialend, tdur, recstart, recend = _get_tdur(triallines)
        
        _log.info(f"Read Stamped Eye Movements: Trial {trial_id}; Type {trial_type}")
        stamp_df_temp = _rec_timestamp(exp_type, trial_id, blinklines, stamplines, sampfreq, eyerec,
                                       script, sessdate, srcfile, trial_type, trialstart, trialend,
                                       tdur, recstart, recend)
        stamp_df = pd.concat([stamp_df, stamp_df_temp], ignore_index=True)
    
    return stamp_df


def read_write_timestamp(direct: str, subj_id: str, exp_type: str) -> None:
    """
    Read and write a subject's time-stamped data.
    
    Args:
        direct: Directory containing the ASC file
        subj_id: Subject ID
        exp_type: Type of experiment ('RAN', 'RP')
    """
    stamp_df = read_timestamp(direct, subj_id, exp_type)
    write_timestamp_report(direct, subj_id, stamp_df)


def read_write_timestamp_b(direct: str, exp_type: str) -> None:
    """
    Batch process all subjects' time-stamped data.
    
    Args:
        direct: Directory containing all ASC files
        exp_type: Type of experiment ('RAN', 'RP')
    """
    asc_file_exist, asc_file_dic = _crt_asc_dic(1, direct, '')
    if asc_file_exist:
        for subj_id in asc_file_dic:
            stamp_df = read_timestamp(direct, subj_id, exp_type)
            write_timestamp_report(direct, subj_id, stamp_df)


LINE_METHODS = ('SLICE', 'DIFF', 'SAC', 'NEAREST', 'PROVIDED')
"""Accepted values of ``classify_method``."""

DEFAULT_LINE_METHOD = 'SLICE'
"""Line-assignment method used when none is specified."""


def load_regions(direct: str, regfile_name_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Read region files into a dictionary keyed by trial type.

    Parameters
    ----------
    direct : str
        Directory containing the region files.
    regfile_name_list : list of str
        File names such as ``"story01.region.csv"``. An empty list reads
        every ``*.region.csv`` file in ``direct``.

    Returns
    -------
    dict
        ``{trial_type: region DataFrame}``, where ``trial_type`` is the
        file name without ``.region.csv`` (e.g. ``"story01"``).
    """
    _, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    return {os.path.basename(k)[:-len('.region.csv')]: pd.read_csv(v, sep=',', encoding='utf-8')
            for k, v in regfile_dic.items()}


def _sac_lines_from_fix(sac_df: pd.DataFrame, fix_df: pd.DataFrame,
                        tol: float = 10.0) -> pd.DataFrame:
    """Label saccades with the lines of the fixations they connect.

    A saccade between two fixations on the same line receives that line
    number; a saccade between lines ``a`` and ``b`` receives the string
    ``"a_b"`` (the convention used by the saccade table).
    """
    out = sac_df.copy()
    out['line_no'] = pd.Series([None] * len(out), index=out.index, dtype=object)
    if len(out) == 0 or len(fix_df) == 0:
        return out
    f_end = fix_df['end_time'].to_numpy(float)
    f_start = fix_df['start_time'].to_numpy(float)
    f_line = fix_df['line_no'].to_numpy(float)
    for i in out.index:
        st, en = float(out.at[i, 'start_time']), float(out.at[i, 'end_time'])
        prev = np.where(f_end <= st + tol)[0]
        nxt = np.where(f_start >= en - tol)[0]
        if len(prev) == 0 or len(nxt) == 0:
            continue
        a, b = f_line[prev[-1]], f_line[nxt[0]]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        out.at[i, 'line_no'] = float(a) if a == b else f"{int(a)}_{int(b)}"
    return out


def _crl_sac_from_lines(sac_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-line saccade table from ``"a_b"`` saccade labels."""
    rows = []
    for i in sac_df.index:
        lab = sac_df.at[i, 'line_no']
        if isinstance(lab, str) and '_' in lab:
            a, b = lab.split('_')
            r = sac_df.loc[i]
            rows.append({'subj': r['subj'], 'trial_id': r['trial_id'], 'eye': r['eye'],
                         'startline': float(a), 'endline': float(b), 'SaclineIndex': i,
                         **{c: r.get(c, np.nan) for c in ['start_time', 'end_time', 'duration',
                                                          'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos',
                                                          'ampl', 'pk']}})
    return pd.DataFrame(rows, columns=CRL_SAC_COLUMNS)


def classify_fixations(sac_df: pd.DataFrame, fix_df: pd.DataFrame,
                       regions: Dict[str, pd.DataFrame],
                       classify_method=DEFAULT_LINE_METHOD,
                       diff_ratio: float = 0.6, frontrange_ratio: float = 0.2,
                       y_range: int = 60, add_char_sp: int = 1, check: bool = True
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List]:
    """Assign fixations (and saccades) to text lines and words, in memory.

    Every trial and every recorded eye is processed separately.

    Parameters
    ----------
    sac_df, fix_df : pandas.DataFrame
        Saccade and fixation tables in the pyemread schema (see
        :data:`SAC_COLUMNS` and :data:`FIX_COLUMNS`). ``sac_df`` may be
        empty when only fixations are available.
    regions : dict
        ``{trial_type: region DataFrame}``, e.g. from :func:`load_regions`.
    classify_method : str or dict
        One of ``'SLICE'``, ``'DIFF'``, ``'SAC'``, ``'NEAREST'``, ``'PROVIDED'``,
        or a dictionary that chooses the method per trial, keyed by
        ``(trial_id, eye)`` or ``trial_id``, with an optional ``'default'``
        entry, e.g. ``{(1, 'R'): 'DIFF', 'default': 'SLICE'}``.
        ``'SLICE'``: the slice algorithm of Glandorf and Schroeder (2021)
        (:func:`pyemread.lines.assign_lines_slice`). ``'DIFF'``: the original
        pyemread heuristic based on horizontal displacements between
        fixations. ``'SAC'``: the original heuristic applied to saccades.
        ``'NEAREST'``: nearest line centre. ``'PROVIDED'``: keep the
        ``line_no`` values already in ``fix_df`` (e.g. after manual
        correction in another program). See :mod:`pyemread.lines`.
    diff_ratio, frontrange_ratio, y_range : float
        Parameters of the ``'DIFF'`` and ``'SAC'`` methods.
    add_char_sp : int
        Character widths added at line edges when assigning words.
    check : bool
        Compare the ``'SLICE'`` and ``'DIFF'`` assignments of every trial and
        flag trials on which they disagree on more than 10% of fixations
        (default True).

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame
        Saccades with ``line_no``; cross-line saccades; fixations with
        ``line_no`` and ``region_no``; cross-line fixations.
    flagged : list of tuple
        ``(trial_id, eye)`` pairs whose line assignment should be checked
        visually (see the Notes).

    Raises
    ------
    KeyError
        If a trial's stimulus (``trial_type``) is not in ``regions``.
    ValueError
        If ``classify_method`` is not one of :data:`LINE_METHODS`, if
        ``'SAC'`` is chosen without saccades, or if ``'PROVIDED'`` is chosen
        without line numbers.

    Notes
    -----
    With ``check=True``, a trial (eye) is flagged when the ``'SLICE'`` and
    ``'DIFF'`` assignments disagree on more than 10% of its valid fixations
    (trials whose method is chosen explicitly in ``classify_method`` are not
    checked again) (see
    :func:`pyemread.lines.line_assignment_report`). In the validation
    reported with the package, every trial on which either method assigned
    fewer than 90% of fixations correctly was flagged. Flagged trials are
    logged and should be inspected visually.
    """
    if isinstance(classify_method, dict):
        method_map = {k: str(v).upper() for k, v in classify_method.items()}
        default = method_map.get('default', DEFAULT_LINE_METHOD)
    else:
        method_map, default = {}, str(classify_method).upper()
    for m in set(method_map.values()) | {default}:
        if m not in LINE_METHODS:
            raise ValueError(f"classify_method must be one of {LINE_METHODS}, not {m!r}")
    new_sac, new_fix, crl_sacs, crl_fixs, flagged = [], [], [], [], []
    trial_ids = list(dict.fromkeys(list(fix_df['trial_id'].unique())))
    for trial_id in trial_ids:
        f_trial = fix_df[fix_df['trial_id'] == trial_id]
        s_trial = sac_df[sac_df['trial_id'] == trial_id] if len(sac_df) else sac_df
        trial_type = str(f_trial['trial_type'].iloc[0])
        if trial_type not in regions:
            raise KeyError(f"No region file for trial type {trial_type!r} (trial {trial_id}). "
                           f"Available: {sorted(regions)}")
        reg_df = regions[trial_type].reset_index(drop=True).copy()
        _mod_reg_df(reg_df, add_char_sp)
        for eye in list(dict.fromkeys(f_trial['eye'])):
            fe = f_trial[f_trial['eye'] == eye].reset_index(drop=True).copy()
            se = (s_trial[s_trial['eye'] == eye].reset_index(drop=True).copy()
                  if len(s_trial) else pd.DataFrame(columns=SAC_COLUMNS))
            se['line_no'] = se['line_no'].astype(object) if 'line_no' in se else None
            method = method_map.get((trial_id, eye), method_map.get(trial_id, default))
            _log.info(f"Classify: trial {trial_id} ({trial_type}), eye {eye}, method {method}")
            if method in ('DIFF', 'SAC'):
                fe['line_no'] = np.nan
                if len(se) > 0:
                    crl_s, _ = _get_crl_sac(reg_df, se, diff_ratio, frontrange_ratio, y_range)
                else:
                    crl_s = pd.DataFrame(columns=CRL_SAC_COLUMNS)
                if method == 'SAC' and len(crl_s) == 0:
                    raise ValueError("classify_method='SAC' requires saccades.")
                try:
                    crl_f, _ = _get_crl_fix(reg_df, crl_s, fe, method, diff_ratio,
                                            frontrange_ratio, y_range)
                except IndexError:
                    # no line change was detected (e.g. a single line was read)
                    fe = _lines.assign_lines_diff(fe, reg_df, diff_ratio=diff_ratio,
                                                  frontrange_ratio=frontrange_ratio,
                                                  y_range=y_range)
                    crl_f = _crl_fix_from_lines(fe)
            else:
                if method == 'PROVIDED':
                    if 'line_no' not in fe or fe['line_no'].notna().sum() == 0:
                        raise ValueError("classify_method='PROVIDED' requires line_no values in fix_df.")
                else:
                    fe = _lines.assign_lines(fe, reg_df, method=method.lower())
                crl_f = _crl_fix_from_lines(fe)
                se = _sac_lines_from_fix(se, fe)
                crl_s = _crl_sac_from_lines(se)
            question, agree = False, np.nan
            chosen = (trial_id, eye) in method_map or trial_id in method_map
            fv = fe[fe['valid'].astype(str).str.lower() == 'yes'] if 'valid' in fe else fe
            if check and not chosen and len(fv) > 2:
                rep = _lines.line_assignment_report(fv, reg_df)
                question, agree = rep['flag'], rep['agree_slice_diff']
            fe = _lines.assign_regions(fe, reg_df, add_char_sp)
            if question:
                flagged.append((trial_id, eye))
                _log.warning(f"Participant {fe['subj'].iloc[0]}, trial {trial_id}, eye {eye}: the SLICE "
                             f"and DIFF line assignments agree on only {agree:.0%} of fixations; "
                             f"inspect this trial (gen.draw_line_assignment).")
            new_fix.append(fe); new_sac.append(se); crl_fixs.append(crl_f); crl_sacs.append(crl_s)
    cat = lambda xs, cols: (pd.concat([x for x in xs if len(x)], ignore_index=True)
                            if any(len(x) for x in xs) else pd.DataFrame(columns=cols))
    return (cat(new_sac, SAC_COLUMNS), cat(crl_sacs, CRL_SAC_COLUMNS),
            cat(new_fix, FIX_COLUMNS), cat(crl_fixs, CRL_FIX_COLUMNS), flagged)


def _log_flags(direct: str, subj_id: str, flagged: List) -> None:
    path = os.path.join(direct, 'log.txt')
    prefix = f'Subj: {subj_id} '
    kept = []
    if os.path.exists(path):
        with open(path) as f:
            kept = [ln for ln in f if not ln.startswith(prefix)]
    if flagged or kept:
        with open(path, 'w') as logfile:
            logfile.writelines(kept)
            for trial_id, eye in flagged:
                logfile.write(f'Subj: {subj_id} Trial {trial_id} Eye {eye}: SLICE and DIFF line '
                              f'assignments disagree; inspect this trial\n')


def cal_crl_sac_fix(direct: str, subj_id: str, regfile_name_list: List[str],
                    exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                    rec_status: bool = True, diff_ratio: float = 0.6,
                    frontrange_ratio: float = 0.2, y_range: int = 60,
                    add_char_sp: int = 1, write: bool = False
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify a participant's saccades and fixations into text lines and words.

    Reads ``<direct>/<subj_id>/<subj_id>_Sac.csv`` and ``_Fix.csv`` (written
    by :func:`read_write_srr_asc`), assigns every fixation to a line and a
    word, and returns the updated tables.

    Parameters
    ----------
    direct : str
        Data directory; region files are read from it and participant files
        from its sub-folder ``subj_id``.
    subj_id : str
        Participant identifier (e.g. ``"1950138"``).
    regfile_name_list : list of str
        Region files, e.g. ``["story01.region.csv"]``.
    exp_type : str
        Experiment type label, kept for compatibility (default ``"RP"``).
    classify_method : {'SLICE', 'DIFF', 'SAC', 'NEAREST', 'PROVIDED'}
        Line-assignment method; see :func:`classify_fixations`.
    rec_status : bool
        Append trials whose assignment should be checked to
        ``<direct>/log.txt`` (default True).
    diff_ratio, frontrange_ratio, y_range : float
        Parameters of the ``'DIFF'``/``'SAC'`` methods.
    add_char_sp : int
        Character widths added at line edges when assigning words.
    write : bool
        If True, overwrite ``_Sac.csv`` and ``_Fix.csv`` with the classified
        tables and write ``_crlSac.csv`` and ``_crlFix.csv``. If False
        (default), nothing is written; pass the returned DataFrames to
        :func:`pyemread.cal.cal_write_em` (``fix_df=``, ``sac_df=``) or call
        :func:`write_sac_crl_sac` and :func:`write_fix_crl_fix`.

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame
    """
    _, sac_file_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
    _, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
    regions = load_regions(direct, regfile_name_list)
    sac_df = pd.read_csv(sac_file_dic[subj_id], sep=',')
    fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',')
    new_sac_df, crl_sac, new_fix_df, crl_fix, flagged = classify_fixations(
        sac_df, fix_df, regions, classify_method, diff_ratio, frontrange_ratio,
        y_range, add_char_sp)
    if rec_status:
        _log_flags(direct, subj_id, flagged)
    if write:
        write_sac_crl_sac(direct, subj_id, new_sac_df, crl_sac)
        write_fix_crl_fix(direct, subj_id, new_fix_df, crl_fix)
    return new_sac_df, crl_sac, new_fix_df, crl_fix


def write_sac_crl_sac(direct: str, subj_id: str, sac_df: pd.DataFrame, 
                      crl_sac: pd.DataFrame) -> None:
    """Write saccade and cross-line saccade DataFrames to CSV files."""
    sac_df.to_csv(os.path.join(direct, subj_id, f'{subj_id}_Sac.csv'), index=False)
    crl_sac.to_csv(os.path.join(direct, subj_id, f'{subj_id}_crlSac.csv'), index=False)


def write_fix_crl_fix(direct: str, subj_id: str, fix_df: pd.DataFrame, 
                      crl_fix: pd.DataFrame) -> None:
    """Write fixation and cross-line fixation DataFrames to CSV files."""
    fix_df.to_csv(os.path.join(direct, subj_id, f'{subj_id}_Fix.csv'), index=False)
    crl_fix.to_csv(os.path.join(direct, subj_id, f'{subj_id}_crlFix.csv'), index=False)


def cal_write_sac_fix_crl_sac_fix(direct: str, subj_id: str, regfile_name_list: List[str],
                                  exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                                  rec_status: bool = True, diff_ratio: float = 0.6,
                                  frontrange_ratio: float = 0.2, y_range: int = 60,
                                  add_char_sp: int = 1) -> None:
    """
    Process a subject's saccades and fixations and write to CSV files.
    
    Args:
        direct: Directory containing CSV files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'SLICE', 'DIFF', 'SAC', 'NEAREST' or 'PROVIDED' (see classify_fixations)
        rec_status: Record questionable data (default True)
        diff_ratio: Ratio for cross-line detection (default 0.6)
        frontrange_ratio: Ratio for backward cross-line check (default 0.2)
        y_range: Maximum y difference for line crossing (default 60)
        add_char_sp: Number of characters to add for overshoot (default 1)
    """
    sac_df, crl_sac, fix_df, crl_fix = cal_crl_sac_fix(
        direct, subj_id, regfile_name_list, exp_type, classify_method,
        rec_status, diff_ratio, frontrange_ratio, y_range, add_char_sp
    )
    write_sac_crl_sac(direct, subj_id, sac_df, crl_sac)
    write_fix_crl_fix(direct, subj_id, fix_df, crl_fix)


def cal_write_sac_fix_crl_sac_fix_b(direct: str, regfile_name_list: List[str],
                                    exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                                    rec_status: bool = True, diff_ratio: float = 0.6,
                                    frontrange_ratio: float = 0.2, y_range: int = 60,
                                    add_char_sp: int = 1) -> None:
    """
    Batch process all subjects' saccades and fixations.
    
    Args:
        direct: Directory containing all CSV files
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'SLICE', 'DIFF', 'SAC', 'NEAREST' or 'PROVIDED' (see classify_fixations)
        rec_status: Record questionable data (default True)
        diff_ratio: Ratio for cross-line detection (default 0.6)
        frontrange_ratio: Ratio for backward cross-line check (default 0.2)
        y_range: Maximum y difference for line crossing (default 60)
        add_char_sp: Number of characters to add for overshoot (default 1)
    """
    sac_file_exist, sac_file_dic = _crt_csv_dic(1, direct, '', '_Sac')
    fix_file_exist, fix_file_dic = _crt_csv_dic(1, direct, '', '_Fix')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if sac_file_exist and fix_file_exist and regfile_exist:
        for subj_id in sac_file_dic:
            cal_write_sac_fix_crl_sac_fix(
                direct, subj_id, regfile_name_list, exp_type, classify_method,
                rec_status, diff_ratio, frontrange_ratio, y_range, add_char_sp
            )


# =============================================================================
# Combined read-calculate-write functions
# =============================================================================

def read_cal_srr_asc(direct: str, subj_id: str, regfile_name_list: List[str],
                     exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                     rec_status: bool = True, diff_ratio: float = 0.6,
                     frontrange_ratio: float = 0.2, y_range: int = 60,
                     add_char_sp: int = 1, rec_last_fix: bool = False,
                     lump_fix: bool = True, ln: int = 50, zn: int = 50,
                     mn: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read an ASC file and classify its fixations and saccades, in memory.

    Equivalent to :func:`read_srr_asc` followed by
    :func:`classify_fixations`. See those functions for the parameters.

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame
    """
    regions = load_regions(direct, regfile_name_list)
    sac_df, fix_df = read_srr_asc(direct, subj_id, exp_type, rec_last_fix, lump_fix, ln, zn, mn)
    sac_df, crl_sac, fix_df, crl_fix, flagged = classify_fixations(
        sac_df, fix_df, regions, classify_method, diff_ratio, frontrange_ratio,
        y_range, add_char_sp)
    if rec_status:
        _log_flags(direct, subj_id, flagged)
    return sac_df, crl_sac, fix_df, crl_fix


def process_subject(direct: str, subj_id: str, regfile_name_list: List[str],
                    classify_method: str = DEFAULT_LINE_METHOD, exp_type: str = 'RP',
                    **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read, classify and write one participant's EyeLink data in a single call.

    Reads ``<direct>/<subj_id>/<subj_id>.asc``, extracts fixations and
    saccades, assigns fixations to lines and words, and writes
    ``<subj_id>_Sac.csv``, ``_Fix.csv``, ``_crlSac.csv`` and ``_crlFix.csv``
    to ``<direct>/<subj_id>/``. The written files are ready for
    :func:`pyemread.cal.cal_write_em`.

    Parameters
    ----------
    direct : str
        Data directory containing the region files and one sub-folder per
        participant.
    subj_id : str
        Participant identifier; the ASC file must be
        ``<direct>/<subj_id>/<subj_id>.asc``.
    regfile_name_list : list of str
        Region files, one per stimulus (trial type).
    classify_method : {'SLICE', 'DIFF', 'SAC', 'NEAREST', 'PROVIDED'}
        Line-assignment method (see :func:`classify_fixations`).
    exp_type : str
        Experiment type label (default ``"RP"``).
    **kwargs
        Further arguments of :func:`read_cal_srr_asc` (e.g. ``lump_fix``,
        ``diff_ratio``).

    Returns
    -------
    sac_df, crl_sac, fix_df, crl_fix : pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If the ASC file or a region file does not exist (the message names
        the expected path).
    KeyError
        If a trial's stimulus has no region file.
    ValueError
        If ``classify_method`` is not one of :data:`LINE_METHODS`.
    """
    sac_df, crl_sac, fix_df, crl_fix = read_cal_srr_asc(
        direct, subj_id, regfile_name_list, exp_type=exp_type,
        classify_method=classify_method, **kwargs)
    write_sac_crl_sac(direct, subj_id, sac_df, crl_sac)
    write_fix_crl_fix(direct, subj_id, fix_df, crl_fix)
    return sac_df, crl_sac, fix_df, crl_fix


def read_cal_write_srr_asc(direct: str, subj_id: str, regfile_name_list: List[str],
                           exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                           rec_status: bool = True, diff_ratio: float = 0.6,
                           frontrange_ratio: float = 0.2, y_range: int = 60,
                           add_char_sp: int = 1, rec_last_fix: bool = False,
                           lump_fix: bool = True, ln: int = 50, zn: int = 50,
                           mn: int = 50) -> None:
    """
    Read ASC file, extract and classify data, and write to CSV files.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'SLICE', 'DIFF', 'SAC', 'NEAREST' or 'PROVIDED' (see classify_fixations)
        rec_status: Record questionable data (default True)
        diff_ratio: Ratio for cross-line detection (default 0.6)
        frontrange_ratio: Ratio for backward cross-line check (default 0.2)
        y_range: Maximum y difference for line crossing (default 60)
        add_char_sp: Number of characters for overshoot (default 1)
        rec_last_fix: Include last fixation (default False)
        lump_fix: Lump short fixations (default True)
        ln: Maximum duration for lumping (default 50)
        zn: Maximum distance for lumping (default 50)
        mn: Minimum legal fixation duration (default 50)
    """
    sac_df, crl_sac, fix_df, crl_fix = read_cal_srr_asc(
        direct, subj_id, regfile_name_list, exp_type, classify_method,
        rec_status, diff_ratio, frontrange_ratio, y_range, add_char_sp,
        rec_last_fix, lump_fix, ln, zn, mn
    )
    if sac_df is not None:
        write_sac_crl_sac(direct, subj_id, sac_df, crl_sac)
        write_fix_crl_fix(direct, subj_id, fix_df, crl_fix)


def read_cal_write_srr_asc_b(direct: str, regfile_name_list: List[str],
                             exp_type: str = 'RP', classify_method: str = DEFAULT_LINE_METHOD,
                             rec_last_fix: bool = False, lump_fix: bool = True,
                             ln: int = 50, zn: int = 50, mn: int = 50,
                             rec_status: bool = True, diff_ratio: float = 0.6,
                             frontrange_ratio: float = 0.2, y_range: int = 60,
                             add_char_sp: int = 1) -> None:
    """
    Batch process all subjects: read ASC files, extract, classify, and write data.
    
    Args:
        direct: Directory containing all ASC files
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'SLICE', 'DIFF', 'SAC', 'NEAREST' or 'PROVIDED' (see classify_fixations)
        rec_last_fix: Include last fixation (default False)
        lump_fix: Lump short fixations (default True)
        ln: Maximum duration for lumping (default 50)
        zn: Maximum distance for lumping (default 50)
        mn: Minimum legal fixation duration (default 50)
        rec_status: Record questionable data (default True)
        diff_ratio: Ratio for cross-line detection (default 0.6)
        frontrange_ratio: Ratio for backward cross-line check (default 0.2)
        y_range: Maximum y difference for line crossing (default 60)
        add_char_sp: Number of characters for overshoot (default 1)
    """
    ascfile_exist, ascfile_dic = _crt_asc_dic(1, direct, '')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if ascfile_exist and regfile_exist:
        for subj_id in ascfile_dic:
            read_cal_write_srr_asc(
                direct, subj_id, regfile_name_list, exp_type, classify_method,
                rec_status, diff_ratio, frontrange_ratio, y_range, add_char_sp,
                rec_last_fix, lump_fix, ln, zn, mn
            )


def read_cal_timestamp(direct: str, subj_id: str, regfile_name_list: List[str],
                       exp_type: str, align_method: str, add_char_sp: int = 1) -> pd.DataFrame:
    """
    Read ASC file and extract classified time-stamped data.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        align_method: 'FixRep' or 'Fix_Sac'
        add_char_sp: Number of characters for overshoot (default 1)
        
    Returns:
        DataFrame with classified time-stamped data
    """
    ascfile_exist, ascfile_dic = _crt_asc_dic(0, direct, subj_id)
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if align_method == 'FixRep':
        fix_rep_exist, fix_rep_dic = _crt_fix_rep_dic(0, direct, subj_id)
    elif align_method == 'Fix_Sac':
        sac_file_exist, sac_file_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
        fix_file_exist, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
    
    if not ascfile_exist or not regfile_exist:
        return None
    
    if align_method == 'FixRep' and not fix_rep_exist:
        return None
    if align_method == 'Fix_Sac' and not (sac_file_exist and fix_file_exist):
        return None
    
    # Read ASCII file
    with open(ascfile_dic[subj_id], 'r') as f:
        _log.info(f"Read ASC: {f.name}")
        lines = f.readlines()
    
    script, sessdate, srcfile = _get_header(lines)
    t_idx, t_lines = _get_trial_reg(lines)
    
    stamp_df = pd.DataFrame(columns=['subj', 'trial_id', 'trial_type', 'sampfreq', 'script',
                                     'sessdate', 'srcfile', 'trialstart', 'trialend', 'tdur',
                                     'recstart', 'recend', 'blinks', 'eye', 'time', 'x_pos1',
                                     'y_pos1', 'pup_size1', 'x_pos2', 'y_pos2', 'pup_size2',
                                     'line_no', 'gaze_region_no', 'label', 'error_free', 'Fix_Sac'])
    
    if align_method == 'FixRep':
        fix_rep_df = pd.read_csv(fix_rep_dic[subj_id], sep='\t')
        sac_df = None
        fix_df = None
    elif align_method == 'Fix_Sac':
        fix_rep_df = None
        sac_df = pd.read_csv(sac_file_dic[subj_id], sep=',')
        fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',')
    
    for ind in range(len(t_lines)):
        triallines = lines[t_lines[ind, 0] + 1:t_lines[ind, 1]]
        # Convert 0-indexed trial_id from ASC file to 1-indexed
        trial_id = int(t_idx[ind, 0].split(' ')[-1]) + 1
        blinklines, stamplines, sampfreq, eyerec = _get_blink_fix_sac_sampfreq_eyerec(triallines, 1)
        trial_type, trialstart, trialend, tdur, recstart, recend = _get_tdur(triallines)
        error_free = 1
        
        reg_df = _get_reg_df(regfile_dic, trial_type)
        _mod_reg_df(reg_df, add_char_sp)
        
        _log.info(f"Read Time Stamped Data: Trial {trial_id}, Type {trial_type}")
        stamp_df_temp = _rec_timestamp(exp_type, trial_id, blinklines, stamplines, sampfreq,
                                       eyerec, script, sessdate, srcfile, trial_type,
                                       trialstart, trialend, tdur, recstart, recend, error_free)
        _cal_timestamp(align_method, trial_type, trialstart, reg_df, stamp_df_temp,
                      fix_rep_df, sac_df, fix_df)
        
        stamp_df = pd.concat([stamp_df, stamp_df_temp], ignore_index=True)
    
    return stamp_df


def read_cal_write_timestamp(direct: str, subj_id: str, regfile_name_list: List[str],
                             exp_type: str, align_method: str, add_char_sp: int = 1) -> None:
    """
    Read ASC file, extract and classify time-stamped data, and write to CSV.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        align_method: 'FixRep' or 'Fix_Sac'
        add_char_sp: Number of characters for overshoot (default 1)
    """
    stamp_df = read_cal_timestamp(direct, subj_id, regfile_name_list, exp_type,
                                  align_method, add_char_sp)
    if stamp_df is not None:
        write_timestamp_report(direct, subj_id, stamp_df)


def read_cal_write_timestamp_b(direct: str, regfile_name_list: List[str],
                               exp_type: str, align_method: str,
                               add_char_sp: int = 1) -> None:
    """
    Batch process: read ASC files, extract/classify time-stamped data, write to CSV.
    
    Args:
        direct: Directory containing all ASC files
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        align_method: 'FixRep' or 'Fix_Sac'
        add_char_sp: Number of characters for overshoot (default 1)
    """
    ascfile_exist, ascfile_dic = _crt_asc_dic(1, direct, '')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if ascfile_exist and regfile_exist:
        for subj_id in ascfile_dic:
            read_cal_write_timestamp(direct, subj_id, regfile_name_list, exp_type,
                                    align_method, add_char_sp)


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Timestamp processing functions
def cal_timestamp(direct: str, subj_id: str, regfile_name_list: List[str],
                  exp_type: str, align_method: str = 'Fix_Sac',
                  add_char_sp: int = 1) -> Optional[pd.DataFrame]:
    """
    Read CSV timestamp data file and classify into text lines and word regions.
    
    Args:
        direct: Directory for storing timestamp CSV and output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiments ('RAN', 'RP')
        align_method: 'FixRep' (based on FixRepDF) or 'Fix_Sac' (based on SacDF, FixDF)
        add_char_sp: Number of single character space added for overshoot fixations
    
    Returns:
        DataFrame with classified timestamp data, or None if files not found
    """
    # Check if stamp and region files exist
    stamp_exist, stamp_dic = _crt_csv_dic(0, direct, subj_id, '_Stamp')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if not stamp_exist or not regfile_exist:
        return None
    
    # Check alignment method files
    if align_method == 'FixRep':
        fix_rep_exist, fix_rep_dic = _crt_fix_rep_dic(0, direct, subj_id)
        if not fix_rep_exist:
            return None
        fix_rep_df = pd.read_csv(fix_rep_dic[subj_id], sep='\t')
        sac_df = None
        fix_df = None
    elif align_method == 'Fix_Sac':
        sac_exist, sac_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
        fix_exist, fix_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
        if not sac_exist or not fix_exist:
            return None
        fix_rep_df = None
        sac_df = pd.read_csv(sac_dic[subj_id], sep=',')
        fix_df = pd.read_csv(fix_dic[subj_id], sep=',')
    else:
        _log.info(f"Invalid align_method: {align_method}")
        return None
    
    stamp_df = pd.read_csv(stamp_dic[subj_id], sep=',')
    new_stamp_df = pd.DataFrame()
    
    _log.info(f"Processing subject: {subj_id}")
    
    for trial_id in stamp_df['trial_id'].unique():
        trial_type = stamp_df.loc[stamp_df.trial_id == trial_id, 'trial_type'].iloc[0]
        trialstart = stamp_df.loc[stamp_df.trial_id == trial_id, 'trialstart'].iloc[0]
        
        reg_df = _get_reg_df(regfile_dic, trial_type)
        _mod_reg_df(reg_df, add_char_sp)
        
        _log.info(f"  Processing trial {trial_id}, type: {trial_type}")
        stamp_df_temp = stamp_df[stamp_df.trial_id == trial_id].reset_index(drop=True)
        _cal_timestamp(align_method, trial_type, trialstart, reg_df, 
                       stamp_df_temp, fix_rep_df, sac_df, fix_df)
        
        new_stamp_df = pd.concat([new_stamp_df, stamp_df_temp], ignore_index=True)
    
    return new_stamp_df


def cal_write_timestamp(direct: str, subj_id: str, regfile_name_list: List[str],
                        exp_type: str, align_method: str = 'Fix_Sac',
                        add_char_sp: int = 1) -> None:
    """
    Process subject's timestamp data and write to CSV file.
    
    Args:
        direct: Directory containing data files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiments ('RAN', 'RP')
        align_method: 'FixRep' or 'Fix_Sac'
        add_char_sp: Number of single character space added for overshoot fixations
    """
    stamp_df = cal_timestamp(direct, subj_id, regfile_name_list, exp_type,
                             align_method, add_char_sp)
    if stamp_df is not None:
        write_timestamp_report(direct, subj_id, stamp_df)


def cal_write_timestamp_b(direct: str, regfile_name_list: List[str],
                          exp_type: str, align_method: str = 'Fix_Sac',
                          add_char_sp: int = 1) -> None:
    """
    Process all subjects' timestamp data and write to CSV files.
    
    Args:
        direct: Directory containing data files
        regfile_name_list: List of region file names
        exp_type: Type of experiments ('RAN', 'RP')
        align_method: 'FixRep' or 'Fix_Sac'
        add_char_sp: Number of single character space added for overshoot fixations
    """
    stamp_exist, stamp_dic = _crt_csv_dic(1, direct, '', '_Stamp')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if stamp_exist and regfile_exist:
        for subj_id in stamp_dic:
            cal_write_timestamp(direct, subj_id, regfile_name_list, exp_type,
                               align_method, add_char_sp)


read_SRRasc = read_srr_asc
write_Sac_Report = write_sac_report
write_Fix_Report = write_fix_report
write_TimeStamp_Report = write_timestamp_report
read_write_SRRasc = read_write_srr_asc
read_write_SRRasc_b = read_write_srr_asc_b
read_TimeStamp = read_timestamp
read_write_TimeStamp = read_write_timestamp
read_write_TimeStamp_b = read_write_timestamp_b
cal_crlSacFix = cal_crl_sac_fix
write_Sac_crlSac = write_sac_crl_sac
write_Fix_crlFix = write_fix_crl_fix
cal_write_SacFix_crlSacFix = cal_write_sac_fix_crl_sac_fix
cal_write_SacFix_crlSacFix_b = cal_write_sac_fix_crl_sac_fix_b
read_cal_SRRasc = read_cal_srr_asc
read_cal_write_SRRasc = read_cal_write_srr_asc
read_cal_write_SRRasc_b = read_cal_write_srr_asc_b
read_cal_TimeStamp = read_cal_timestamp
read_cal_write_TimeStamp = read_cal_write_timestamp
read_cal_write_TimeStamp_b = read_cal_write_timestamp_b
cal_TimeStamp = cal_timestamp
cal_write_TimeStamp = cal_write_timestamp
cal_write_TimeStamp_b = cal_write_timestamp_b
