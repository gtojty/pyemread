# -*- coding: utf-8 -*-
"""
Pyemread - ext.py

This module provides functions for extracting saccades and fixations
detected by eye trackers (SR Research EyeLink devices) during reading
and classifying them into different text lines.

Updates in version 2.1.1:
- trial_id is now 1-indexed (1, 2, 3...) to match story names (story01, story02, story03...)
- This aligns trial_id with human-readable naming conventions

Updates in version 2.0.0:
- Python 3 compatibility
- Improved error handling and type hints
- Better file encoding handling
- Removed deprecated Python 2 constructs

Usage:
    from pyemread import ext
    # Or: import pyemread as pr; pr.ext.function_name()
"""

__author__ = "Tao Gong and David Braze"
__copyright__ = "Copyright 2017-2024, The Pyemread Project"
__credits__ = ["Tao Gong", "David Braze", "Jonathan Gordils", "Hosung Nam"]
__license__ = "MIT"
__version__ = "2.1.1"
__maintainer__ = ["Tao Gong", "David Braze"]
__email__ = ["gtojty@gmail.com", "davebraze@gmail.com"]
__status__ = "Production"

import os
import sys
import fnmatch
import re
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path

import pandas as pd
import numpy as np


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
        if re.search('!V TRIAL_VAR picture_name', line):
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
        print('Warning! crlFix start/end need check!')
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
    
    crl_fix = pd.DataFrame(np.zeros((len(lines), 13)),
                           columns=['subj', 'trial_id', 'eye', 'startline', 'endline',
                                   'FixlineIndex', 'start_time', 'end_time', 'duration',
                                   'x_pos', 'y_pos', 'pup_size', 'valid'])
    crl_fix['subj'] = fix_df['subj'].iloc[0]
    crl_fix['trial_id'] = fix_df['trial_id'].iloc[0]
    
    for cur, item in enumerate(lines):
        cur_fix = fix_df.loc[item[3]]
        crl_fix.loc[cur, 'eye'] = cur_fix['eye']
        crl_fix.loc[cur, 'startline'] = item[1]
        crl_fix.loc[cur, 'endline'] = item[2]
        crl_fix.loc[cur, 'FixlineIndex'] = item[3]
        crl_fix.loc[cur, 'start_time'] = cur_fix['start_time']
        crl_fix.loc[cur, 'end_time'] = cur_fix['end_time']
        crl_fix.loc[cur, 'duration'] = cur_fix['duration']
        crl_fix.loc[cur, 'x_pos'] = cur_fix['x_pos']
        crl_fix.loc[cur, 'y_pos'] = cur_fix['y_pos']
        crl_fix.loc[cur, 'pup_size'] = cur_fix['pup_size']
        crl_fix.loc[cur, 'valid'] = cur_fix['valid']
    
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
    
    stamp_df = pd.DataFrame(np.zeros((stamp_number, 26)),
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
    
    stamp_df['line_no'] = np.nan
    stamp_df['gaze_region_no'] = np.nan
    stamp_df['label'] = np.nan
    stamp_df['error_free'] = error_free
    stamp_df['Fix_Sac'] = np.nan
    
    return stamp_df


def _rec_fix(exp_type: str, trial_id: int, blinklines: List, fixlines: List,
             sampfreq: int, eyerec: str, script: str, sessdate: str, srcfile: str,
             trial_type: str, trialstart: int, trialend: int, tdur: int,
             recstart: int, recend: int, rec_last_fix: bool, lump_fix: bool,
             ln: int, zn: int, mn: int) -> pd.DataFrame:
    """
    Get fixation data from trials.
    
    Args:
        exp_type: Type of experiment
        trial_id: Trial ID
        blinklines: Blink data lines
        fixlines: Fixation data lines
        sampfreq: Sampling frequency
        eyerec: Eye recorded
        script: Script file
        sessdate: Session date
        srcfile: Source file
        trial_type: Trial type
        trialstart: Trial start time
        trialend: Trial end time
        tdur: Trial duration
        recstart: Recording start time
        recend: Recording end time
        rec_last_fix: Whether to include last fixation
        lump_fix: Whether to lump short fixations
        ln: Maximum duration for lumping
        zn: Maximum distance for lumping
        mn: Minimum legal fixation duration
        
    Returns:
        Fixation DataFrame
    """
    blink_number = len(blinklines)
    fix_number = len(fixlines)
    addtime = 1 / float(sampfreq) * 1000
    
    columns = ['subj', 'trial_id', 'trial_type', 'sampfreq', 'script', 'sessdate',
               'srcfile', 'trialstart', 'trialend', 'tdur', 'recstart', 'recend',
               'blinks', 'eye', 'start_time', 'end_time', 'duration', 'x_pos',
               'y_pos', 'pup_size', 'valid', 'line_no', 'region_no']
    
    if eyerec in ['L', 'R']:
        fix_df = pd.DataFrame(np.zeros((fix_number, 23)), columns=columns)
        fix_df['subj'] = srcfile.split('.')[0]
        fix_df['trial_id'] = int(trial_id)
        fix_df['trial_type'] = trial_type
        fix_df['sampfreq'] = int(sampfreq)
        fix_df['script'] = script
        fix_df['sessdate'] = sessdate
        fix_df['srcfile'] = srcfile
        fix_df['trialstart'] = trialstart
        fix_df['trialend'] = trialend
        fix_df['tdur'] = tdur
        fix_df['recstart'] = recstart
        fix_df['recend'] = recend
        fix_df['blinks'] = int(blink_number)
        fix_df['eye'] = [line[1] for line in fixlines]
        fix_df['start_time'] = [float(line[2]) for line in fixlines]
        fix_df['end_time'] = [float(line[3]) for line in fixlines]
        fix_df['duration'] = [float(line[4]) for line in fixlines]
        fix_df['x_pos'] = [float(line[5]) for line in fixlines]
        fix_df['y_pos'] = [float(line[6]) for line in fixlines]
        fix_df['pup_size'] = [float(line[7]) for line in fixlines]
        fix_df['valid'] = 'yes'
        
        if not rec_last_fix:
            fix_df.loc[fix_number - 1, 'valid'] = 'no'
        
        if lump_fix:
            short_index = [ind for ind in range(fix_number) 
                          if fix_df.loc[ind, 'duration'] <= ln and fix_df.loc[ind, 'valid'] == 'yes']
            endindex = fix_number - 2 if not rec_last_fix else fix_number - 1
            fix_df = _lump_fix(fix_df, endindex, short_index, addtime, ln, zn)
    
    elif eyerec == 'LR':
        num_left = sum(1 for line in fixlines if line[1] == 'L')
        num_right = sum(1 for line in fixlines if line[1] == 'R')
        
        if num_left == 0:
            print('Warning! No left eye Fix under both eyes Fix!')
        if num_right == 0:
            print('Warning! No right eye Fix under both eyes Fix!')
        
        last_lr = fixlines[-1][1] if fixlines else 'R'
        
        # Process left eye
        if num_left > 0:
            fix_df1 = pd.DataFrame(np.zeros((num_left, 23)), columns=columns)
            fix_df1['subj'] = srcfile.split('.')[0]
            fix_df1['trial_id'] = int(trial_id)
            fix_df1['trial_type'] = trial_type
            fix_df1['sampfreq'] = int(sampfreq)
            fix_df1['script'] = script
            fix_df1['sessdate'] = sessdate
            fix_df1['srcfile'] = srcfile
            fix_df1['trialstart'] = trialstart
            fix_df1['trialend'] = trialend
            fix_df1['tdur'] = tdur
            fix_df1['recstart'] = recstart
            fix_df1['recend'] = recend
            fix_df1['blinks'] = int(blink_number)
            
            cur = 0
            for line in fixlines:
                if line[1] == 'L':
                    fix_df1.loc[cur, 'eye'] = line[1]
                    fix_df1.loc[cur, 'start_time'] = float(line[2])
                    fix_df1.loc[cur, 'end_time'] = float(line[3])
                    fix_df1.loc[cur, 'duration'] = float(line[4])
                    fix_df1.loc[cur, 'x_pos'] = float(line[5])
                    fix_df1.loc[cur, 'y_pos'] = float(line[6])
                    fix_df1.loc[cur, 'pup_size'] = float(line[7])
                    cur += 1
            
            fix_df1['valid'] = 'yes'
            if not rec_last_fix and last_lr == 'L':
                fix_df1.loc[num_left - 1, 'valid'] = 'no'
            
            if lump_fix:
                short_index1 = [ind for ind in range(num_left) 
                               if fix_df1.loc[ind, 'duration'] <= ln and fix_df1.loc[ind, 'valid'] == 'yes']
                endindex1 = fix_number - 2 if not rec_last_fix and num_left == fix_number else num_left - 1
                fix_df1 = _lump_fix(fix_df1, endindex1, short_index1, addtime, ln, zn)
        
        # Process right eye
        if num_right > 0:
            fix_df2 = pd.DataFrame(np.zeros((num_right, 23)), columns=columns)
            fix_df2['subj'] = srcfile.split('.')[0]
            fix_df2['trial_id'] = int(trial_id)
            fix_df2['trial_type'] = trial_type
            fix_df2['sampfreq'] = int(sampfreq)
            fix_df2['script'] = script
            fix_df2['sessdate'] = sessdate
            fix_df2['srcfile'] = srcfile
            fix_df2['trialstart'] = trialstart
            fix_df2['trialend'] = trialend
            fix_df2['tdur'] = tdur
            fix_df2['recstart'] = recstart
            fix_df2['recend'] = recend
            fix_df2['blinks'] = int(blink_number)
            
            cur = 0
            for line in fixlines:
                if line[1] == 'R':
                    fix_df2.loc[cur, 'eye'] = line[1]
                    fix_df2.loc[cur, 'start_time'] = float(line[2])
                    fix_df2.loc[cur, 'end_time'] = float(line[3])
                    fix_df2.loc[cur, 'duration'] = float(line[4])
                    fix_df2.loc[cur, 'x_pos'] = float(line[5])
                    fix_df2.loc[cur, 'y_pos'] = float(line[6])
                    fix_df2.loc[cur, 'pup_size'] = float(line[7])
                    cur += 1
            
            fix_df2['valid'] = 'yes'
            if not rec_last_fix and last_lr == 'R':
                fix_df2.loc[num_right - 1, 'valid'] = 'no'
            
            if lump_fix:
                short_index2 = [ind for ind in range(num_right) 
                               if fix_df2.loc[ind, 'duration'] <= ln and fix_df2.loc[ind, 'valid'] == 'yes']
                endindex2 = fix_number - 2 if not rec_last_fix and num_right == fix_number else num_right - 1
                fix_df2 = _lump_fix(fix_df2, endindex2, short_index2, addtime, ln, zn)
        
        # Merge
        fix_df = pd.DataFrame(columns=columns)
        if num_left > 0:
            fix_df = pd.concat([fix_df, fix_df1], ignore_index=True)
        if num_right > 0:
            fix_df = pd.concat([fix_df, fix_df2], ignore_index=True)
    
    if lump_fix:
        for ind in range(len(fix_df)):
            if fix_df.loc[ind, 'duration'] < mn:
                fix_df.loc[ind, 'valid'] = 'no'
    
    fix_df['line_no'] = np.nan
    fix_df['region_no'] = np.nan
    
    return fix_df


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
        print('Warning! crlSac start/end need check!')
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
    
    crl_sac = pd.DataFrame(np.zeros((len(lines), 15)),
                           columns=['subj', 'trial_id', 'eye', 'startline', 'endline',
                                   'SaclineIndex', 'start_time', 'end_time', 'duration',
                                   'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos', 'ampl', 'pk'])
    crl_sac['subj'] = sac_df['subj'].iloc[0]
    crl_sac['trial_id'] = sac_df['trial_id'].iloc[0]
    
    for cur, item in enumerate(lines):
        cur_sac = sac_df.loc[item[3]]
        crl_sac.loc[cur, 'eye'] = cur_sac['eye']
        crl_sac.loc[cur, 'startline'] = item[1]
        crl_sac.loc[cur, 'endline'] = item[2]
        crl_sac.loc[cur, 'SaclineIndex'] = item[3]
        crl_sac.loc[cur, 'start_time'] = cur_sac['start_time']
        crl_sac.loc[cur, 'end_time'] = cur_sac['end_time']
        crl_sac.loc[cur, 'duration'] = cur_sac['duration']
        crl_sac.loc[cur, 'x1_pos'] = cur_sac['x1_pos']
        crl_sac.loc[cur, 'y1_pos'] = cur_sac['y1_pos']
        crl_sac.loc[cur, 'x2_pos'] = cur_sac['x2_pos']
        crl_sac.loc[cur, 'y2_pos'] = cur_sac['y2_pos']
        crl_sac.loc[cur, 'ampl'] = cur_sac['ampl']
        crl_sac.loc[cur, 'pk'] = cur_sac['pk']
    
    return crl_sac, question


def _rec_sac(exp_type: str, trial_id: int, blinklines: List, saclines: List,
             sampfreq: int, eyerec: str, script: str, sessdate: str, srcfile: str,
             trial_type: str, trialstart: int, trialend: int, tdur: int,
             recstart: int, recend: int) -> pd.DataFrame:
    """
    Record saccade data from trials.
    
    Args:
        exp_type: Type of experiment
        trial_id: Trial ID
        blinklines: Blink data lines
        saclines: Saccade data lines
        sampfreq: Sampling frequency
        eyerec: Eye recorded
        script: Script file
        sessdate: Session date
        srcfile: Source file
        trial_type: Trial type
        trialstart: Trial start time
        trialend: Trial end time
        tdur: Trial duration
        recstart: Recording start time
        recend: Recording end time
        
    Returns:
        Saccade DataFrame
    """
    blink_number = len(blinklines)
    
    # Remove invalid saccades
    saclines = [line for line in saclines 
                if all(line[ind] != '.' for ind in range(2, 11))]
    sac_number = len(saclines)
    
    columns = ['subj', 'trial_id', 'trial_type', 'sampfreq', 'script', 'sessdate',
               'srcfile', 'trialstart', 'trialend', 'tdur', 'recstart', 'recend',
               'blinks', 'eye', 'start_time', 'end_time', 'duration', 'x1_pos',
               'y1_pos', 'x2_pos', 'y2_pos', 'ampl', 'pk', 'line_no']
    
    sac_df = pd.DataFrame(np.zeros((sac_number, 24)), columns=columns)
    sac_df['subj'] = srcfile.split('.')[0]
    sac_df['trial_id'] = int(trial_id)
    sac_df['trial_type'] = trial_type
    sac_df['sampfreq'] = int(sampfreq)
    sac_df['script'] = script
    sac_df['sessdate'] = sessdate
    sac_df['srcfile'] = srcfile
    sac_df['trialstart'] = trialstart
    sac_df['trialend'] = trialend
    sac_df['tdur'] = tdur
    sac_df['recstart'] = recstart
    sac_df['recend'] = recend
    sac_df['blinks'] = int(blink_number)
    
    if eyerec in ['L', 'R']:
        sac_df['eye'] = [line[1] for line in saclines]
        sac_df['start_time'] = [float(line[2]) for line in saclines]
        sac_df['end_time'] = [float(line[3]) for line in saclines]
        sac_df['duration'] = [float(line[4]) for line in saclines]
        sac_df['x1_pos'] = [float(line[5]) for line in saclines]
        sac_df['y1_pos'] = [float(line[6]) for line in saclines]
        sac_df['x2_pos'] = [float(line[7]) for line in saclines]
        sac_df['y2_pos'] = [float(line[8]) for line in saclines]
        sac_df['ampl'] = [float(line[9]) for line in saclines]
        sac_df['pk'] = [float(line[10]) for line in saclines]
    elif eyerec == 'LR':
        num_left = sum(1 for line in saclines if line[1] == 'L')
        num_right = sum(1 for line in saclines if line[1] == 'R')
        
        if num_left == 0:
            print('Warning! Both eyes fixations recorded, but no left eye fixation data!')
        if num_right == 0:
            print('Warning! Both eyes fixations recorded, but no right eye fixation data!')
        
        cur = 0
        for line in saclines:
            if line[1] == 'L':
                sac_df.loc[cur, 'eye'] = line[1]
                sac_df.loc[cur, 'start_time'] = float(line[2])
                sac_df.loc[cur, 'end_time'] = float(line[3])
                sac_df.loc[cur, 'duration'] = float(line[4])
                sac_df.loc[cur, 'x1_pos'] = float(line[5])
                sac_df.loc[cur, 'y1_pos'] = float(line[6])
                sac_df.loc[cur, 'x2_pos'] = float(line[7])
                sac_df.loc[cur, 'y2_pos'] = float(line[8])
                sac_df.loc[cur, 'ampl'] = float(line[9])
                sac_df.loc[cur, 'pk'] = float(line[10])
                cur += 1
        for line in saclines:
            if line[1] == 'R':
                sac_df.loc[cur, 'eye'] = line[1]
                sac_df.loc[cur, 'start_time'] = float(line[2])
                sac_df.loc[cur, 'end_time'] = float(line[3])
                sac_df.loc[cur, 'duration'] = float(line[4])
                sac_df.loc[cur, 'x1_pos'] = float(line[5])
                sac_df.loc[cur, 'y1_pos'] = float(line[6])
                sac_df.loc[cur, 'x2_pos'] = float(line[7])
                sac_df.loc[cur, 'y2_pos'] = float(line[8])
                sac_df.loc[cur, 'ampl'] = float(line[9])
                sac_df.loc[cur, 'pk'] = float(line[10])
                cur += 1
    
    sac_df['line_no'] = np.nan
    
    return sac_df


def _mod_reg_df(reg_df: pd.DataFrame, add_char_sp: int) -> None:
    """
    Modify region DataFrame boundaries for overshoot fixation capture.
    
    Args:
        reg_df: Region DataFrame (modified in place)
        add_char_sp: Number of single character spaces to add
    """
    reg_df['mod_x1'] = reg_df['x1_pos'].copy()
    reg_df['mod_x2'] = reg_df['x2_pos'].copy()
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
            print(f'{subj_id}.asc does not exist!')
            asc_file_exist = False
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith('.asc'):
                    subj = name.split('.')[0]
                    asc_file_dic[subj] = os.path.join(direct, subj, name)
        if not asc_file_dic:
            print('No ascii files in subfolders!')
            asc_file_exist = False
    
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
            print(f'{subj_id}{csv_filetype}.csv does not exist!')
            csv_file_exist = False
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(target_file_end):
                    subj = name.split(target_file_end)[0]
                    csv_file_dic[subj] = os.path.join(direct, subj, name)
        if not csv_file_dic:
            print('No csv files in subfolders!')
            csv_file_exist = False
    
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
            print(f'No region file exists in {direct}!')
            regfile_exist = False
    else:
        for regfile in regfile_name_list:
            regfile_name = os.path.join(direct, regfile)
            if os.path.isfile(regfile_name):
                regfile_dic[regfile] = regfile_name
            else:
                print(f'{regfile} does not exist!')
                regfile_exist = False
    
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
            print(f'{subj_id}{fix_rep_name_end} does not exist!')
            fix_rep_exist = False
    elif sit == 1:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(fix_rep_name_end):
                    subj = name.split(fix_rep_name_end)[0]
                    fix_rep_dic[subj] = os.path.join(direct, subj, name)
        if not fix_rep_dic:
            print('No fixation report txt files in subfolders!')
            fix_rep_exist = False
    
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
        if (not np.isnan(stamp_df_temp.loc[cur_stamp, 'x_pos1']) and 
            stamp_df_temp.loc[cur_stamp, 'Fix_Sac'] == 'Fix' and 
            not np.isnan(stamp_df_temp.loc[cur_stamp, 'line_no'])):
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
        print(f"Read ASC: {f.name}")
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
        
        print(f"Read Sac: Trial {trial_id}, Type {trial_type}")
        sac_df_temp = _rec_sac(exp_type, trial_id, blinklines, saclines, sampfreq, eyerec,
                               script, sessdate, srcfile, trial_type, trialstart, trialend,
                               tdur, recstart, recend)
        sac_df = pd.concat([sac_df, sac_df_temp], ignore_index=True)
        
        print(f"Read Fix: Trial {trial_id}, Type {trial_type}")
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
        print(f"Read ASC: {f.name}")
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
        
        print(f"Read Stamped Eye Movements: Trial {trial_id}; Type {trial_type}")
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


def cal_crl_sac_fix(direct: str, subj_id: str, regfile_name_list: List[str],
                    exp_type: str, classify_method: str = 'DIFF',
                    rec_status: bool = True, diff_ratio: float = 0.6,
                    frontrange_ratio: float = 0.2, y_range: int = 60,
                    add_char_sp: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate cross-line saccades and fixations.
    
    Args:
        direct: Directory for CSV and output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
        rec_status: Record questionable data (default True)
        diff_ratio: Ratio for cross-line detection (default 0.6)
        frontrange_ratio: Ratio for backward cross-line check (default 0.2)
        y_range: Maximum y difference for line crossing (default 60)
        add_char_sp: Number of characters to add for overshoot (default 1)
        
    Returns:
        Tuple of (new_sac_df, crl_sac, new_fix_df, crl_fix)
    """
    sac_file_exist, sac_file_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
    fix_file_exist, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if not (sac_file_exist and fix_file_exist and regfile_exist):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    sac_df = pd.read_csv(sac_file_dic[subj_id], sep=',')
    fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',')
    new_sac_df = pd.DataFrame()
    new_fix_df = pd.DataFrame()
    crl_sac = pd.DataFrame()
    crl_fix = pd.DataFrame()
    
    print(f"Subj: {subj_id}")
    
    for trial_id in sac_df.trial_id.unique():
        trial_type = sac_df[sac_df.trial_id == trial_id].trial_type.iloc[0]
        reg_df = _get_reg_df(regfile_dic, trial_type)
        _mod_reg_df(reg_df, add_char_sp)
        
        print(f"Get crlSac: Trial {trial_id}, Type {trial_type}")
        sac_df_temp = sac_df[sac_df.trial_id == trial_id].reset_index(drop=True)
        crl_sac_temp, question = _get_crl_sac(reg_df, sac_df_temp, diff_ratio, frontrange_ratio, y_range)
        new_sac_df = pd.concat([new_sac_df, sac_df_temp], ignore_index=True)
        crl_sac = pd.concat([crl_sac, crl_sac_temp], ignore_index=True)
        
        if rec_status and question:
            with open(os.path.join(direct, 'log.txt'), 'a+') as logfile:
                logfile.write(f'Subj: {subj_id} Trial {trial_id} crlSac start/end need check!\n')
        
        print(f"Get Fix: Trial {trial_id}, Type {trial_type}")
        fix_df_temp = fix_df[fix_df.trial_id == trial_id].reset_index(drop=True)
        crl_fix_temp, question = _get_crl_fix(reg_df, crl_sac_temp, fix_df_temp, classify_method,
                                              diff_ratio, frontrange_ratio, y_range)
        
        # Assign region_no
        for cur_fix in range(len(fix_df_temp)):
            if not np.isnan(fix_df_temp.loc[cur_fix, 'line_no']):
                indlist = reg_df[
                    (reg_df['line_no'] == fix_df_temp.loc[cur_fix, 'line_no']) & 
                    (reg_df['mod_x1'] <= fix_df_temp.loc[cur_fix, 'x_pos']) & 
                    (reg_df['mod_x2'] >= fix_df_temp.loc[cur_fix, 'x_pos'])
                ].index.tolist()
                if len(indlist) == 1:
                    fix_df_temp.loc[cur_fix, 'region_no'] = int(reg_df.WordID[indlist[0]])
                else:
                    fix_df_temp.loc[cur_fix, 'region_no'] = np.nan
            else:
                fix_df_temp.loc[cur_fix, 'region_no'] = np.nan
        
        new_fix_df = pd.concat([new_fix_df, fix_df_temp], ignore_index=True)
        crl_fix = pd.concat([crl_fix, crl_fix_temp], ignore_index=True)
        
        if rec_status and question:
            with open(os.path.join(direct, 'log.txt'), 'a+') as logfile:
                logfile.write(f'Subj: {subj_id} Trial {trial_id} crlFix start/end need check!\n')
    
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
                                  exp_type: str, classify_method: str = 'DIFF',
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
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
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
                                    exp_type: str, classify_method: str = 'DIFF',
                                    rec_status: bool = True, diff_ratio: float = 0.6,
                                    frontrange_ratio: float = 0.2, y_range: int = 60,
                                    add_char_sp: int = 1) -> None:
    """
    Batch process all subjects' saccades and fixations.
    
    Args:
        direct: Directory containing all CSV files
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
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
                     exp_type: str, classify_method: str = 'DIFF',
                     rec_status: bool = True, diff_ratio: float = 0.6,
                     frontrange_ratio: float = 0.2, y_range: int = 60,
                     add_char_sp: int = 1, rec_last_fix: bool = False,
                     lump_fix: bool = True, ln: int = 50, zn: int = 50,
                     mn: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read ASC file and extract fixation/saccade data with cross-line classification.
    
    Combines reading ASCII file, extracting saccades/fixations, and calculating
    cross-line saccades and fixations in one operation.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        regfile_name_list: List of region file names
        exp_type: Type of experiment ('RAN', 'RP')
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
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
        
    Returns:
        Tuple of (sac_df, crl_sac, fix_df, crl_fix)
    """
    ascfile_exist, ascfile_dic = _crt_asc_dic(0, direct, subj_id)
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if not (ascfile_exist and regfile_exist):
        return None, None, None, None
    
    # Read ASCII file
    with open(ascfile_dic[subj_id], 'r') as f:
        print(f"Read ASC: {f.name}")
        lines = f.readlines()
    
    script, sessdate, srcfile = _get_header(lines)
    t_idx, t_lines = _get_trial_reg(lines)
    
    sac_df = pd.DataFrame(columns=['subj', 'trial_id', 'trial_type', 'sampfreq', 'script',
                                   'sessdate', 'srcfile', 'trialstart', 'trialend', 'tdur',
                                   'recstart', 'recend', 'blinks', 'eye', 'start_time',
                                   'end_time', 'duration', 'x1_pos', 'y1_pos', 'x2_pos',
                                   'y2_pos', 'ampl', 'pk', 'line_no'])
    fix_df = pd.DataFrame(columns=['subj', 'trial_id', 'trial_type', 'sampfreq', 'script',
                                   'sessdate', 'srcfile', 'trialstart', 'trialend', 'tdur',
                                   'recstart', 'recend', 'blinks', 'eye', 'start_time',
                                   'end_time', 'duration', 'x_pos', 'y_pos', 'pup_size',
                                   'valid', 'line_no', 'region_no'])
    crl_sac = pd.DataFrame(columns=['subj', 'trial_id', 'eye', 'startline', 'endline',
                                    'SaclineIndex', 'start_time', 'end_time', 'duration',
                                    'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos', 'ampl', 'pk'])
    crl_fix = pd.DataFrame(columns=['subj', 'trial_id', 'eye', 'startline', 'endline',
                                    'FixlineIndex', 'start_time', 'end_time', 'duration',
                                    'x_pos', 'y_pos', 'pup_size', 'valid'])
    
    for ind in range(len(t_lines)):
        triallines = lines[t_lines[ind, 0] + 1:t_lines[ind, 1]]
        # Convert 0-indexed trial_id from ASC file to 1-indexed
        trial_id = int(t_idx[ind, 0].split(' ')[-1]) + 1
        blinklines, fixlines, saclines, sampfreq, eyerec = _get_blink_fix_sac_sampfreq_eyerec(triallines, 0)
        trial_type, trialstart, trialend, tdur, recstart, recend = _get_tdur(triallines)
        reg_df = _get_reg_df(regfile_dic, trial_type)
        _mod_reg_df(reg_df, add_char_sp)
        
        # Read and process saccades
        print(f"Read Sac and Get crlSac: Trial {trial_id}, Type {trial_type}")
        sac_df_temp = _rec_sac(exp_type, trial_id, blinklines, saclines, sampfreq,
                               eyerec, script, sessdate, srcfile, trial_type,
                               trialstart, trialend, tdur, recstart, recend)
        crl_sac_temp, question = _get_crl_sac(reg_df, sac_df_temp, diff_ratio,
                                              frontrange_ratio, y_range)
        sac_df = pd.concat([sac_df, sac_df_temp], ignore_index=True)
        crl_sac = pd.concat([crl_sac, crl_sac_temp], ignore_index=True)
        
        if rec_status and question:
            log_path = os.path.join(direct, 'log.txt')
            with open(log_path, 'a+') as logfile:
                logfile.write(f'Subj: {sac_df_temp.subj.iloc[0]} Trial {trial_id} crlSac start/end need check!\n')
        
        # Read and process fixations
        print(f"Read Fix and Get crlFix: Trial {trial_id}, Type {trial_type}")
        fix_df_temp = _rec_fix(exp_type, trial_id, blinklines, fixlines, sampfreq,
                               eyerec, script, sessdate, srcfile, trial_type,
                               trialstart, trialend, tdur, recstart, recend,
                               rec_last_fix, lump_fix, ln, zn, mn)
        crl_fix_temp, question = _get_crl_fix(reg_df, crl_sac_temp, fix_df_temp,
                                              classify_method, diff_ratio,
                                              frontrange_ratio, y_range)
        
        # Assign region numbers
        for cur_fix in range(len(fix_df_temp)):
            if not pd.isna(fix_df_temp.loc[cur_fix, 'line_no']):
                indlist = reg_df[(reg_df['line_no'] == fix_df_temp.loc[cur_fix, 'line_no']) &
                                 ((reg_df['mod_x1'] <= fix_df_temp.loc[cur_fix, 'x_pos']) &
                                  (reg_df['mod_x2'] >= fix_df_temp.loc[cur_fix, 'x_pos']))].index.tolist()
                if len(indlist) == 1:
                    fix_df_temp.loc[cur_fix, 'region_no'] = int(reg_df.WordID.iloc[indlist[0]])
                else:
                    fix_df_temp.loc[cur_fix, 'region_no'] = np.nan
            else:
                fix_df_temp.loc[cur_fix, 'region_no'] = np.nan
        
        fix_df = pd.concat([fix_df, fix_df_temp], ignore_index=True)
        crl_fix = pd.concat([crl_fix, crl_fix_temp], ignore_index=True)
        
        if rec_status and question:
            log_path = os.path.join(direct, 'log.txt')
            with open(log_path, 'a+') as logfile:
                logfile.write(f'Subj: {fix_df_temp.subj.iloc[0]} Trial {trial_id} crlFix start/end need check!\n')
    
    return sac_df, crl_sac, fix_df, crl_fix


def read_cal_write_srr_asc(direct: str, subj_id: str, regfile_name_list: List[str],
                           exp_type: str, classify_method: str = 'DIFF',
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
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
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
                             exp_type: str, classify_method: str = 'DIFF',
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
        classify_method: 'DIFF' or 'SAC' (default 'DIFF')
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
        print(f"Read ASC: {f.name}")
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
        
        print(f"Read Time Stamped Data: Trial {trial_id}, Type {trial_type}")
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
        print(f"Invalid align_method: {align_method}")
        return None
    
    stamp_df = pd.read_csv(stamp_dic[subj_id], sep=',')
    new_stamp_df = pd.DataFrame()
    
    print(f"Processing subject: {subj_id}")
    
    for trial_id in stamp_df['trial_id'].unique():
        trial_type = stamp_df.loc[stamp_df.trial_id == trial_id, 'trial_type'].iloc[0]
        trialstart = stamp_df.loc[stamp_df.trial_id == trial_id, 'trialstart'].iloc[0]
        
        reg_df = _get_reg_df(regfile_dic, trial_type)
        _mod_reg_df(reg_df, add_char_sp)
        
        print(f"  Processing trial {trial_id}, type: {trial_type}")
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
