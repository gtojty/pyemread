# -*- coding: utf-8 -*-
"""
Pyemread - cal module
====================

Created on Wed Feb 22 09:21:44 2017
Updated for Python 3 compatibility: 2024

This module provides functions for calculating regional summaries of
widely-adopted eye-movement measures used in reading research.

Eye-Movement Measures Calculated
-------------------------------
Whole-trial measures:
    - tffixos: Total offset of first-pass fixation from sentence beginning
    - tffixurt: Total duration of first-pass fixations
    - tfixcnt: Total number of valid fixations in trial
    - tregrcnt: Total number of regressive saccades

Region (word) measures:
    - fpurt: First-pass fixation time
    - fpcount: Number of first-pass fixations
    - fpregres: Whether first-pass regression occurred
    - fpregreg: Region where first-pass regression ends
    - fpregchr: Character offset where first-pass regression ends
    - ffos: Offset of first first-pass fixation
    - ffixurt: Duration of first first-pass fixation
    - spilover: Duration of first fixation beyond region
    - rpurt: Regression path duration
    - rpcount: Number of fixations in regression path
    - rpregreg: Smallest region visited by regression path
    - rpregchr: Character offset in smallest regression region
    - spurt: Second-pass fixation time
    - spcount: Number of second-pass fixations

Usage
-----
    from pyemread import cal
    cal.cal_write_em(direct, subjID, regfileNameList)

Or:
    import pyemread as pr
    pr.cal.cal_write_em(direct, subjID, regfileNameList)

Authors
-------
Tao Gong, David Braze

License
-------
MIT License
"""

from __future__ import annotations

__author__ = "Tao Gong and David Braze"
__copyright__ = "Copyright 2017-2024, The Pyemread Project"
__credits__ = ["Tao Gong", "David Braze", "Jonathan Gordils", "Hosung Nam"]
__license__ = "MIT"
__version__ = "2.1.1"
__maintainer__ = ["Tao Gong", "David Braze"]
__email__ = ["gtojty@gmail.com", "davebraze@gmail.com"]
__status__ = "Production"

import os
import fnmatch
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np


# =============================================================================
# Helper Functions
# =============================================================================

def _get_reg_df(regfile_dic: Dict[str, str], trial_type: str) -> pd.DataFrame:
    """
    Get the region file data frame based on trial type.
    
    Parameters
    ----------
    regfile_dic : dict
        Dictionary mapping region file names to their full paths.
    trial_type : str
        Current trial type identifier.
    
    Returns
    -------
    pd.DataFrame
        Region file data frame.
    
    Raises
    ------
    ValueError
        If trial_type is not found in regfile_dic.
    """
    regfile_name = trial_type + '.region.csv'
    if regfile_name not in regfile_dic:
        raise ValueError(f"Invalid trial_type: {trial_type}")
    return pd.read_csv(regfile_dic[regfile_name], sep=',', encoding='utf-8')


def _crt_csv_dic(
    sit: int,
    direct: str,
    subj_id: str,
    csv_file_type: str
) -> Tuple[bool, Dict[str, str]]:
    """
    Create dictionary for different types of CSV files.
    
    Parameters
    ----------
    sit : int
        Situation flag: 0 = subjID is given, 1 = no subjID (batch mode).
    direct : str
        Root directory containing CSV files in subfolders.
    subj_id : str
        Subject ID (used when sit=0).
    csv_file_type : str
        Type of CSV file: "_Stamp", "_Sac", "_crlSac", "_Fix", "_crlFix".
    
    Returns
    -------
    tuple
        (csv_file_exist: bool, csv_file_dic: dict)
    """
    csv_file_exist = True
    csv_file_dic = {}
    
    target_file_end = csv_file_type + '.csv'
    
    if sit == 0:
        file_name = os.path.join(direct, subj_id, subj_id + target_file_end)
        if os.path.isfile(file_name):
            csv_file_dic[subj_id] = file_name
        else:
            print(f"{subj_id}{csv_file_type}.csv does not exist!")
            csv_file_exist = False
    elif sit == 1:
        # Search all subfolders for CSV files
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(target_file_end):
                    subj = name.split(target_file_end)[0]
                    csv_file_dic[subj] = os.path.join(direct, subj, name)
        if len(csv_file_dic) == 0:
            print('No CSV files found in subfolders!')
            csv_file_exist = False
    
    return csv_file_exist, csv_file_dic


def _crt_region_dic(
    direct: str,
    regfile_name_list: List[str]
) -> Tuple[bool, Dict[str, str]]:
    """
    Create region file dictionary.
    
    Parameters
    ----------
    direct : str
        Root directory containing region files.
    regfile_name_list : list
        List of region file names (empty list = auto-detect).
    
    Returns
    -------
    tuple
        (regfile_exist: bool, regfile_dic: dict)
    """
    regfile_exist = True
    regfile_dic = {}
    
    target_file_end = '.region.csv'
    
    if len(regfile_name_list) == 0:
        # Automatically gather all region files
        for file in os.listdir(direct):
            if fnmatch.fnmatch(file, '*' + target_file_end):
                regfile_dic[str(file)] = os.path.join(direct, str(file))
        if len(regfile_dic) == 0:
            print(f'No region file exists in {direct}!')
            regfile_exist = False
    else:
        # Check whether particular region files exist
        for regfile in regfile_name_list:
            regfile_name = os.path.join(direct, regfile)
            if os.path.isfile(regfile_name):
                regfile_dic[regfile] = regfile_name
            else:
                print(f'{regfile} does not exist!')
                regfile_exist = False
    
    return regfile_exist, regfile_dic


# =============================================================================
# Eye-Movement Calculation Helper Functions
# =============================================================================

def _mod_em(emdf: pd.DataFrame, add_char_sp: int) -> None:
    """
    Modify EMDF's mod_x1 and mod_x2 to add space to boundaries.
    
    Adds space to boundaries of line starting and ending words to catch
    overshoot fixations.
    
    Parameters
    ----------
    emdf : pd.DataFrame
        Result data frame (modified in place).
    add_char_sp : int
        Number of single character spaces to add.
    """
    emdf['mod_x1'] = emdf['x1_pos'].copy()
    emdf['mod_x2'] = emdf['x2_pos'].copy()
    
    add_dist = add_char_sp * (emdf.loc[0, 'x2_pos'] - emdf.loc[0, 'x1_pos']) / float(emdf.loc[0, 'reglen'])
    
    for cur_em in range(len(emdf)):
        if cur_em == 0:
            # First word: add left side
            emdf.loc[cur_em, 'mod_x1'] -= add_dist
        elif cur_em == len(emdf) - 1:
            # Last word: add right side
            emdf.loc[cur_em, 'mod_x2'] += add_dist
        else:
            # Check whether it is a line ending or line starting word
            if emdf.loc[cur_em - 1, 'line_no'] == emdf.loc[cur_em, 'line_no'] - 1:
                # Current region is line starting: add left side
                emdf.loc[cur_em, 'mod_x1'] -= add_dist
            elif emdf.loc[cur_em + 1, 'line_no'] == emdf.loc[cur_em, 'line_no'] + 1:
                # Current region is line ending: add right side
                emdf.loc[cur_em, 'mod_x2'] += add_dist


def _chk_fp_fix(
    fix_df: pd.DataFrame,
    emdf: pd.DataFrame,
    cur_fix: int,
    cur_em: int
) -> Tuple[int, int]:
    """
    Calculate first-pass fixation measures.
    
    Measures calculated:
        - fpurt: First-pass fixation time
        - fpcount: Number of first-pass fixations
        - ffos: Offset of first first-pass fixation
        - ffixurt: Duration of first first-pass fixation
        - spilover: Duration of first fixation beyond region
    
    Parameters
    ----------
    fix_df : pd.DataFrame
        Fixation data of the trial.
    emdf : pd.DataFrame
        Result data frame.
    cur_fix : int
        Current fixation index.
    cur_em : int
        Current region index.
    
    Returns
    -------
    tuple
        (st_fix, end_fix): Starting and ending fixation indices.
    """
    # First-pass fixation time
    emdf.loc[cur_em, 'fpurt'] += fix_df.loc[cur_fix, 'duration']
    # Number of first-pass fixations
    emdf.loc[cur_em, 'fpcount'] += 1
    
    # Offset of first first-pass fixation
    mod_width = emdf.loc[cur_em, 'mod_x2'] - emdf.loc[cur_em, 'mod_x1']
    if mod_width != 0:
        emdf.loc[cur_em, 'ffos'] = np.ceil(
            (fix_df.loc[cur_fix, 'x_pos'] - emdf.loc[cur_em, 'mod_x1']) / 
            float(mod_width) * emdf.loc[cur_em, 'reglen']
        ) - 1
    
    # First first-pass fixation duration
    emdf.loc[cur_em, 'ffixurt'] += fix_df.loc[cur_fix, 'duration']
    
    # Locate starting and ending indices of first-pass fixation
    st_fix, end_fix = cur_fix, cur_fix + 1
    
    # Skip NaN line numbers
    while end_fix < len(fix_df) - 1 and np.isnan(fix_df.loc[end_fix, 'line_no']):
        end_fix += 1
    
    # Continue while fixation is in same word region
    while (end_fix < len(fix_df) - 1 and 
           fix_df.loc[end_fix, 'valid'] == 'yes' and 
           fix_df.loc[end_fix, 'line_no'] == emdf.loc[cur_em, 'line_no'] and 
           emdf.loc[cur_em, 'mod_x1'] <= fix_df.loc[end_fix, 'x_pos'] and 
           fix_df.loc[end_fix, 'x_pos'] <= emdf.loc[cur_em, 'mod_x2']):
        emdf.loc[cur_em, 'fpurt'] += fix_df.loc[end_fix, 'duration']
        emdf.loc[cur_em, 'fpcount'] += 1
        end_fix += 1
        while end_fix < len(fix_df) - 1 and np.isnan(fix_df.loc[end_fix, 'line_no']):
            end_fix += 1
    
    # Calculate spillover
    if (end_fix < len(fix_df) and 
        fix_df.loc[end_fix, 'valid'] == 'yes' and 
        not np.isnan(fix_df.loc[end_fix, 'line_no'])):
        emdf.loc[cur_em, 'spilover'] += fix_df.loc[end_fix, 'duration']
    
    return st_fix, end_fix


def _chk_fp_reg(
    fix_df: pd.DataFrame,
    emdf: pd.DataFrame,
    st_fix: int,
    end_fix: int,
    cur_em: int
) -> None:
    """
    Calculate first-pass regression measures.
    
    Measures calculated:
        - fpregres: Whether first-pass regression occurred (1) or not (0)
        - fpregreg: Region where first-pass regression ends
        - fpregchr: Character offset where first-pass regression ends
    
    Parameters
    ----------
    fix_df : pd.DataFrame
        Fixation data of the trial.
    emdf : pd.DataFrame
        Result data frame.
    st_fix : int
        Starting fixation index.
    end_fix : int
        Ending fixation index.
    cur_em : int
        Current region index.
    """
    if end_fix >= len(fix_df):
        return
        
    if fix_df.loc[end_fix, 'line_no'] == emdf.loc[cur_em, 'line_no']:
        # Fixation after first-pass is in same line
        if fix_df.loc[end_fix, 'x_pos'] < emdf.loc[cur_em, 'mod_x1']:
            # Regression fixation
            emdf.loc[cur_em, 'fpregres'] = 1
            # Find region where regression fixation falls
            for cur in range(len(emdf)):
                if fix_df.loc[end_fix, 'region_no'] == emdf.loc[cur, 'region']:
                    emdf.loc[cur_em, 'fpregreg'] = emdf.loc[cur, 'region']
                    mod_width = emdf.loc[cur, 'mod_x2'] - emdf.loc[cur, 'mod_x1']
                    if mod_width != 0:
                        char_offset = np.ceil(
                            (fix_df.loc[end_fix, 'x_pos'] - emdf.loc[cur, 'mod_x1']) / 
                            float(mod_width) * emdf.loc[cur, 'reglen']
                        ) - 1
                        if cur == 0:
                            emdf.loc[cur_em, 'fpregchr'] = char_offset
                        else:
                            emdf.loc[cur_em, 'fpregchr'] = sum(emdf.reglen[0:cur-1]) + char_offset
                    break
        else:
            # Forward fixation
            emdf.loc[cur_em, 'fpregres'] = 0
            emdf.loc[cur_em, 'fpregreg'] = 0
            emdf.loc[cur_em, 'fpregchr'] = sum(emdf.reglen)
    else:
        # Fixation after first-pass is in different line
        if fix_df.loc[end_fix, 'line_no'] < emdf.loc[cur_em, 'line_no']:
            # Regression fixation
            emdf.loc[cur_em, 'fpregres'] = 1
            for cur in range(len(emdf)):
                if fix_df.loc[end_fix, 'region_no'] == emdf.loc[cur, 'region']:
                    emdf.loc[cur_em, 'fpregreg'] = emdf.loc[cur, 'region']
                    mod_width = emdf.loc[cur, 'mod_x2'] - emdf.loc[cur, 'mod_x1']
                    if mod_width != 0:
                        char_offset = np.ceil(
                            (fix_df.loc[end_fix, 'x_pos'] - emdf.loc[cur, 'mod_x1']) / 
                            float(mod_width) * emdf.loc[cur, 'reglen']
                        ) - 1
                        if cur == 0:
                            emdf.loc[cur_em, 'fpregchr'] = char_offset
                        else:
                            emdf.loc[cur_em, 'fpregchr'] = sum(emdf.reglen[0:cur-1]) + char_offset
                    break
        else:
            # Forward fixation
            emdf.loc[cur_em, 'fpregres'] = 0
            emdf.loc[cur_em, 'fpregreg'] = 0
            emdf.loc[cur_em, 'fpregchr'] = sum(emdf.reglen)


def _get_reg(fix_df: pd.DataFrame, cur_fix: int, emdf: pd.DataFrame) -> int:
    """
    Find which region a fixation falls into.
    
    Parameters
    ----------
    fix_df : pd.DataFrame
        Fixation data of the trial.
    cur_fix : int
        Current fixation index.
    emdf : pd.DataFrame
        Result data frame.
    
    Returns
    -------
    int
        Index in EMDF, or 0 if not found.
    """
    if np.isnan(fix_df.line_no[cur_fix]) or np.isnan(fix_df.region_no[cur_fix]):
        return 0
    
    ind_list = emdf[
        (emdf['line_no'] == fix_df.line_no[cur_fix]) & 
        (emdf['region'] == fix_df.region_no[cur_fix])
    ].index.tolist()
    
    return ind_list[0] if len(ind_list) == 1 else 0


def _chk_rp_reg(
    fix_df: pd.DataFrame,
    emdf: pd.DataFrame,
    st_fix: int,
    end_fix: int,
    cur_em: int
) -> None:
    """
    Calculate regression path measures.
    
    Measures calculated:
        - rpurt: Regression path duration
        - rpcount: Number of fixations in regression path
        - rpregreg: Smallest region visited by regression path
        - rpregchr: Character offset in smallest region
    
    Parameters
    ----------
    fix_df : pd.DataFrame
        Fixation data of the trial.
    emdf : pd.DataFrame
        Result data frame.
    st_fix : int
        Starting fixation index.
    end_fix : int
        Ending fixation index.
    cur_em : int
        Current region index.
    """
    if emdf.loc[cur_em, 'fpregres'] == 0:
        # No regression, so no regression path
        emdf.loc[cur_em, 'rpurt'] = emdf.loc[cur_em, 'fpurt']
        emdf.loc[cur_em, 'rpcount'] = 0
        emdf.loc[cur_em, 'rpregreg'] = 0
        emdf.loc[cur_em, 'rpregchr'] = sum(emdf.reglen)
    else:
        # There is a regression, find the regression path
        if cur_em == 0:
            # First region: treat as no regression
            emdf.loc[cur_em, 'rpurt'] = emdf.loc[cur_em, 'fpurt']
            emdf.loc[cur_em, 'rpcount'] = 0
            emdf.loc[cur_em, 'rpregreg'] = 0
            emdf.loc[cur_em, 'rpregchr'] = sum(emdf.reglen)
        elif cur_em == len(emdf) - 1:
            # Last region
            if end_fix < len(fix_df):
                emdf.loc[cur_em, 'rpurt'] = emdf.loc[cur_em, 'fpurt'] + fix_df.loc[end_fix, 'duration']
            else:
                emdf.loc[cur_em, 'rpurt'] = emdf.loc[cur_em, 'fpurt']
            emdf.loc[cur_em, 'rpcount'] += 1
            
            cur_fix = end_fix + 1
            leftmost_reg_ind = _get_reg(fix_df, end_fix, emdf)
            leftmost_reg = emdf.region[leftmost_reg_ind]
            leftmost_cur_fix = end_fix
            
            while (cur_fix < len(fix_df) and 
                   fix_df.loc[cur_fix, 'valid'] == 'yes' and 
                   not np.isnan(fix_df.loc[cur_fix, 'line_no'])):
                emdf.loc[cur_em, 'rpurt'] += fix_df.loc[cur_fix, 'duration']
                emdf.loc[cur_em, 'rpcount'] += 1
                new_left_ind = _get_reg(fix_df, cur_fix, emdf)
                new_left = emdf.region[new_left_ind]
                if leftmost_reg > new_left:
                    leftmost_reg_ind = new_left_ind
                    leftmost_reg = new_left
                    leftmost_cur_fix = cur_fix
                cur_fix += 1
            
            emdf.loc[cur_em, 'rpregreg'] = leftmost_reg
            mod_width = emdf.loc[leftmost_reg_ind, 'mod_x2'] - emdf.loc[leftmost_reg_ind, 'mod_x1']
            if mod_width != 0:
                char_offset = np.ceil(
                    (fix_df.loc[leftmost_cur_fix, 'x_pos'] - emdf.loc[leftmost_reg_ind, 'mod_x1']) / 
                    float(mod_width) * emdf.loc[leftmost_reg_ind, 'reglen']
                ) - 1
                if leftmost_reg_ind == 0:
                    emdf.loc[cur_em, 'rpregchr'] = char_offset
                else:
                    emdf.loc[cur_em, 'rpregchr'] = sum(emdf.reglen[0:leftmost_reg_ind]) + char_offset
        else:
            # Middle region
            emdf.loc[cur_em, 'rpurt'] = emdf.loc[cur_em, 'fpurt']
            new_end_fix = end_fix + 1
            
            while (new_end_fix < len(fix_df) and 
                   fix_df.loc[new_end_fix, 'valid'] == 'yes' and 
                   not np.isnan(fix_df.loc[new_end_fix, 'line_no']) and 
                   not np.isnan(fix_df.loc[new_end_fix, 'region_no']) and 
                   fix_df.loc[new_end_fix, 'region_no'] <= fix_df.loc[st_fix, 'region_no']):
                new_end_fix += 1
            
            leftmost_reg_ind = _get_reg(fix_df, end_fix, emdf)
            leftmost_reg = emdf.region[leftmost_reg_ind]
            leftmost_cur_fix = end_fix
            
            for ind_fix in range(end_fix, new_end_fix):
                if not np.isnan(fix_df.loc[ind_fix, 'region_no']):
                    emdf.loc[cur_em, 'rpurt'] += fix_df.loc[ind_fix, 'duration']
                    emdf.loc[cur_em, 'rpcount'] += 1
                    new_left_ind = _get_reg(fix_df, ind_fix, emdf)
                    new_left = emdf.region[new_left_ind]
                    if leftmost_reg > new_left:
                        leftmost_reg_ind = new_left_ind
                        leftmost_reg = new_left
                        leftmost_cur_fix = ind_fix
            
            emdf.loc[cur_em, 'rpregreg'] = leftmost_reg
            mod_width = emdf.loc[leftmost_reg_ind, 'mod_x2'] - emdf.loc[leftmost_reg_ind, 'mod_x1']
            if mod_width != 0:
                char_offset = np.ceil(
                    (fix_df.loc[leftmost_cur_fix, 'x_pos'] - emdf.loc[leftmost_reg_ind, 'mod_x1']) / 
                    float(mod_width) * emdf.loc[leftmost_reg_ind, 'reglen']
                ) - 1
                if leftmost_reg_ind == 0:
                    emdf.loc[cur_em, 'rpregchr'] = char_offset
                else:
                    emdf.loc[cur_em, 'rpregchr'] = sum(emdf.reglen[0:leftmost_reg_ind]) + char_offset


def _chk_sp_fix(
    fix_df: pd.DataFrame,
    emdf: pd.DataFrame,
    end_fix: int,
    cur_em: int
) -> None:
    """
    Calculate second-pass fixation measures.
    
    Measures calculated:
        - spurt: Second-pass fixation time
        - spcount: Number of second-pass fixations
    
    Parameters
    ----------
    fix_df : pd.DataFrame
        Fixation data of the trial.
    emdf : pd.DataFrame
        Result data frame.
    end_fix : int
        Ending fixation index of first-pass reading.
    cur_em : int
        Current region index.
    """
    for cur_fix in range(end_fix, len(fix_df)):
        if fix_df.loc[cur_fix, 'region_no'] == emdf.loc[cur_em, 'region']:
            emdf.loc[cur_em, 'spurt'] += fix_df.loc[cur_fix, 'duration']
            emdf.loc[cur_em, 'spcount'] += 1


def _chk_tffixos(emdf: pd.DataFrame) -> float:
    """
    Calculate total first fixation offset.
    
    Parameters
    ----------
    emdf : pd.DataFrame
        Result data frame.
    
    Returns
    -------
    float
        Total offset of first fixation from sentence beginning.
    """
    tffixos = 0.0
    for ind in range(len(emdf)):
        if not np.isnan(emdf.loc[ind, 'ffos']):
            if ind == 0:
                tffixos += emdf.loc[ind, 'ffos']
            else:
                tffixos += sum(emdf.reglen[0:ind-1]) + emdf.loc[ind, 'ffos']
    return tffixos


def _chk_tregrcnt(sac_df: pd.DataFrame) -> int:
    """
    Calculate total number of regressive saccades.
    
    Parameters
    ----------
    sac_df : pd.DataFrame
        Saccade data of the trial.
    
    Returns
    -------
    int
        Total number of regressive saccades.
    """
    tot_regr = 0
    for ind in range(len(sac_df)):
        crl_info = str(sac_df.line_no[ind]).split('_')
        if len(crl_info) == 1:
            if crl_info != ['nan']:
                # Not crossline saccade
                if sac_df.x1_pos[ind] > sac_df.x2_pos[ind]:
                    tot_regr += 1
        else:
            # Crossline saccade
            if int(float(crl_info[0])) > int(float(crl_info[1])):
                tot_regr += 1
    return tot_regr


def _cal_em(
    reg_df: pd.DataFrame,
    fix_df: pd.DataFrame,
    sac_df: pd.DataFrame,
    emdf: pd.DataFrame
) -> None:
    """
    Calculate eye-movement measures for a trial.
    
    This function calculates all region-based and whole-trial eye-movement
    measures and updates the EMDF data frame in place.
    
    Parameters
    ----------
    reg_df : pd.DataFrame
        Region file data frame.
    fix_df : pd.DataFrame
        Fixation data of the trial.
    sac_df : pd.DataFrame
        Saccade data of the trial.
    emdf : pd.DataFrame
        Result data frame (modified in place).
    
    Notes
    -----
    Eye-movement measures calculated:
    
    Whole-trial measures:
        - tffixos: Total offset of first-pass fixation
        - tffixurt/ttfixurt: Total duration of first-pass fixations
        - tfixcnt: Total number of valid fixations
        - tregrcnt: Total number of regressive saccades
    
    Region measures:
        - fpurt, fpcount: First-pass fixation time and count
        - fpregres, fpregreg, fpregchr: First-pass regression measures
        - ffos, ffixurt: First fixation offset and duration
        - spilover: Spillover fixation duration
        - rpurt, rpcount, rpregreg, rpregchr: Regression path measures
        - spurt, spcount: Second-pass fixation measures
    """
    # Set default values
    emdf['ffos'] = np.nan
    emdf['fpregres'] = np.nan
    emdf['fpregreg'] = np.nan
    emdf['fpregchr'] = np.nan
    emdf['rpregres'] = np.nan
    emdf['rpregreg'] = np.nan
    emdf['rpregchr'] = np.nan
    
    # Calculate region (word) measures
    for cur_em in range(len(emdf)):
        for cur_fix in range(len(fix_df)):
            if fix_df.loc[cur_fix, 'region_no'] == emdf.loc[cur_em, 'region']:
                # Found first-pass fixation on current word
                st_fix, end_fix = _chk_fp_fix(fix_df, emdf, cur_fix, cur_em)
                _chk_fp_reg(fix_df, emdf, st_fix, end_fix, cur_em)
                _chk_rp_reg(fix_df, emdf, st_fix, end_fix, cur_em)
                _chk_sp_fix(fix_df, emdf, end_fix, cur_em)
                # First-pass reading finished, go to next word
                break
    
    # Convert zero values to NaN
    zero_cols = ['fpurt', 'fpcount', 'ffixurt', 'spilover', 'spurt', 'spcount']
    for col in zero_cols:
        emdf.loc[emdf[emdf[col] == 0].index, col] = np.nan
    
    # Set regression measures to NaN where no first-pass fixation
    nan_fpurt = emdf[np.isnan(emdf.fpurt)].index
    emdf.loc[nan_fpurt, 'rpurt'] = np.nan
    emdf.loc[nan_fpurt, 'rpcount'] = np.nan
    emdf.loc[nan_fpurt, 'rpregreg'] = np.nan
    
    # Calculate whole-trial measures
    emdf['tffixos'] = _chk_tffixos(emdf)
    emdf['ttfixurt'] = sum(x for x in emdf.fpurt if not np.isnan(x))
    emdf['tfixcnt'] = len(fix_df[fix_df.valid == 'yes'])
    emdf['tregrcnt'] = _chk_tregrcnt(sac_df)


# =============================================================================
# User Functions
# =============================================================================

def cal_write_em(
    direct: str,
    subj_id: str,
    regfile_name_list: List[str],
    add_char_sp: int = 1
) -> None:
    """
    Calculate and write eye-movement measures for a subject.
    
    Reads fixation and saccade data, calculates eye-movement measures,
    and writes results to CSV files.
    
    Parameters
    ----------
    direct : str
        Directory containing CSV and output files.
    subj_id : str
        Subject ID.
    regfile_name_list : list
        List of region file names.
    add_char_sp : int, optional
        Number of character spaces to add for catching overshoot fixations.
        Default is 1.
    
    Output
    ------
    Creates CSV files: {subj_id}_EM_{trial_type}_{L/R}.csv
    """
    # Check required files
    sac_file_exist, sac_file_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
    fix_file_exist, fix_file_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if not (sac_file_exist and fix_file_exist and regfile_exist):
        return
    
    # Read data
    sac_df = pd.read_csv(sac_file_dic[subj_id], sep=',', encoding='utf-8')
    fix_df = pd.read_csv(fix_file_dic[subj_id], sep=',', encoding='utf-8')
    
    # Process each trial (using actual trial IDs from data)
    for trial_id in sorted(sac_df.trial_id.unique()):
        trial_data = sac_df[sac_df.trial_id == trial_id]
        if len(trial_data) == 0:
            continue
        trial_type = trial_data.trial_type.unique()[0]
        reg_df = _get_reg_df(regfile_dic, trial_type)
        sac_df_temp = sac_df[sac_df.trial_id == trial_id].reset_index(drop=True)
        fix_df_temp = fix_df[fix_df.trial_id == trial_id].reset_index(drop=True)
        
        if len(np.unique(sac_df_temp.eye)) == 1:
            # Single eye data
            eye = np.unique(sac_df_temp.eye)[0]
            eye_name = 'Left' if eye == 'L' else 'Right'
            print(f'Cal EM measures: Subj: {subj_id}, Trial: {trial_id} {eye_name} Eye')
            
            # Create result data frame
            emdf = _create_emdf(reg_df, fix_df_temp, subj_id)
            _mod_em(emdf, add_char_sp)
            _cal_em(reg_df, fix_df_temp, sac_df_temp, emdf)
            
            # Save results
            output_path = os.path.join(direct, subj_id, f'{subj_id}_EM_{trial_type}_{eye}.csv')
            emdf.to_csv(output_path, index=False, encoding='utf-8')
        else:
            # Double eye data
            for eye in ['L', 'R']:
                eye_name = 'Left' if eye == 'L' else 'Right'
                print(f'Cal EM measures: Subj: {subj_id}, Trial: {trial_id} {eye_name} Eye')
                
                sac_df_eye = sac_df_temp[sac_df_temp.eye == eye].reset_index(drop=True)
                fix_df_eye = fix_df_temp[fix_df_temp.eye == eye].reset_index(drop=True)
                
                if len(fix_df_eye) == 0:
                    continue
                
                # Create result data frame
                emdf = _create_emdf(reg_df, fix_df_eye, subj_id)
                _mod_em(emdf, add_char_sp)
                _cal_em(reg_df, fix_df_eye, sac_df_eye, emdf)
                
                # Save results
                output_path = os.path.join(direct, subj_id, f'{subj_id}_EM_{trial_type}_{eye}.csv')
                emdf.to_csv(output_path, index=False, encoding='utf-8')


def _create_emdf(reg_df: pd.DataFrame, fix_df_temp: pd.DataFrame, subj_id: str) -> pd.DataFrame:
    """Create and initialize the eye-movement data frame."""
    emdf = pd.DataFrame(np.zeros((len(reg_df), 36)))
    emdf.columns = [
        'subj', 'trial_id', 'trial_type', 'trialstart', 'trialend', 'tdur',
        'recstart', 'recend', 'blinks', 'eye', 'tffixos', 'tffixurt',
        'tfixcnt', 'tregrcnt', 'region', 'reglen', 'word', 'line_no',
        'x1_pos', 'x2_pos', 'mod_x1', 'mod_x2', 'fpurt', 'fpcount',
        'fpregres', 'fpregreg', 'fpregchr', 'ffos', 'ffixurt', 'spilover',
        'rpurt', 'rpcount', 'rpregreg', 'rpregchr', 'spurt', 'spcount'
    ]
    
    # Copy trial information
    emdf['subj'] = subj_id
    emdf['trial_id'] = fix_df_temp.trial_id.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['trial_type'] = fix_df_temp.trial_type.iloc[0] if len(fix_df_temp) > 0 else ''
    emdf['trialstart'] = fix_df_temp.trialstart.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['trialend'] = fix_df_temp.trialend.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['tdur'] = fix_df_temp.tdur.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['recstart'] = fix_df_temp.recstart.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['recend'] = fix_df_temp.recend.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['blinks'] = fix_df_temp.blinks.iloc[0] if len(fix_df_temp) > 0 else np.nan
    emdf['eye'] = fix_df_temp.eye.iloc[0] if len(fix_df_temp) > 0 else ''
    
    # Copy region information
    emdf['region'] = reg_df.WordID
    emdf['reglen'] = reg_df.length
    emdf['word'] = reg_df.Word
    emdf['line_no'] = reg_df.line_no
    emdf['x1_pos'] = reg_df.x1_pos
    emdf['x2_pos'] = reg_df.x2_pos
    
    return emdf


def cal_write_em_b(
    direct: str,
    regfile_name_list: List[str],
    add_char_sp: int = 1
) -> None:
    """
    Batch calculate eye-movement measures for all subjects.
    
    Parameters
    ----------
    direct : str
        Directory containing all CSV files.
    regfile_name_list : list
        List of region file names.
    add_char_sp : int, optional
        Number of character spaces to add. Default is 1.
    
    Output
    ------
    Creates CSV files for each subject's each trial.
    """
    sac_file_exist, sac_file_dic = _crt_csv_dic(1, direct, '', '_Sac')
    fix_file_exist, fix_file_dic = _crt_csv_dic(1, direct, '', '_Fix')
    regfile_exist, regfile_dic = _crt_region_dic(direct, regfile_name_list)
    
    if sac_file_exist and fix_file_exist and regfile_exist:
        for subj_id in sac_file_dic.keys():
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
    
    print(f'SubjID: {subj_id}')
    merge_dfs = []
    
    emdf = pd.read_csv(stamp_file_dic[subj_id], sep=',', encoding='utf-8')
    emdf['gaze_time'] = emdf.time - emdf.recstart
    emdf['audio_time'] = np.nan
    emdf['audio_label'] = np.nan
    emdf['audio_region_no'] = np.nan
    
    trial_list = list(np.unique(emdf.trial_type))
    
    for trial in trial_list:
        print(f'Processing Trial: {trial}')
        
        # Get region file
        reg_df = pd.read_csv(regfile_dic[trial + '.region.csv'], encoding='utf-8')
        
        # Get ETime file
        aufile = os.path.join(direct, subj_id, f'{subj_id}-{trial}_ETime.csv')
        if not os.path.isfile(aufile):
            print(f'{aufile} does not exist!')
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
                if cur_region < 37 and cur_region in list(reg_df.WordID):
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
# Backward Compatibility Aliases
# =============================================================================

# Helper functions (internal)
_getRegDF = _get_reg_df
_crtCSV_dic = _crt_csv_dic
_crtRegion_dic = _crt_region_dic
_modEM = _mod_em
_chk_fp_fix_legacy = _chk_fp_fix
_chk_fp_reg_legacy = _chk_fp_reg
_getReg = _get_reg
_chk_rp_reg_legacy = _chk_rp_reg
_chk_sp_fix_legacy = _chk_sp_fix
_chk_tffixos_legacy = _chk_tffixos
_chk_tregrcnt_legacy = _chk_tregrcnt
_cal_EM = _cal_em

# User functions
cal_write_EM = cal_write_em
cal_write_EM_b = cal_write_em_b
mergeCSV = merge_csv
mergeCSV_b = merge_csv_b


# =============================================================================
# Module Information
# =============================================================================

__all__ = [
    # New names (snake_case)
    'cal_write_em',
    'cal_write_em_b',
    'merge_csv',
    'merge_csv_b',
    # Legacy names (backward compatibility)
    'cal_write_EM',
    'cal_write_EM_b',
    'mergeCSV',
    'mergeCSV_b',
]
