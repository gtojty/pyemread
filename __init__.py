# -*- coding: utf-8 -*-
"""
Pyemread: A Python package for multi-line text reading eye movement experiments

This package provides comprehensive tools for:
1. Generating bitmaps and region files for single/multi-line text reading experiments
2. Extracting and classifying saccades/fixations from eye-tracker data
3. Calculating eye-movement metrics used in reading research
4. Visualizing and animating eye-movement data

Version: 2.1.1
Authors: Tao Gong, David Braze
License: MIT

Modules:
    gen - Generate bitmaps, region files, and visualizations
    ext - Extract and classify eye-movement data from ASCII files
    cal - Calculate eye-movement measures

Example usage:
    import pyemread as pr
    
    # Generate bitmap and region file
    pr.gen.praster(direct, fontpath, st_pos, lang_type, text=text_lines)
    
    # Extract saccades and fixations from ASCII file
    sac_df, fix_df = pr.ext.read_srrasc(direct, subj_id, 'RAN')
    
    # Calculate eye-movement measures
    pr.cal.cal_write_em(direct, subj_id, regfile_list)
"""

__all__ = ["gen", "ext", "cal"]
__version__ = "2.1.1"
__author__ = "Tao Gong, David Braze"
__email__ = "gtojty@gmail.com"
__license__ = "MIT"

from . import gen
from . import ext
from . import cal
