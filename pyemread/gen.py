# -*- coding: utf-8 -*-
"""
Stimulus generation and visualisation for reading experiments.

The main function, :func:`praster`, renders single- or multi-line text as a
bitmap (PNG) and writes a region file (``<ID>.region.csv``) that gives, for
every word, its line number and bounding box. These are the word-level areas
of interest used by the rest of the package. Alphabetic scripts (words
separated by spaces) and Chinese, Japanese and Korean (see
:func:`praster` for the input conventions) are supported.

The module also draws fixations and saccades on stimulus bitmaps
(:func:`draw_sac_fix`) and renders MP4 animations of the fixation sequence,
optionally synchronised with the audio recorded during oral reading
(:func:`animate`, which uses FFmpeg).

Examples
--------
>>> from pyemread import gen
>>> gen.praster(direct="out", fontpath="fonts/LiberationMono-Regular.ttf",
...             st_pos="TopLeft", lang_type="English",
...             text=["The quick brown fox", "jumps over the lazy dog."],
...             ID="demo")                                  # doctest: +SKIP
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
import csv
import time
import warnings
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import codecs

import shutil
import subprocess

_log = logging.getLogger(__name__)

# FFmpeg is needed only for animations; it is looked up when first used.
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None


# Global variables for language types
ENG_LANG_LIST = ['English', 'French', 'German', 'Dutch', 'Spanish', 'Italian', 'Greek',
                 'Portuguese', 'Polish']
"""Languages whose words are separated by spaces."""
CHN_LANG_LIST = ['Chinese']
KJ_LANG_LIST = ['Korean', 'Japanese']
PUNC_LIST = ['，', '。', '、', '：', '？', '！']


class Dictlist(dict):
    """
    Dictionary subclass that stores multiple values per key as a list.
    """
    def __setitem__(self, key, value):
        if key not in self:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)


def save_dict(filename: str, dict_data: dict) -> None:
    """
    Save dictionary to CSV file.
    
    Args:
        filename: Output file path
        dict_data: Dictionary to save
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for key, val in dict_data.items():
            writer.writerow([key, val])


def read_dict(filename: str) -> Dictlist:
    """
    Read dictionary from CSV file.
    
    Args:
        filename: Input file path
        
    Returns:
        Dictlist containing the loaded data
    """
    dict_data = Dictlist()
    with open(filename, 'r', encoding='utf-8') as f:
        for key, val in csv.reader(f):
            dict_data[key] = eval(val)
    return dict_data


class FontDict(dict):
    """
    Build a list of installed fonts with metadata.
    
    This class creates a dictionary where keys are font family names and
    values are dictionaries mapping style names to font file paths.
    
    Example:
        fontdict['Arial']['Regular'] -> 'c:\\windows\\fonts\\arial.ttf'
    """
    
    def __init__(self):
        """Initialize FontDict by scanning system fonts."""
        dict.__init__(self)
        fontpathlist = font_manager.findSystemFonts()
        fontpathlist.sort()
        
        for fp in fontpathlist:
            try:
                fi = ImageFont.truetype(fp, 12)
                family = re.sub(r'[ \-._]', '', fi.getname()[0])
                style = fi.getname()[1]
                
                if family not in self:
                    self[family] = {}
                self[family][style] = fp
            except (OSError, IOError):
                # Skip fonts that can't be loaded
                continue
    
    def families(self) -> List[str]:
        """Return sorted list of font families."""
        return sorted(self.keys())
    
    def family_count(self) -> int:
        """Return number of font families."""
        return len(self)
    
    def family_get(self, family: str) -> Optional[Dict[str, str]]:
        """
        Get available styles for a font family.
        
        Args:
            family: Font family name
            
        Returns:
            Dictionary of {style: filepath} or None if family not found
        """
        return self.get(family)
    
    def font_get(self, family: str, style: str) -> Optional[str]:
        """
        Get font file path for a specific family and style.
        
        Args:
            family: Font family name
            style: Font style name
            
        Returns:
            Path to font file or None if not found
        """
        if family not in self:
            _log.info(f"Family '{family}' does not exist.")
            return None
        
        if style not in self[family]:
            _log.info(f"Family '{family}' does not include style '{style}'")
            return None
        
        return self[family][style]
    
    # Backward compatibility aliases
    familyN = family_count
    familyGet = family_get
    fontGet = font_get


# Helper functions
def _get_reg_df(regfile_dic: Dict[str, str], trial_type: str) -> pd.DataFrame:
    """
    Get the region file data frame based on trial type.
    
    Args:
        regfile_dic: Dictionary mapping region file names to paths
        trial_type: Current trial ID
        
    Returns:
        DataFrame containing region file data
        
    Raises:
        ValueError: If trial_type is not valid
    """
    regfile_name = f'{trial_type}.region.csv'
    if regfile_name not in regfile_dic:
        raise ValueError(f"Invalid trial_type: {trial_type}")
    return pd.read_csv(regfile_dic[regfile_name], sep=',')


def _crt_csv_dic(sit: int, direct: str, subj_id: str, csvfiletype: str) -> Tuple[bool, Dict[str, str]]:
    """
    Create dictionary for different types of CSV files.
    
    Args:
        sit: Situation - 0: subjID is given; 1: no subjID
        direct: Root directory
        subj_id: Subject ID (for sit=0)
        csvfiletype: Type of CSV file ("_Stamp", "_Sac", "_crlSac", "_Fix", "_crlFix")
        
    Returns:
        Tuple of (file_exists, file_dictionary)
    """
    csv_dic = {}
    target_end = f'{csvfiletype}.csv'
    
    if sit == 0:
        filename = os.path.join(direct, subj_id, f'{subj_id}{target_end}')
        if os.path.isfile(filename):
            csv_dic[subj_id] = filename
            return True, csv_dic
        else:
            _log.info(f'{subj_id}{csvfiletype}.csv does not exist!')
            return False, csv_dic
    else:
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith(target_end):
                    key = name.split(target_end)[0]
                    csv_dic[key] = os.path.join(direct, key, name)
        
        if not csv_dic:
            _log.info('No CSV files in subfolders!')
            return False, csv_dic
        return True, csv_dic


def _crt_region_dic(direct: str, regfile_namelist: List[str]) -> Tuple[bool, Dict[str, str]]:
    """
    Create region file dictionary.
    
    Args:
        direct: Root directory where region files are located
        regfile_namelist: List of region file names
        
    Returns:
        Tuple of (files_exist, file_dictionary)
    """
    regfile_dic = {}
    target_end = '.region.csv'
    
    if not regfile_namelist:
        # Automatically gather all region files
        for file in os.listdir(direct):
            if fnmatch.fnmatch(file, f'*{target_end}'):
                regfile_dic[str(file)] = os.path.join(direct, str(file))
        
        if not regfile_dic:
            _log.info(f'No region file exists in {direct}!')
            return False, regfile_dic
    else:
        # Check specific region files
        for regfile in regfile_namelist:
            regfile_path = os.path.join(direct, regfile)
            if os.path.isfile(regfile_path):
                regfile_dic[regfile] = regfile_path
            else:
                _log.info(f'{regfile} does not exist!')
                return False, regfile_dic
    
    return True, regfile_dic


def _input_dict(res_dict: Dictlist, cur_key: int, name: str, lang_type: str,
                word: str, length: int, height: int, baseline: float,
                curline: int, x1_pos: float, y1_pos: float, x2_pos: float,
                y2_pos: float, b_x1: float, b_y1: float, b_x2: float,
                b_y2: float) -> Dictlist:
    """Write result of each word into result dictionary."""
    res_dict[cur_key] = name
    res_dict[cur_key] = lang_type
    res_dict[cur_key] = word
    res_dict[cur_key] = length
    res_dict[cur_key] = height
    res_dict[cur_key] = baseline
    res_dict[cur_key] = curline
    res_dict[cur_key] = x1_pos
    res_dict[cur_key] = y1_pos
    res_dict[cur_key] = x2_pos
    res_dict[cur_key] = y2_pos
    res_dict[cur_key] = b_x1
    res_dict[cur_key] = b_y1
    res_dict[cur_key] = b_x2
    res_dict[cur_key] = b_y2
    return res_dict


def _write_csv(reg_file: str, res_dict: Dictlist, code_method: str = 'utf-8') -> None:
    """
    Write result dictionary to CSV file.
    
    Output columns: Name, Language, WordID, Word, length, height, baseline,
    line_no, x1_pos, y1_pos, x2_pos, y2_pos, b_x1, b_y1, b_x2, b_y2
    """
    col = ['Name', 'Language', 'WordID', 'Word', 'length', 'height',
           'baseline', 'line_no', 'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos',
           'b_x1', 'b_y1', 'b_x2', 'b_y2']
    
    # Build data as list of dicts to avoid dtype issues
    data = []
    for key in res_dict.keys():
        row = {
            'Name': res_dict[key][0],
            'Language': res_dict[key][1],
            'WordID': key,
            'Word': res_dict[key][2],
            'length': int(res_dict[key][3]),
            'height': int(res_dict[key][4]),
            'baseline': int(res_dict[key][5]),
            'line_no': int(res_dict[key][6]),
            'x1_pos': int(res_dict[key][7]),
            'y1_pos': int(res_dict[key][8]),
            'x2_pos': int(res_dict[key][9]),
            'y2_pos': int(res_dict[key][10]),
            'b_x1': int(res_dict[key][11]),
            'b_y1': int(res_dict[key][12]),
            'b_x2': int(res_dict[key][13]),
            'b_y2': int(res_dict[key][14])
        }
        data.append(row)
    
    df = pd.DataFrame(data, columns=col)
    df = df.sort_values(by='WordID')
    df.to_csv(reg_file, index=False, encoding=code_method)


def _get_strike_ascents(font_filename: str, size: int) -> Dict[str, int]:
    """
    Build and return a dictionary of ascents (in pixels) for a font/size.
    
    Groups characters by their typical ascent relative to baseline character 'o':
    - group1: bdfhijkl|()\"'
    - group2: t
    - group3: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    - group4: 0123456789
    - group5: ?!
    - group6: %&@$
    - group7-9: Extended Latin characters
    """
    ttf = ImageFont.truetype(font_filename, size)
    
    # Get baseline height using 'o'
    bbox_o = ttf.getbbox('o')
    ht_o = bbox_o[3] - bbox_o[1]
    
    def get_height(char):
        bbox = ttf.getbbox(char)
        return bbox[3] - bbox[1]
    
    return {
        'bdfhijkl|()\"\'': get_height('b') - ht_o,
        't': get_height('t') - ht_o,
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ': get_height('A') - ht_o,
        '0123456789': get_height('0') - ht_o,
        '?!': get_height('?') - ht_o,
        '%&@$': get_height('&') - ht_o,
        'àáâãäåèéêëìíîïñòóôõöùúûüāăćĉċčēĕėěĩīĭńňōŏőŕřśšũūŭůűŵźżž': get_height('à') - ht_o,
        'ÀÁÂÃÄÅÆÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝĀĂĆĈĊČĎĐĒĔĖĚĜĞĠĤĥĴĹĺłŃŇŊŌŎŐŒŔŘŚŜŠŤŦŨŪŬŮŰŴŶŸŹŻŽƁſ': get_height('À') - ht_o,
        'ßÞƀƄƚɔɫɬʖʕʔʡʢ': get_height('ß') - ht_o
    }


def _get_strike_centers(font_filename: str, size: int) -> int:
    """Get height of characters with no descents or ascents like 'acemnorsuvwxz'."""
    ttf = ImageFont.truetype(font_filename, size)
    bbox = ttf.getbbox('o')
    return bbox[3] - bbox[1]


def _get_strike_descents(font_filename: str, size: int) -> Dict[str, int]:
    """
    Build and return a dictionary of descents (in pixels) for a font/size.
    
    Groups characters by their typical descent below baseline.
    """
    ttf = ImageFont.truetype(font_filename, size)
    
    def get_height(char):
        bbox = ttf.getbbox(char)
        return bbox[3] - bbox[1]
    
    ht_o = get_height('o')
    ht_g = get_height('g')
    ht_i = get_height('i')
    ht_j = get_height('j')
    ht_O = get_height('O')
    ht_Q = get_height('Q')
    ht_dot = get_height('.')
    ht_comma = get_height(',')
    ht_colon = get_height(':')
    ht_semicolon = get_height(';')
    ht_l = get_height('l')
    ht_or = get_height('|')
    ht_scbelow = get_height('ç')
    ht_yhead = get_height('ý')
    ht_cCbelow = get_height('Ç')
    
    return {
        'gpqy': ht_o - ht_g,
        'j': ht_i - ht_j,
        'Q@&$': ht_O - ht_Q,
        ',': ht_dot - ht_comma,
        ';': ht_colon - ht_semicolon,
        '|()_': ht_l - ht_or,
        'çęŋųƁƹȈȷȿɿʅ': ht_o - ht_scbelow,
        'ýÿğġģįĵţÿğġģįĵţ': ht_l - ht_yhead,
        'ÇĄĢĘĮķļļŅŖŞŢŲ': ht_O - ht_cCbelow
    }


def _get_key_val(char: str, desasc_dict: Dict[str, int]) -> int:
    """Get the value of the key that contains char."""
    for key in desasc_dict:
        if char in key:
            return desasc_dict[key]
    return 0


def _get_desasc(word: str, descents: Dict[str, int], ascents: Dict[str, int]) -> Tuple[int, int]:
    """Calculate minimum descent and maximum ascent for a word."""
    mdes, masc = 0, 0
    for c in word:
        mdes = min(mdes, _get_key_val(c, descents))
        masc = max(masc, _get_key_val(c, ascents))
    return (mdes, masc)


def _c_aspect(imgfont: ImageFont.FreeTypeFont, char: str) -> float:
    """Determine the height to width ratio for a character in the specified font."""
    bbox = imgfont.getbbox(char)
    wd = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]
    return float(ht) / wd if wd > 0 else 1.0


def praster(direct: str, fontpath: str, st_pos: str, lang_type: str,
            code_method: str = 'utf-8',
            text: List[str] = ['The quick brown fox jumps over the lazy dog.',
                              'The lazy tabby cat sleeps in the sun all afternoon.'],
            dim: Tuple[int, int] = (1280, 1024),
            fg: Tuple[int, int, int] = (0, 0, 0),
            bg: Tuple[int, int, int] = (232, 232, 232),
            wfont: Optional[str] = None,
            regfile: bool = True,
            lmargin: int = 215,
            tmargin: int = 86,
            linespace: int = 65,
            fht: int = 18,
            fwd: Optional[int] = None,
            bbox: bool = False,
            bbox_big: bool = False,
            ID: str = 'test',
            addspace: int = 18,
            log: bool = False) -> None:
    """Render a text passage as a stimulus bitmap and write its region file.

    Generates a bitmap image of single- or multi-line text and, optionally,
    a CSV region file with the bounding box and line number of every word
    (the word-level areas of interest used by the rest of the pipeline).

    Parameters
    ----------
    direct : str
        Output directory (created if needed).
    fontpath : str
        Path to a TrueType/OpenType font file.
    st_pos : {'TopLeft', 'Center', 'Auto'}
        Anchor of the text on the screen.
    lang_type : str
        Language: one of ``ENG_LANG_LIST`` (space-separated scripts),
        ``'Chinese'`` (word boundaries marked with ``|``), ``'Japanese'`` or
        ``'Korean'``.
    code_method : str, default 'utf-8'
        Text encoding.
    text : list of str
        Lines of the passage, exactly as they should appear on screen.
    dim : tuple of int, default (1280, 1024)
        Bitmap size in pixels (x, y).
    fg, bg : tuple of int
        RGB font and background colors.
    wfont : str, optional
        Font for a watermark.
    regfile : bool, default True
        Whether to write the region file.
    lmargin, tmargin : int
        Left and top margins in pixels.
    linespace : int, default 65
        Line spacing in pixels (baseline to baseline).
    fht : int, default 18
        Font height in pixels.
    fwd : int, optional
        Target character width in pixels (takes precedence over ``fht``).
    bbox, bbox_big : bool, default False
        Draw a box around each word, or around each line.
    ID : str, default 'test'
        Stimulus identifier, used in the file names.
    addspace : int, default 18
        Extra pixels above and below each line in the word boxes.
    log : bool, default False
        Write intermediate results to a log file.

    Returns
    -------
    None
        Writes ``<direct>/<ID>.png`` and, if ``regfile`` is True,
        ``<direct>/<ID>.region.csv``.

    Raises
    ------
    ValueError
        If ``text`` is not a list or the language is not supported.

    Examples
    --------
    >>> from matplotlib import font_manager
    >>> font = font_manager.findfont("DejaVu Sans Mono")
    >>> praster("stimuli", font, "TopLeft", "English",
    ...         text=["The quick brown fox", "jumps over the lazy dog."],
    ...         ID="demo")  # doctest: +SKIP
    """
    if not isinstance(text, (list, tuple)):
        raise ValueError("text argument must be a list of text string(s), not a bare string!")
    # a byte-order mark (e.g. from a text file saved as 'UTF-8 with BOM') is not text
    text = [t.replace('\ufeff', '') for t in text]
    os.makedirs(direct, exist_ok=True)
    
    # Adjust font height if width is specified
    if fwd:
        ttf = ImageFont.truetype(fontpath, fht)
        std_char = 'n'
        casp = _c_aspect(ttf, std_char)
        fht = int(round(fwd * casp))
        ttf = ImageFont.truetype(fontpath, fht)
    else:
        ttf = ImageFont.truetype(fontpath, fht)
    
    # Initialize image
    img = Image.new('RGB', dim, bg)
    draw = ImageDraw.Draw(img)
    
    # Initialize region dictionary
    if regfile:
        res_dict = Dictlist()
        cur_key = 1
    
    # Calculate vertical starting position
    if st_pos == 'Center':
        if lang_type in CHN_LANG_LIST + KJ_LANG_LIST:
            vpos = dim[1] / 2.0 + fht / 2.0
        else:
            vpos = dim[1] / 2.0
    elif st_pos == 'TopLeft':
        if lang_type in CHN_LANG_LIST + KJ_LANG_LIST:
            vpos = tmargin + fht / 2.0
        else:
            vpos = tmargin
    elif st_pos == 'Auto':
        if lang_type in CHN_LANG_LIST + KJ_LANG_LIST:
            vpos = dim[1] / 2.0 - (len(text) - 1) / 2.0 * linespace + fht / 2.0
        else:
            vpos = dim[1] / 2.0 - (len(text) - 1) / 2.0 * linespace
    
    # Process based on language type
    if lang_type in ENG_LANG_LIST:
        descents = _get_strike_descents(fontpath, fht)
        ascents = _get_strike_ascents(fontpath, fht)
        
        if log:
            import json
            with codecs.open(os.path.join(direct, 'Praster.log'), 'w', encoding=code_method) as logfile:
                logfile.write('ascents\n')
                json.dump(ascents, logfile)
                logfile.write('\n')
                logfile.write('descents\n')
                json.dump(descents, logfile)
                logfile.write('\n')
        
        curline = 1
        for line in text:
            words = line.split(' ')
            words = [w for w in words if w]  # Remove empty strings
            words = [' ' + w if i > 0 else w for i, w in enumerate(words)]
            
            # Calculate line-level extents for expanded bounding box (bbox_big)
            line_max_bottom = 0
            line_max_top = float('inf')
            for w in words:
                bbox_w = ttf.getbbox(w)
                line_max_bottom = max(line_max_bottom, bbox_w[3])
                line_max_top = min(line_max_top, bbox_w[1])
            
            # Paint words
            xpos1 = lmargin
            for w in words:
                wordlen = len(w)
                bbox_w = ttf.getbbox(w)
                wd = bbox_w[2] - bbox_w[0]
                xpos2 = xpos1 + wd
                
                # Draw text at (xpos1, vpos)
                draw.text((xpos1, vpos), w, font=ttf, fill=fg)
                
                # Word-specific tight bounding box - each word has its own height
                box_y1 = vpos + bbox_w[1]   # Word's actual top
                box_y2 = vpos + bbox_w[3]   # Word's actual bottom
                box_x1 = xpos1 + bbox_w[0]
                box_x2 = xpos1 + bbox_w[2]
                
                # Height is word-specific
                ht = bbox_w[3] - bbox_w[1]
                
                if bbox:
                    draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], outline=fg)
                
                # Line-level expanded bounding box (uniform per line)
                box_y1_b = vpos + line_max_top - addspace
                box_y2_b = vpos + line_max_bottom + addspace
                if bbox_big:
                    draw.rectangle([(box_x1, box_y1_b), (box_x2, box_y2_b)], outline=fg)
                
                if regfile:
                    res_dict = _input_dict(res_dict, cur_key, ID, lang_type, w, wordlen, ht,
                                          vpos, curline, box_x1, box_y1, box_x2, box_y2,
                                          box_x1, box_y1_b, box_x2, box_y2_b)
                    cur_key += 1
                
                xpos1 = xpos2
            
            if vpos >= dim[1] + linespace:
                raise ValueError(f"{vpos} warning! {ID} has too many words! They cannot be shown within one screen!")
            
            vpos += linespace
            curline += 1
    
    elif lang_type in CHN_LANG_LIST:
        curline = 1
        for line in text:
            words = line.split('|')
            
            # Calculate line-level max height
            line_max_top = float('inf')
            line_max_bottom = 0
            for w in words:
                bbox_w = ttf.getbbox(w)
                line_max_top = min(line_max_top, bbox_w[1])
                line_max_bottom = max(line_max_bottom, bbox_w[3])
            line_height = line_max_bottom - line_max_top
            
            xpos1 = lmargin
            for w in words:
                wordlen = len(w)
                bbox_w = ttf.getbbox(w)
                wd = bbox_w[2] - bbox_w[0]
                ht = bbox_w[3] - bbox_w[1]
                xpos2 = xpos1 + wd
                
                # Draw text
                draw.text((xpos1, vpos), w, font=ttf, fill=fg)
                
                # Bounding box matches text rendering
                box_y1 = vpos + line_max_top
                box_y2 = vpos + line_max_bottom
                box_x1 = xpos1 + bbox_w[0]
                box_x2 = xpos1 + bbox_w[2]
                
                if bbox:
                    draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], outline=fg)
                
                box_y1_b = box_y1 - addspace
                box_y2_b = box_y2 + addspace
                if bbox_big:
                    draw.rectangle([(box_x1, box_y1_b), (box_x2, box_y2_b)], outline=fg)
                
                if regfile:
                    res_dict = _input_dict(res_dict, cur_key, ID, lang_type, w, wordlen, line_height,
                                          vpos, curline, box_x1, box_y1, box_x2, box_y2,
                                          box_x1, box_y1_b, box_x2, box_y2_b)
                    cur_key += 1
                
                xpos1 = xpos2
            
            if vpos >= dim[1] + linespace:
                raise ValueError(f"{vpos} warning! {ID} has too many words!")
            
            vpos += linespace
            curline += 1
    
    elif lang_type in KJ_LANG_LIST:
        curline = 1
        for line in text:
            if ' ' not in line.strip():
                # no inter-word spaces: each character is its own AOI, with
                # punctuation attached to the preceding character
                words = []
                for ch in line.strip():
                    if words and (ch in PUNC_LIST or ch in '」』）〉》・…!?.,:;)'):
                        words[-1] += ch
                    else:
                        words.append(ch)
            else:
                words = line.split(' ')
                words = [w for w in words if w]
                words = [' ' + w if i > 0 else w for i, w in enumerate(words)]
            
            # Remove space after punctuation
            for ind_w in range(1, len(words)):
                if words[ind_w - 1] and words[ind_w - 1][-1] in PUNC_LIST:
                    words[ind_w] = re.sub(r'^ ', '', words[ind_w])
            
            # Calculate line-level max height
            line_max_top = float('inf')
            line_max_bottom = 0
            for w in words:
                bbox_w = ttf.getbbox(w)
                line_max_top = min(line_max_top, bbox_w[1])
                line_max_bottom = max(line_max_bottom, bbox_w[3])
            line_height = line_max_bottom - line_max_top
            
            xpos1 = lmargin
            for w in words:
                wordlen = len(w)
                bbox_w = ttf.getbbox(w)
                wd = bbox_w[2] - bbox_w[0]
                ht = bbox_w[3] - bbox_w[1]
                xpos2 = xpos1 + wd
                
                # Draw text
                draw.text((xpos1, vpos), w, font=ttf, fill=fg)
                
                # Bounding box matches text rendering
                box_y1 = vpos + line_max_top
                box_y2 = vpos + line_max_bottom
                box_x1 = xpos1 + bbox_w[0]
                box_x2 = xpos1 + bbox_w[2]
                
                if bbox:
                    draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], outline=fg)
                
                box_y1_b = box_y1 - addspace
                box_y2_b = box_y2 + addspace
                if bbox_big:
                    draw.rectangle([(box_x1, box_y1_b), (box_x2, box_y2_b)], outline=fg)
                
                if regfile:
                    res_dict = _input_dict(res_dict, cur_key, ID, lang_type, w, wordlen, line_height,
                                          vpos, curline, box_x1, box_y1, box_x2, box_y2,
                                          box_x1, box_y1_b, box_x2, box_y2_b)
                    cur_key += 1
                
                xpos1 = xpos2
            
            if vpos >= dim[1] + linespace:
                raise ValueError(f"{vpos} warning! {ID} has too many words!")
            
            vpos += linespace
            curline += 1
    
    else:
        raise ValueError(f"Invalid langType {lang_type}!")
    
    # Save outputs
    if regfile:
        _write_csv(os.path.join(direct, f'{ID}.region.csv'), res_dict, code_method)
    
    img.save(os.path.join(direct, f'{ID}.png'), 'PNG')


def gen_bitmap_regfile(direct: str, font_name: str, st_pos: str, lang_type: str,
                       text_filenames: List[str], genmethod: int = 2,
                       code_method: str = 'utf-8',
                       dim: Tuple[int, int] = (1280, 1024),
                       fg: Tuple[int, int, int] = (0, 0, 0),
                       bg: Tuple[int, int, int] = (232, 232, 232),
                       lmargin: int = 215, tmargin: int = 86,
                       linespace: int = 65, fht: int = 18,
                       fwd: Optional[int] = None, bbox: bool = False,
                       bbox_big: bool = False, ID: str = 'story',
                       addspace: int = 18, log: bool = False) -> None:
    """
    Generate bitmaps (PNG) and region files from text files.
    
    Args:
        direct: Directory for text files
        font_name: Font name or path to font file
        st_pos: Starting position - 'TopLeft', 'Center', or 'Auto'
        lang_type: Language type
        text_filenames: List of text file names
        genmethod: Generation method (0: simple test, 1: single file, 2: multiple files)
        code_method: Encoding method
        dim: Bitmap dimensions
        fg: Foreground color
        bg: Background color
        lmargin: Left margin
        tmargin: Top margin
        linespace: Line spacing
        fht: Font height
        fwd: Target character width
        bbox: Draw word bounding boxes
        bbox_big: Draw line bounding boxes
        ID: Stimulus ID prefix
        addspace: Extra vertical spacing
        log: Enable logging
    """
    # Get font path
    if lang_type in ENG_LANG_LIST:
        fd = FontDict()
        fontpath = fd.font_get(font_name, 'Regular')
    else:
        fontpath = font_name
    
    if genmethod == 0:
        # Simple tests
        if lang_type in ENG_LANG_LIST:
            praster(direct, fontpath, st_pos, lang_type, fht=fht, bbox=True, log=True)
            praster(direct, fontpath, st_pos, lang_type,
                   text=['This is a test.', 'This is another.'], fht=fht)
            praster(direct, fontpath, st_pos, lang_type,
                   text=['This is a one-liner.'], fht=fht)
        elif lang_type in CHN_LANG_LIST:
            praster(direct, fontpath, st_pos, lang_type,
                   text=['我们|爱|你。', '为什么|不让|他|走？'], fht=fht)
        elif lang_type == 'Korean':
            praster(direct, fontpath, st_pos, lang_type,
                   text=['저는 7년 동안 한국에서 살았어요', '이름은 무엇입니까?'], fht=fht)
        elif lang_type == 'Japanese':
            praster(direct, fontpath, st_pos, lang_type,
                   text=['むかし、 むかし、 ある ところ に', 'おじいさん と おばあさん が いました。'], fht=fht)
        else:
            raise ValueError(f"Invalid langType {lang_type}!")
    
    elif genmethod == 1:
        # Read from single text file
        txtfile = text_filenames[0]
        real_txtfile = os.path.join(direct, txtfile)
        
        if not os.path.isfile(real_txtfile):
            _log.info(f'{txtfile} does not exist!')
            return
        
        with codecs.open(real_txtfile, mode='r', encoding=code_method) as infile:
            _log.info(f"Read text file: {infile.name}")
            lines = infile.readlines()
        
        # Remove BOM if present
        if lines and lines[0].startswith('\ufeff'):
            lines[0] = lines[0][1:]
        
        # Remove comments
        lines = [l for l in lines if not re.match('^#', l)]
        text_content = ''.join(lines)
        
        # Split into separate texts
        texts = re.split(r'\r?\n\r?\n', text_content)
        texts[-1] = re.sub(r'\r?\n$', '', texts[-1])
        texts = [re.split(r'\r?\n', t) for t in texts]
        
        for i, P in enumerate(texts):
            _log.info(f"storyID = {i+1:02d} line = {len(P)}")
            praster(direct, fontpath, st_pos, lang_type,
                   code_method=code_method, text=P, dim=dim, fg=fg, bg=bg,
                   lmargin=lmargin, tmargin=tmargin, linespace=linespace,
                   fht=fht, fwd=fwd, bbox=bbox, bbox_big=bbox_big,
                   ID=f'{ID}{i+1:02d}', addspace=addspace, log=log)
    
    elif genmethod == 2:
        # Read from multiple text files
        if not text_filenames:
            text_filenames = [f for f in os.listdir(direct) if fnmatch.fnmatch(f, '*.txt')]
        else:
            # Verify files exist
            text_filenames = [f for f in text_filenames if os.path.isfile(os.path.join(direct, f))]
        
        for txtfile in text_filenames:
            ID_current = txtfile.split('.')[0]
            real_txtfile = os.path.join(direct, txtfile)
            
            with codecs.open(real_txtfile, mode='r', encoding=code_method) as infile:
                _log.info(f"Read text file: {infile.name}")
                lines = infile.readlines()
            
            # Remove BOM if present
            if lines and lines[0].startswith('\ufeff'):
                lines[0] = lines[0][1:]
            
            # Remove comments and trailing newlines
            lines = [l for l in lines if not re.match('^#', l)]
            lines = [re.sub(r'\r?\n$', '', l) for l in lines]
            
            praster(direct, fontpath, st_pos, lang_type,
                   code_method=code_method, text=lines, dim=dim, fg=fg, bg=bg,
                   lmargin=lmargin, tmargin=tmargin, linespace=linespace,
                   fht=fht, fwd=fwd, bbox=bbox, bbox_big=bbox_big,
                   ID=ID_current, addspace=addspace, log=log)


def upd_reg(direct: str, regfile_namelist: List[str], addspace: int) -> None:
    """
    Convert region files written by pyemread < 2.0 to the current format.
    
    Args:
        direct: Directory containing region files
        regfile_namelist: List of region file names
        addspace: Added space for bigger boundary
    """
    for trial_id, regfile in enumerate(regfile_namelist):
        reg_df = pd.read_csv(os.path.join(direct, regfile), sep=',', header=None)
        reg_df.columns = ['Name', 'Word', 'length', 'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos']
        
        # Add WordID
        reg_df['WordID'] = range(1, len(reg_df) + 1)
        
        # Add line_no
        reg_df['line_no'] = 0
        line_ind = 1
        line_reg_low = 0
        cur_y = reg_df.y2_pos[0]
        
        for cur_ind in range(len(reg_df)):
            if reg_df.y2_pos[cur_ind] >= cur_y + 20:
                line_reg_up = cur_ind
                reg_df.loc[line_reg_low:line_reg_up-1, 'line_no'] = line_ind
                line_ind += 1
                line_reg_low = line_reg_up
                cur_y = reg_df.y2_pos[cur_ind]
        
        if line_reg_up < len(reg_df):
            reg_df.loc[line_reg_low:, 'line_no'] = line_ind
        
        # Add baseline
        reg_df['baseline'] = 0
        for line_num in reg_df.line_no.unique():
            mask = reg_df.line_no == line_num
            reg_df.loc[mask, 'baseline'] = reg_df.loc[mask, 'y2_pos'].min()
        
        # Add height
        reg_df['height'] = reg_df.y2_pos - reg_df.y1_pos
        
        # Add boundary columns
        reg_df['b_x1'] = reg_df.x1_pos
        reg_df['b_y1'] = 0
        reg_df['b_x2'] = reg_df.x2_pos
        reg_df['b_y2'] = 0
        
        for line_num in reg_df.line_no.unique():
            mask = reg_df.line_no == line_num
            reg_df.loc[mask, 'b_y1'] = reg_df.loc[mask, 'y1_pos'].max() - addspace
            reg_df.loc[mask, 'b_y2'] = reg_df.loc[mask, 'y2_pos'].min() + addspace
        
        # Reorder columns
        reg_df = reg_df[['Name', 'WordID', 'Word', 'length', 'height', 'baseline',
                        'line_no', 'x1_pos', 'y1_pos', 'x2_pos', 'y2_pos',
                        'b_x1', 'b_y1', 'b_x2', 'b_y2']]
        
        reg_df.to_csv(os.path.join(direct, regfile), index=False)


def _rgb2gray(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to grayscale string."""
    return str(0.2989 * rgb[0] / 256.0 + 0.5870 * rgb[1] / 256.0 + 0.1140 * rgb[2] / 256.0)


def _image_sac_fix(direct: str, subj_id: str, bitmap_namelist: List[str],
                   sac_df: pd.DataFrame, crl_sac_df: pd.DataFrame,
                   fix_df: pd.DataFrame, crl_fix_df: pd.DataFrame,
                   reg_df: pd.DataFrame, trial_type: str,
                   draw_type: str, max_fix_radius: int,
                   draw_final: bool, show_fix_dur: bool, png_opt: int) -> None:
    """
    Draw saccade and fixation data of a trial.
    
    Args:
        direct: Directory to store drawn figures
        subj_id: Subject ID
        bitmap_namelist: List of bitmap files as backgrounds
        sac_df: Saccade data of the trial
        crl_sac_df: Cross-line saccade data
        fix_df: Fixation data of the trial
        crl_fix_df: Cross-line fixation data
        reg_df: Region file data
        trial_type: Trial type (e.g., 'story01') - used for bitmap matching and output naming
        draw_type: 'ALL', 'SAC', or 'FIX'
        max_fix_radius: Maximum radius of fixation circles
        draw_final: Whether to draw fixations after reading ends
        show_fix_dur: Whether to show fixation duration numbers
        png_opt: 0: use existing PNG, 1: draw texts from region file
    """
    # Prepare fonts
    fd = FontDict()
    fontpath = fd.font_get('LiberationMono', 'Regular')
    if fontpath is None:
        # Try fallback fonts
        for font_family in ['DejaVuSansMono', 'Courier', 'FreeMono']:
            fontpath = fd.font_get(font_family, 'Regular')
            if fontpath:
                break
        if fontpath is None:
            fontpath = font_manager.findSystemFonts()[0]
    
    xsz = 18
    ttf = ImageFont.truetype(fontpath, xsz)
    
    if png_opt == 0:
        # Find bitmap matching trial_type (e.g., 'story01' -> 'story01.png')
        bitmap_file = f"{trial_type}.png"
        if bitmap_file not in bitmap_namelist:
            # Try to find a matching bitmap in the list
            matching = [b for b in bitmap_namelist if trial_type in b]
            if matching:
                bitmap_file = matching[0]
            else:
                _log.info(f"Warning: No bitmap found for trial_type '{trial_type}'")
                return
        
        # Load and convert to RGBA for transparency support
        img1_base = Image.open(os.path.join(direct, bitmap_file)).convert('RGBA')
        img2_base = Image.open(os.path.join(direct, bitmap_file)).convert('RGBA')
        # Create overlay layers for transparent drawing
        img1_overlay = Image.new('RGBA', img1_base.size, (255, 255, 255, 0))
        img2_overlay = Image.new('RGBA', img2_base.size, (255, 255, 255, 0))
        draw1 = ImageDraw.Draw(img1_overlay)
        draw2 = ImageDraw.Draw(img2_overlay)
        draw1_base = ImageDraw.Draw(img1_base)
        draw2_base = ImageDraw.Draw(img2_base)
    else:
        descents = _get_strike_descents(fontpath, xsz)
        ascents = _get_strike_ascents(fontpath, xsz)
        fg = (0, 0, 0)
        bg = (232, 232, 232, 255)
        dim = (1280, 1024)
        
        # Create RGBA images for transparency support
        img1_base = Image.new('RGBA', dim, bg)
        img2_base = Image.new('RGBA', dim, bg)
        img1_overlay = Image.new('RGBA', dim, (255, 255, 255, 0))
        img2_overlay = Image.new('RGBA', dim, (255, 255, 255, 0))
        draw1 = ImageDraw.Draw(img1_overlay)
        draw2 = ImageDraw.Draw(img2_overlay)
        draw1_base = ImageDraw.Draw(img1_base)
        draw2_base = ImageDraw.Draw(img2_base)
        
        # Draw texts on base images
        for curline in reg_df.line_no.unique():
            line_data = reg_df[reg_df.line_no == curline].reset_index(drop=True)
            for ind in range(len(line_data)):
                mdes, masc = _get_desasc(line_data.Word[ind], descents, ascents)
                vpos_text = line_data.y1_pos[ind] + masc - xsz / 4.5 - 1
                draw1_base.text((line_data.x1_pos[ind], vpos_text), line_data.Word[ind], font=ttf, fill=fg)
                draw2_base.text((line_data.x1_pos[ind], vpos_text), line_data.Word[ind], font=ttf, fill=fg)
    
    # Determine ending fixation
    single_eye = len(fix_df.eye.unique()) == 1
    
    if single_eye:
        end_fix = len(fix_df)
        end_time = fix_df.loc[end_fix - 1, 'end_time'] if end_fix > 0 else 0
        if not draw_final:
            for ind in range(end_fix):
                if fix_df.valid.iloc[ind] == 'yes' and pd.isna(fix_df.line_no.iloc[ind]):
                    end_fix = ind
                    break
            end_time = fix_df.loc[end_fix - 1, 'end_time'] if end_fix > 0 else 0
    else:
        fix_left = fix_df[fix_df.eye == 'L']
        fix_right = fix_df[fix_df.eye == 'R']
        end_fix_L = len(fix_left)
        end_fix_R = len(fix_right)
        end_time_L = fix_left.iloc[-1]['end_time'] if len(fix_left) > 0 else 0
        end_time_R = fix_right.iloc[-1]['end_time'] if len(fix_right) > 0 else 0
        
        if not draw_final:
            for ind in range(len(fix_left)):
                if fix_left.iloc[ind]['valid'] == 'yes' and pd.isna(fix_left.iloc[ind]['line_no']):
                    end_fix_L = ind
                    break
            end_time_L = fix_left.iloc[end_fix_L - 1]['end_time'] if end_fix_L > 0 else 0
            
            for ind in range(len(fix_right)):
                if fix_right.iloc[ind]['valid'] == 'yes' and pd.isna(fix_right.iloc[ind]['line_no']):
                    end_fix_R = ind
                    break
            end_time_R = fix_right.iloc[end_fix_R - 1]['end_time'] if end_fix_R > 0 else 0
    
    # Colors - transparent red for left eye, transparent green for right eye
    col_left_sac = 'blue'
    col_right_sac = 'red'
    # RGBA colors with transparency (alpha=128 for 50% transparency)
    col_left_eye_fix = (255, 0, 0, 128)  # Transparent red for left eye
    col_right_eye_fix = (0, 255, 0, 128)  # Transparent green for right eye
    col_left_eye_outline = (255, 0, 0, 255)  # Solid red outline for left eye
    col_right_eye_outline = (0, 255, 0, 255)  # Solid green outline for right eye
    col_num = 'blue'
    
    # Get radius ratio
    max_dur = fix_df.duration.max() if len(fix_df) > 0 else 1
    radius_ratio = max_fix_radius / max_dur if max_dur > 0 else 1
    
    # Draw based on type
    if draw_type == 'ALL':
        if single_eye:
            # Draw fixations on overlay (transparent)
            for ind in range(min(end_fix, len(fix_df))):
                if fix_df.valid.iloc[ind] == 'yes':
                    r = fix_df.duration.iloc[ind] * radius_ratio
                    x, y = fix_df.x_pos.iloc[ind], fix_df.y_pos.iloc[ind]
                    fill_col = col_left_eye_fix if fix_df.eye.iloc[ind] == 'L' else col_right_eye_fix
                    outline_col = col_left_eye_outline if fix_df.eye.iloc[ind] == 'L' else col_right_eye_outline
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=outline_col, fill=fill_col)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_df.duration.iloc[ind])), font=ttf, fill=col_num)
            
            # Draw saccades on base image
            for ind in range(len(sac_df)):
                if sac_df.loc[ind, 'end_time'] <= end_time:
                    x1, y1 = sac_df.x1_pos.iloc[ind], sac_df.y1_pos.iloc[ind]
                    x2, y2 = sac_df.x2_pos.iloc[ind], sac_df.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
        else:
            # Binocular data - draw both eyes
            # Transparent red for left eye, transparent green for right eye
            fix_left = fix_df[fix_df.eye == 'L'].reset_index(drop=True)
            fix_right = fix_df[fix_df.eye == 'R'].reset_index(drop=True)
            sac_left = sac_df[sac_df.eye == 'L'].reset_index(drop=True)
            sac_right = sac_df[sac_df.eye == 'R'].reset_index(drop=True)
            
            # Draw right eye fixations as transparent green circles
            for ind in range(min(end_fix_R, len(fix_right))):
                if fix_right.valid.iloc[ind] == 'yes':
                    r = fix_right.duration.iloc[ind] * radius_ratio
                    x, y = fix_right.x_pos.iloc[ind], fix_right.y_pos.iloc[ind]
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=col_right_eye_outline, fill=col_right_eye_fix)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_right.duration.iloc[ind])), font=ttf, fill=col_num)
            
            # Draw left eye fixations as transparent red circles
            for ind in range(min(end_fix_L, len(fix_left))):
                if fix_left.valid.iloc[ind] == 'yes':
                    r = fix_left.duration.iloc[ind] * radius_ratio
                    x, y = fix_left.x_pos.iloc[ind], fix_left.y_pos.iloc[ind]
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=col_left_eye_outline, fill=col_left_eye_fix)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_left.duration.iloc[ind])), font=ttf, fill=col_num)
            
            # Draw left eye saccades on base
            for ind in range(len(sac_left)):
                if sac_left.iloc[ind]['end_time'] <= end_time_L:
                    x1, y1 = sac_left.x1_pos.iloc[ind], sac_left.y1_pos.iloc[ind]
                    x2, y2 = sac_left.x2_pos.iloc[ind], sac_left.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
            
            # Draw right eye saccades on base
            for ind in range(len(sac_right)):
                if sac_right.iloc[ind]['end_time'] <= end_time_R:
                    x1, y1 = sac_right.x1_pos.iloc[ind], sac_right.y1_pos.iloc[ind]
                    x2, y2 = sac_right.x2_pos.iloc[ind], sac_right.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
        
        # Draw cross-line fixations on img2 overlay (transparent)
        for ind in range(len(crl_fix_df)):
            r = crl_fix_df.duration.iloc[ind] * radius_ratio
            x, y = crl_fix_df.x_pos.iloc[ind], crl_fix_df.y_pos.iloc[ind]
            fill_col = col_left_eye_fix if crl_fix_df.eye.iloc[ind] == 'L' else col_right_eye_fix
            outline_col = col_left_eye_outline if crl_fix_df.eye.iloc[ind] == 'L' else col_right_eye_outline
            draw2.ellipse((x-r, y-r, x+r, y+r), outline=outline_col, fill=fill_col)
            if show_fix_dur:
                draw2.text((x, y), str(int(crl_fix_df.duration.iloc[ind])), font=ttf, fill=col_num)
        
        # Draw cross-line saccades on img2 base
        for ind in range(len(crl_sac_df)):
            x1, y1 = crl_sac_df.x1_pos.iloc[ind], crl_sac_df.y1_pos.iloc[ind]
            x2, y2 = crl_sac_df.x2_pos.iloc[ind], crl_sac_df.y2_pos.iloc[ind]
            col = col_right_sac if x1 < x2 else col_left_sac
            draw2_base.line((x1, y1, x2, y2), fill=col, width=2)
        
        # Composite overlays onto base images and save
        os.makedirs(os.path.join(direct, subj_id), exist_ok=True)
        img1_final = Image.alpha_composite(img1_base, img1_overlay)
        img2_final = Image.alpha_composite(img2_base, img2_overlay)
        img1_final.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_FixSac_{trial_type}.png'), 'PNG')
        img2_final.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_crlFixSac_{trial_type}.png'), 'PNG')
    
    elif draw_type == 'SAC':
        if single_eye:
            for ind in range(len(sac_df)):
                if sac_df.loc[ind, 'end_time'] <= end_time:
                    x1, y1 = sac_df.x1_pos.iloc[ind], sac_df.y1_pos.iloc[ind]
                    x2, y2 = sac_df.x2_pos.iloc[ind], sac_df.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
        else:
            # Binocular data - draw both eyes
            sac_left = sac_df[sac_df.eye == 'L'].reset_index(drop=True)
            sac_right = sac_df[sac_df.eye == 'R'].reset_index(drop=True)
            
            # Draw left eye saccades
            for ind in range(len(sac_left)):
                if sac_left.iloc[ind]['end_time'] <= end_time_L:
                    x1, y1 = sac_left.x1_pos.iloc[ind], sac_left.y1_pos.iloc[ind]
                    x2, y2 = sac_left.x2_pos.iloc[ind], sac_left.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
            
            # Draw right eye saccades
            for ind in range(len(sac_right)):
                if sac_right.iloc[ind]['end_time'] <= end_time_R:
                    x1, y1 = sac_right.x1_pos.iloc[ind], sac_right.y1_pos.iloc[ind]
                    x2, y2 = sac_right.x2_pos.iloc[ind], sac_right.y2_pos.iloc[ind]
                    col = col_right_sac if x1 < x2 else col_left_sac
                    draw1_base.line((x1, y1, x2, y2), fill=col, width=2)
        
        for ind in range(len(crl_sac_df)):
            x1, y1 = crl_sac_df.x1_pos.iloc[ind], crl_sac_df.y1_pos.iloc[ind]
            x2, y2 = crl_sac_df.x2_pos.iloc[ind], crl_sac_df.y2_pos.iloc[ind]
            col = col_right_sac if x1 < x2 else col_left_sac
            draw2_base.line((x1, y1, x2, y2), fill=col, width=2)
        
        os.makedirs(os.path.join(direct, subj_id), exist_ok=True)
        img1_base.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_Sac_{trial_type}.png'), 'PNG')
        img2_base.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_crlSac_{trial_type}.png'), 'PNG')
    
    elif draw_type == 'FIX':
        if single_eye:
            for ind in range(min(end_fix, len(fix_df))):
                if fix_df.valid.iloc[ind] == 'yes':
                    r = fix_df.duration.iloc[ind] * radius_ratio
                    x, y = fix_df.x_pos.iloc[ind], fix_df.y_pos.iloc[ind]
                    fill_col = col_left_eye_fix if fix_df.eye.iloc[ind] == 'L' else col_right_eye_fix
                    outline_col = col_left_eye_outline if fix_df.eye.iloc[ind] == 'L' else col_right_eye_outline
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=outline_col, fill=fill_col)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_df.duration.iloc[ind])), font=ttf, fill=col_num)
        else:
            # Binocular data - draw both eyes
            fix_left = fix_df[fix_df.eye == 'L'].reset_index(drop=True)
            fix_right = fix_df[fix_df.eye == 'R'].reset_index(drop=True)
            
            # Draw left eye fixations (transparent red)
            for ind in range(min(end_fix_L, len(fix_left))):
                if fix_left.valid.iloc[ind] == 'yes':
                    r = fix_left.duration.iloc[ind] * radius_ratio
                    x, y = fix_left.x_pos.iloc[ind], fix_left.y_pos.iloc[ind]
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=col_left_eye_outline, fill=col_left_eye_fix)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_left.duration.iloc[ind])), font=ttf, fill=col_num)
            
            # Draw right eye fixations (transparent green)
            for ind in range(min(end_fix_R, len(fix_right))):
                if fix_right.valid.iloc[ind] == 'yes':
                    r = fix_right.duration.iloc[ind] * radius_ratio
                    x, y = fix_right.x_pos.iloc[ind], fix_right.y_pos.iloc[ind]
                    draw1.ellipse((x-r, y-r, x+r, y+r), outline=col_right_eye_outline, fill=col_right_eye_fix)
                    if show_fix_dur:
                        draw1.text((x, y), str(int(fix_right.duration.iloc[ind])), font=ttf, fill=col_num)
        
        for ind in range(len(crl_fix_df)):
            r = crl_fix_df.duration.iloc[ind] * radius_ratio
            x, y = crl_fix_df.x_pos.iloc[ind], crl_fix_df.y_pos.iloc[ind]
            fill_col = col_left_eye_fix if crl_fix_df.eye.iloc[ind] == 'L' else col_right_eye_fix
            outline_col = col_left_eye_outline if crl_fix_df.eye.iloc[ind] == 'L' else col_right_eye_outline
            draw2.ellipse((x-r, y-r, x+r, y+r), outline=outline_col, fill=fill_col)
            if show_fix_dur:
                draw2.text((x, y), str(int(crl_fix_df.duration.iloc[ind])), font=ttf, fill=col_num)
        
        os.makedirs(os.path.join(direct, subj_id), exist_ok=True)
        img1_final = Image.alpha_composite(img1_base, img1_overlay)
        img2_final = Image.alpha_composite(img2_base, img2_overlay)
        img1_final.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_Fix_{trial_type}.png'), 'PNG')
        img2_final.convert('RGB').save(os.path.join(direct, subj_id, f'{subj_id}_crlFix_{trial_type}.png'), 'PNG')


def draw_sac_fix(direct: str, subj_id: str, regfile_namelist: List[str],
                 bitmap_namelist: List[str], draw_type: str,
                 max_fix_radius: int = 30, draw_final: bool = False,
                 show_fix_dur: bool = False, png_opt: int = 0) -> None:
    """
    Read and draw saccade and fixation data.
    
    Args:
        direct: Directory storing CSV and region files
        subj_id: Subject ID
        regfile_namelist: List of region file names
        bitmap_namelist: List of PNG bitmap file names
        draw_type: 'ALL', 'SAC', or 'FIX'
        max_fix_radius: Maximum radius for fixation circles
        draw_final: Whether to draw fixations after reading ends
        show_fix_dur: Whether to show fixation durations
        png_opt: 0: use existing PNG, 1: draw from region file
    """
    # Check required files
    sac_exists, sac_dic = _crt_csv_dic(0, direct, subj_id, '_Sac')
    crl_sac_exists, crl_sac_dic = _crt_csv_dic(0, direct, subj_id, '_crlSac')
    fix_exists, fix_dic = _crt_csv_dic(0, direct, subj_id, '_Fix')
    crl_fix_exists, crl_fix_dic = _crt_csv_dic(0, direct, subj_id, '_crlFix')
    reg_exists, reg_dic = _crt_region_dic(direct, regfile_namelist)
    
    if png_opt == 0:
        if not bitmap_namelist:
            bitmap_namelist = [f for f in os.listdir(direct) if fnmatch.fnmatch(f, '*.png')]
        
        bitmap_exists = all(os.path.isfile(os.path.join(direct, b)) for b in bitmap_namelist)
    else:
        bitmap_exists = True
    
    if not all([sac_exists, fix_exists, crl_sac_exists, crl_fix_exists, reg_exists, bitmap_exists]):
        return
    
    # Read data
    sac_df = pd.read_csv(sac_dic[subj_id], sep=',')
    crl_sac_df = pd.read_csv(crl_sac_dic[subj_id], sep=',')
    fix_df = pd.read_csv(fix_dic[subj_id], sep=',')
    crl_fix_df = pd.read_csv(crl_fix_dic[subj_id], sep=',')
    
    # Draw for each trial (using actual trial IDs from data)
    for trial_id in sorted(sac_df.trial_id.unique()):
        trial_data = sac_df[sac_df.trial_id == trial_id]
        if len(trial_data) == 0:
            continue
        trial_type = trial_data.trial_type.unique()[0]
        reg_df = _get_reg_df(reg_dic, trial_type)
        
        _log.info(f"Draw Sac and Fix: Subj: {subj_id}, Trial {trial_id}: {trial_type}")
        
        sac = sac_df[sac_df.trial_id == trial_id].reset_index(drop=True)
        crl_sac = crl_sac_df[crl_sac_df.trial_id == trial_id].reset_index(drop=True)
        fix = fix_df[fix_df.trial_id == trial_id].reset_index(drop=True)
        crl_fix = crl_fix_df[crl_fix_df.trial_id == trial_id].reset_index(drop=True)
        
        _image_sac_fix(direct, subj_id, bitmap_namelist, sac, crl_sac, fix, crl_fix,
                      reg_df, trial_type, draw_type, max_fix_radius, draw_final,
                      show_fix_dur, png_opt)


def draw_sac_fix_b(direct: str, regfile_namelist: List[str],
                   bitmap_namelist: List[str], method: str,
                   max_fix_radius: int = 30, draw_final: bool = False,
                   show_num: bool = False, png_method: int = 0) -> None:
    """
    Batch drawing of all subjects' fixation and saccade data.
    
    Args:
        direct: Directory containing all CSV files
        regfile_namelist: List of region file names
        bitmap_namelist: List of PNG bitmap file names
        method: 'ALL', 'SAC', or 'FIX'
        max_fix_radius: Maximum radius for fixation circles
        draw_final: Whether to draw fixations after reading ends
        show_num: Whether to show fixation durations
        png_method: 0: use existing PNG, 1: draw from region file
    """
    # Find all subjects
    subj_list = set()
    for root, dirs, files in os.walk(direct):
        for name in files:
            if name.endswith('.asc'):
                subj_list.add(name.split('.')[0])
    
    if not subj_list:
        _log.info('No CSV files in the directory!')
        return
    
    reg_exists, reg_dic = _crt_region_dic(direct, regfile_namelist)
    
    if png_method == 0:
        if not bitmap_namelist:
            bitmap_namelist = [f for f in os.listdir(direct) if fnmatch.fnmatch(f, '*.png')]
        bitmap_exists = all(os.path.isfile(os.path.join(direct, b)) for b in bitmap_namelist)
    else:
        bitmap_exists = True
    
    if reg_exists and bitmap_exists:
        for subj_id in subj_list:
            draw_sac_fix(direct, subj_id, regfile_namelist, bitmap_namelist,
                        method, max_fix_radius, draw_final, show_num, png_method)


def draw_blinks(direct: str, trial_num: int) -> None:
    """
    Draw histogram of individual blinks.
    
    Args:
        direct: Directory storing CSV files
        trial_num: Number of trials in each subject's data
    """
    subj_list = [f.split('.')[0] for f in os.listdir(direct) if fnmatch.fnmatch(f, '*.asc')]
    
    for trial_id in range(trial_num):
        blinks_data = []
        for subj in subj_list:
            fix_df = pd.read_csv(os.path.join(direct, f'{subj}_Fix.csv'), sep=',')
            fix_temp = fix_df[fix_df.trial_id == trial_id].reset_index(drop=True)
            if len(fix_temp) > 0:
                blinks_data.append(fix_temp.blinks.iloc[0])
        
        fig, ax = plt.subplots()
        ax.hist(blinks_data, bins=20, density=True)
        ax.set_title(f'Histogram of Blinks (trial = {trial_id}; n= {len(blinks_data)})')
        ax.set_xlabel('No. Blinks')
        ax.set_ylabel('Frequency')
        plt.savefig(os.path.join(direct, f'Hist_blinks_trial{trial_id}.png'))
        plt.close()


# Line-assignment plots and animations
def draw_line_assignment(fix_df: pd.DataFrame, reg_df: pd.DataFrame,
                         methods: Tuple[str, ...] = ('slice', 'diff'),
                         out_file: Optional[str] = None, bitmap_file: Optional[str] = None,
                         title: str = '') -> 'plt.Figure':
    """
    Plot the line assignments of several methods side by side for one trial.

    Each panel shows the word boxes of the region file (or the stimulus
    bitmap), the scanpath, and the fixations colored by the line to which
    the method assigned them. This is the recommended way to inspect trials
    flagged by :func:`pyemread.ext_robust.trial_report`.

    Parameters
    ----------
    fix_df : pandas.DataFrame
        Fixations of one trial and one eye (``x_pos``, ``y_pos``).
    reg_df : pandas.DataFrame
        Region table of the stimulus.
    methods : tuple of str
        Methods of :func:`pyemread.lines.assign_lines` to compare.
    out_file : str, optional
        If given, the figure is saved to this file.
    bitmap_file : str, optional
        Stimulus bitmap drawn underneath the fixations.
    title : str
        Text prefixed to each panel title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from . import lines as _lines
    fix_df = fix_df.reset_index(drop=True)
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for ax, m in zip(axes[0], methods):
        res = _lines.assign_lines(fix_df, reg_df, method=m)
        if bitmap_file:
            ax.imshow(Image.open(bitmap_file).convert('L'), cmap='gray', alpha=0.5)
        for r in reg_df.itertuples():
            ax.add_patch(plt.Rectangle((r.x1_pos, r.y1_pos), r.x2_pos - r.x1_pos,
                                       r.y2_pos - r.y1_pos, fill=False, lw=0.4, ec='0.6'))
        ax.plot(res['x_pos'], res['y_pos'], lw=0.4, color='0.4')
        ln = res['line_no'].to_numpy(float)
        ok = np.isfinite(ln)
        ax.scatter(res['x_pos'][ok], res['y_pos'][ok], s=14,
                   c=[cmap(int(v - 1) % 10) for v in ln[ok]], zorder=3)
        ax.scatter(res['x_pos'][~ok], res['y_pos'][~ok], s=14, marker='x', color='k', zorder=3)
        x0, x1 = reg_df['x1_pos'].min() - 60, reg_df['x2_pos'].max() + 60
        y0, y1 = reg_df['y1_pos'].min() - 80, reg_df['y2_pos'].max() + 80
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect('equal')
        ax.set_title(f'{title}{m}'.strip())
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=150)
    return fig


def _get_audio_duration(audio_file: str) -> Optional[float]:
    """
    Get the duration of an audio file in seconds using ffprobe.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def create_animation_video(direct: str, subj_id: str, bitmap_file: str,
                          sound_file: Optional[str], fix_df: pd.DataFrame,
                          trial_id: int, output_file: str,
                          max_fix_radius: int = 30,
                          fps: int = 30,
                          eye: str = 'both') -> None:
    """
    Create an animation video of eye movements synchronized with audio.
    
    This function creates an MP4 video showing fixation movements overlaid on
    the reading text bitmap. When an audio file is provided, the video duration
    matches the audio duration and both start simultaneously. Fixation positions
    are shown relative to the recording start time.
    
    Uses efficient PIL-based frame rendering for faster video generation.
    
    Args:
        direct: Directory for output files
        subj_id: Subject ID
        bitmap_file: Path to background bitmap image (PNG format)
        sound_file: Path to audio file (WAV format, optional)
        fix_df: DataFrame containing fixation data with columns:
                - start_time, end_time, duration: timing in ms
                - x_pos, y_pos: fixation position in pixels
                - eye: 'L' or 'R' for left/right eye
                - recstart: recording start time (optional)
        trial_id: Trial ID for labeling
        output_file: Output video filename (should end in .mp4)
        max_fix_radius: Maximum radius for fixation circles in pixels
        fps: Frames per second for the video (default 30)
        eye: Which eye to display - 'L' for left only, 'R' for right only, 
             'both' for both eyes (default 'both')
        
    The output video will have:
        - Red circles for left eye fixations
        - Green circles for right eye fixations
        - Circle size proportional to fixation duration
        - Audio track synchronized with fixation timing
    """
    import tempfile
    import shutil
    
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("FFmpeg is required for video generation. Please install FFmpeg.")
    
    # Load background image as RGBA for transparency support
    base_img = Image.open(bitmap_file).convert('RGBA')
    
    # Check for valid fixation data
    if len(fix_df) == 0:
        _log.info("No fixation data available")
        return
    
    # Filter by eye if specified
    if eye.upper() in ['L', 'R']:
        eye_label = 'Left' if eye.upper() == 'L' else 'Right'
        fix_df = fix_df[fix_df.eye == eye.upper()].reset_index(drop=True)
        _log.info(f"Filtering for {eye_label} eye: {len(fix_df)} fixations")
        if len(fix_df) == 0:
            _log.info(f"No {eye_label} eye fixation data available")
            return
    elif eye.lower() != 'both':
        _log.info(f"Warning: Invalid eye parameter '{eye}', using 'both'")
    
    # Determine reference time (when recording/reading started)
    if 'recstart' in fix_df.columns and not pd.isna(fix_df.loc[fix_df.index[0], 'recstart']):
        ref_time = fix_df.loc[fix_df.index[0], 'recstart']
    else:
        ref_time = fix_df.start_time.min()
    
    # Calculate relative times for fixations
    fix_df = fix_df.copy()
    fix_df['rel_start'] = fix_df['start_time'] - ref_time
    fix_df['rel_end'] = fix_df['end_time'] - ref_time
    
    # Determine video duration from audio if available
    if sound_file and os.path.exists(sound_file):
        audio_duration = _get_audio_duration(sound_file)
        if audio_duration:
            duration_sec = audio_duration
            _log.info(f"Using audio duration: {duration_sec:.2f} seconds")
        else:
            duration_sec = (fix_df['rel_end'].max()) / 1000.0
            _log.info(f"Could not get audio duration, using fixation duration: {duration_sec:.2f} seconds")
    else:
        duration_sec = (fix_df['rel_end'].max()) / 1000.0
        _log.info(f"No audio file, using fixation duration: {duration_sec:.2f} seconds")
    
    # Calculate radius scaling
    max_dur = fix_df.duration.max()
    radius_ratio = max_fix_radius / max_dur if max_dur > 0 else 1
    
    # Calculate number of frames
    num_frames = int(duration_sec * fps)
    _log.info(f"Creating animation: {num_frames} frames at {fps} fps ({duration_sec:.2f} seconds)")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    # Define colors (RGBA) - Red for left eye, Green for right eye
    col_left = (255, 0, 0, 180)
    col_right = (0, 255, 0, 180)
    
    # Fixation fade state tracking
    fade_states = {i: 0 for i in range(len(fix_df))}
    
    # Generate frames
    _log.info(f"Rendering {num_frames} frames...")
    for frame_num in range(num_frames):
        if frame_num % 100 == 0:
            _log.info(f"  Frame {frame_num}/{num_frames}")
        
        # Current time in milliseconds
        current_time_ms = frame_num * (1000 / fps)
        
        # Create frame with overlay
        frame_img = base_img.copy()
        overlay = Image.new('RGBA', frame_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw fixations
        for i, (idx, row) in enumerate(fix_df.iterrows()):
            rel_start = row['rel_start']
            rel_end = row['rel_end']
            
            if rel_start <= current_time_ms <= rel_end:
                # Fixation is active
                r = row['duration'] * radius_ratio
                x, y = row['x_pos'], row['y_pos']
                color = col_left if row['eye'] == 'L' else col_right
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(0, 0, 0, 255))
                fade_states[i] = 180
            elif current_time_ms > rel_end and fade_states[i] > 0:
                # Fade out
                r = row['duration'] * radius_ratio
                x, y = row['x_pos'], row['y_pos']
                alpha = fade_states[i]
                color = (255, 0, 0, alpha) if row['eye'] == 'L' else (0, 255, 0, alpha)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(0, 0, 0, alpha))
                fade_states[i] = max(0, fade_states[i] - 15)
        
        # Composite and save frame
        frame_final = Image.alpha_composite(frame_img, overlay)
        frame_rgb = frame_final.convert('RGB')
        frame_path = os.path.join(temp_dir, f'frame_{frame_num:06d}.png')
        frame_rgb.save(frame_path, 'PNG')
    
    _log.info("Frames rendered.")
    
    # Determine output path
    if not output_file.endswith('.mp4'):
        output_file = output_file + '.mp4'
    out_dir = os.path.join(direct, subj_id) if os.path.isdir(os.path.join(direct, subj_id)) else direct
    video_path = os.path.join(out_dir, output_file)
    
    # Use ffmpeg to create video with audio
    _log.info(f"Encoding video with ffmpeg...")
    if sound_file and os.path.exists(sound_file):
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-i', sound_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            video_path
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    if result.returncode == 0:
        _log.info(f"Animation saved to: {video_path}")
    else:
        _log.info(f"FFmpeg error: {result.stderr}")


def animate(direct: str, subj_id: str, trial_id: int, output_format: str = 'mp4',
            eye: str = 'both', sound_file: Optional[str] = None,
            bitmap_file: Optional[str] = None) -> Optional[str]:
    """
    Render an MP4 animation of one trial's fixations, with audio if available.

    The fixations are read from ``<direct>/<subj_id>/<subj_id>_Fix.csv``. The
    stimulus bitmap is ``<direct>/<trial_type>.png`` and the audio recorded
    during the trial, if any, is ``<direct>/<subj_id>/<subj_id>-<trial_id>.wav``
    (the naming used by the example data). Both can be given explicitly.

    Parameters
    ----------
    direct : str
        Data directory.
    subj_id : str
        Participant identifier.
    trial_id : int
        Trial number as in the fixation table (1-based).
    output_format : str
        Container of the output file (default 'mp4').
    eye : {'both', 'L', 'R'}
        Eye(s) to draw (default 'both').
    sound_file, bitmap_file : str, optional
        Override the default audio and bitmap locations.

    Returns
    -------
    str or None
        Path of the written video, or None if the trial has no fixations.
    """
    subj_dir = os.path.join(direct, subj_id)
    fix_path = os.path.join(subj_dir, f'{subj_id}_Fix.csv')
    if not os.path.isfile(fix_path):
        raise FileNotFoundError(f"{fix_path} not found.")
    fix_df = pd.read_csv(fix_path, sep=',')
    fix_trial = fix_df[fix_df.trial_id == trial_id].reset_index(drop=True)
    if len(fix_trial) == 0:
        _log.warning(f'No fixations for trial {trial_id} of {subj_id}.')
        return None
    trial_type = str(fix_trial.trial_type.iloc[0])
    if bitmap_file is None:
        bitmap_file = os.path.join(direct, f'{trial_type}.png')
    if not os.path.isfile(bitmap_file):
        raise FileNotFoundError(f"Stimulus bitmap not found: {bitmap_file}")
    if sound_file is None:
        cand = os.path.join(subj_dir, f'{subj_id}-{trial_id}.wav')
        sound_file = cand if os.path.isfile(cand) else None
    suffix = f'_{eye.upper()}' if eye.upper() in ('L', 'R') else ''
    output_file = f'{subj_id}_trial{trial_id}_animation{suffix}.{output_format}'
    create_animation_video(direct, subj_id, bitmap_file, sound_file,
                           fix_trial, trial_id, output_file, eye=eye)
    return os.path.join(subj_dir, output_file)


def animate_by_eye(direct: str, subj_id: str, trial_id: int, 
                   output_format: str = 'mp4') -> None:
    """
    Create separate animation videos for left and right eyes.
    
    This function creates two video files:
    - {subj_id}_trial{trial_id}_animation_L.mp4 for left eye fixations
    - {subj_id}_trial{trial_id}_animation_R.mp4 for right eye fixations
    
    Args:
        direct: Directory of bitmaps, sound files, and fixation CSV files
        subj_id: Subject ID
        trial_id: Trial ID
        output_format: Output format ('mp4' recommended)
    """
    _log.info("="*70)
    _log.info(f"Creating Left Eye Animation")
    _log.info("="*70)
    animate(direct, subj_id, trial_id, output_format, eye='L')
    
    _log.info("\n" + "="*70)
    _log.info(f"Creating Right Eye Animation")
    _log.info("="*70)
    animate(direct, subj_id, trial_id, output_format, eye='R')


def animate_timestamp(direct: str, subj_id: str, trial_id: int,
                      output_format: str = 'mp4') -> None:
    """
    Create animation of a trial using time-stamped data.
    
    Args:
        direct: Directory containing files
        subj_id: Subject ID
        trial_id: Trial ID
        output_format: Output format ('mp4' recommended)
    """
    # Find stamp CSV file
    subj_dir = os.path.join(direct, subj_id)
    
    if not os.path.isdir(subj_dir):
        _log.info(f'Subject directory not found: {subj_dir}')
        return
    
    csv_list = [f for f in os.listdir(subj_dir) if fnmatch.fnmatch(f, '*_Stamp.csv')]
    
    if not csv_list:
        _log.info('No Stamp CSV files in the directory!')
        return
    
    csv_file = None
    for f in csv_list:
        if f.split('_Stamp')[0] == subj_id:
            csv_file = f
            break
    
    if csv_file is None:
        _log.info(f'Stamp data of {subj_id} is missing!')
        return
    
    # Find sound file (optional)
    sound_file = None
    
    cand = os.path.join(subj_dir, f'{subj_id}-{trial_id}.wav')
    if os.path.isfile(cand):
        sound_file = cand
    
    # Load timestamp data
    stamp_df = pd.read_csv(os.path.join(subj_dir, csv_file), sep=',')
    stamp_trial = stamp_df[stamp_df.trial_id == trial_id].reset_index(drop=True)
    
    # Convert timestamp data to fixation-like format for animation
    # Filter for valid positions
    valid_mask = ~stamp_trial.x_pos1.isna() & (stamp_trial.Fix_Sac == 'Fix')
    stamp_fix = stamp_trial[valid_mask].copy()
    
    if len(stamp_fix) == 0:
        _log.info('No valid fixation data in timestamp file')
        return
    
    # Rename columns to match fixation format
    stamp_fix = stamp_fix.rename(columns={
        'x_pos1': 'x_pos',
        'y_pos1': 'y_pos',
        'time': 'start_time'
    })
    
    # Estimate duration based on sampling frequency
    if 'sampfreq' in stamp_fix.columns:
        sample_interval = 1000 / stamp_fix.sampfreq.iloc[0]
    else:
        sample_interval = 4  # Default to 250Hz
    
    stamp_fix['duration'] = sample_interval
    stamp_fix['end_time'] = stamp_fix['start_time'] + stamp_fix['duration']
    stamp_fix['valid'] = 'yes'
    
    # Stimulus bitmap named after the trial type
    bitmap_file = os.path.join(direct, f'{stamp_trial.trial_type.iloc[0]}.png')
    if not os.path.isfile(bitmap_file):
        raise FileNotFoundError(f"Stimulus bitmap not found: {bitmap_file}")
    
    output_file = f'{subj_id}_trial{trial_id}_timestamp_animation.{output_format}'
    
    create_animation_video(direct, subj_id, bitmap_file, sound_file,
                          stamp_fix, trial_id, output_file, max_fix_radius=10)


def change_png2gif(direct: str) -> None:
    """
    Convert PNG bitmaps to GIF format (deprecated, kept for compatibility).
    
    Deprecated: animations are rendered from PNG bitmaps, so GIF conversion is
    not needed; the function is kept for compatibility.
    
    Args:
        direct: Directory containing PNG bitmaps
    """
    warnings.warn("change_png2gif is deprecated: animations are rendered from PNG bitmaps.",
                  DeprecationWarning)
    
    png_list = [f for f in os.listdir(direct) if fnmatch.fnmatch(f, '*.png')]
    
    if not png_list:
        _log.info('PNG bitmap is missing!')
        return
    
    for png_file in png_list:
        im = Image.open(os.path.join(direct, png_file))
        im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
        gif_path = os.path.join(direct, png_file.rsplit('.', 1)[0] + '.gif')
        im.save(gif_path)


# Backward compatibility aliases (CamelCase to snake_case)
Gen_Bitmap_RegFile = gen_bitmap_regfile
Gen_Bitmap_RegFile_b = gen_bitmap_regfile  # Batch functionality integrated
Praster = praster
changePNG2GIF = change_png2gif
animate_em = create_animation_video
animate_em_timestamp = create_animation_video
draw_SacFix = draw_sac_fix
draw_SacFix_b = draw_sac_fix_b
draw_blinks = draw_blinks
animate_TimeStamp = animate_timestamp
updReg = upd_reg


def animate_batch(direct: str, subj_ids: Optional[List[str]] = None,
                  output_format: str = 'mp4') -> Dict[str, List[str]]:
    """
    Generate animations for multiple subjects and trials.
    
    Args:
        direct: Directory containing data files
        subj_ids: List of subject IDs (if None, auto-detect from directory)
        output_format: Output format ('mp4' or 'gif')
    
    Returns:
        Dictionary mapping subject IDs to lists of generated animation files
    """
    results = {}
    
    # Auto-detect subjects if not provided
    if subj_ids is None:
        subj_ids = []
        for root, dirs, files in os.walk(direct):
            for name in files:
                if name.endswith('_Fix.csv'):
                    subj_id = name.split('_Fix.csv')[0]
                    if subj_id not in subj_ids:
                        subj_ids.append(subj_id)
    
    for subj_id in subj_ids:
        results[subj_id] = []
        fix_file = os.path.join(direct, subj_id, f"{subj_id}_Fix.csv")
        
        if not os.path.exists(fix_file):
            _log.info(f"Fix file not found for {subj_id}")
            continue
        
        try:
            fix_df = pd.read_csv(fix_file)
            trial_ids = fix_df['trial_id'].unique()
            
            for trial_id in trial_ids:
                try:
                    animate(direct, subj_id, int(trial_id), output_format)
                    output_file = os.path.join(direct, subj_id,
                                              f"{subj_id}_trial{trial_id}_animation.{output_format}")
                    if os.path.exists(output_file):
                        results[subj_id].append(output_file)
                except Exception as e:
                    _log.info(f"Error animating {subj_id} trial {trial_id}: {e}")
        except Exception as e:
            _log.info(f"Error processing {subj_id}: {e}")
    
    return results


# =============================================================================
# Backward-compatible wrapper functions (old API names)
# =============================================================================

def Praster(direct, fontpath, stPos, langType, codeMethod='utf-8',
            text=['The quick brown fox jumps over the lazy dog.',
                  'The lazy tabby cat sleeps in the sun all afternoon.'],
            dim=(1280, 1024), fg=(0, 0, 0), bg=(232, 232, 232), wfont=None,
            regfile=True, lmargin=215, tmargin=86, linespace=65,
            fht=18, fwd=None, bbox=False, bbox_big=False, ID='test',
            addspace=18, log=False):
    """
    Backward-compatible wrapper for praster().
    
    See praster() for full documentation.
    """
    return praster(direct=direct, fontpath=fontpath, st_pos=stPos,
                   lang_type=langType, code_method=codeMethod, text=text,
                   dim=dim, fg=fg, bg=bg, wfont=wfont, regfile=regfile,
                   lmargin=lmargin, tmargin=tmargin, linespace=linespace,
                   fht=fht, fwd=fwd, bbox=bbox, bbox_big=bbox_big, ID=ID,
                   addspace=addspace, log=log)


def Gen_Bitmap_RegFile(direct, fontName, stPos, langType, textFileNameList,
                       genmethod=2, codeMethod='utf-8', dim=(1280, 1024),
                       fg=(0, 0, 0), bg=(232, 232, 232), lmargin=215,
                       tmargin=86, linespace=65, fht=18, fwd=None,
                       bbox=False, bbox_big=False, ID='story', addspace=18, log=False):
    """
    Backward-compatible wrapper for gen_bitmap_regfile().
    
    See gen_bitmap_regfile() for full documentation.
    """
    return gen_bitmap_regfile(direct=direct, font_name=fontName, st_pos=stPos,
                              lang_type=langType, text_file_name_list=textFileNameList,
                              genmethod=genmethod, code_method=codeMethod, dim=dim,
                              fg=fg, bg=bg, lmargin=lmargin, tmargin=tmargin,
                              linespace=linespace, fht=fht, fwd=fwd, bbox=bbox,
                              bbox_big=bbox_big, ID=ID, addspace=addspace, log=log)


def Gen_Bitmap_RegFile_b(direct, fontName, stPos, langType, textFileNameList,
                         genmethod=2, codeMethod='utf-8', dim=(1280, 1024),
                         fg=(0, 0, 0), bg=(232, 232, 232), lmargin=215,
                         tmargin=86, linespace=65, fht=18, fwd=None,
                         bbox=False, bbox_big=False, ID='story', addspace=18, log=False):
    """
    Backward-compatible wrapper for gen_bitmap_regfile() - batch version.
    
    Same as Gen_Bitmap_RegFile.
    """
    return Gen_Bitmap_RegFile(direct, fontName, stPos, langType, textFileNameList,
                              genmethod, codeMethod, dim, fg, bg, lmargin, tmargin,
                              linespace, fht, fwd, bbox, bbox_big, ID, addspace, log)
