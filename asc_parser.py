"""Minimal ASC-file parser used for Figure 5 regeneration on REAL data.

Extracts per-trial EFIX and ESACC events from an EyeLink ASC file,
returning two pandas DataFrames matching the schema used by pyemread's
cross-line classifier.  This is a stripped-down replacement for
ext.read_SRRasc that runs on Python 3 without the legacy encoding
hacks.
"""
from __future__ import annotations
import re
import pandas as pd


def parse_asc(path: str):
    """Parse an EyeLink .asc file into (SacDF, FixDF).

    Returned DataFrames have columns:
      FixDF : subj, trial_id, eye, start, end, dur, x, y, valid, line_no
      SacDF : subj, trial_id, eye, start, end, dur, x1, y1, x2, y2, ampl
    """
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    trial_id = -1
    in_trial = False
    fix_rows = []
    sac_rows = []
    subj = path.split('/')[-1].split('.')[0]

    for ln in lines:
        if ln.startswith('MSG') and 'TRIALID' in ln:
            m = re.search(r'TRIALID\s+(\d+)', ln)
            if m:
                trial_id = int(m.group(1))
                continue
        if ln.startswith('START'):
            in_trial = True
            continue
        if ln.startswith('END'):
            in_trial = False
            continue
        if not in_trial:
            continue

        if ln.startswith('EFIX'):
            # "EFIX L   853040 853340 304 217.2 64.5 755"
            parts = ln.split()
            # parts[0]='EFIX', parts[1]='L'/'R', start, end, dur, x, y, pupil
            try:
                eye = parts[1]
                start = float(parts[2])
                end = float(parts[3])
                dur = float(parts[4])
                x = float(parts[5])
                y = float(parts[6])
            except (ValueError, IndexError):
                continue
            fix_rows.append({
                'subj': subj, 'trial_id': trial_id, 'eye': eye,
                'start': start, 'end': end, 'dur': dur,
                'x': x, 'y': y, 'valid': 1, 'line_no': -1,
            })
        elif ln.startswith('ESACC'):
            parts = ln.split()
            # "ESACC R  853344 853364 24 227.4 68.8 169.2 46.7 1.50 146"
            try:
                eye = parts[1]
                start = float(parts[2])
                end = float(parts[3])
                dur = float(parts[4])
                x1 = float(parts[5])
                y1 = float(parts[6])
                x2 = float(parts[7])
                y2 = float(parts[8])
                ampl = float(parts[9])
            except (ValueError, IndexError):
                continue
            sac_rows.append({
                'subj': subj, 'trial_id': trial_id, 'eye': eye,
                'start': start, 'end': end, 'dur': dur,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'ampl': ampl,
            })

    FixDF = pd.DataFrame(fix_rows)
    SacDF = pd.DataFrame(sac_rows)
    return SacDF, FixDF


if __name__ == '__main__':
    import sys
    SacDF, FixDF = parse_asc(sys.argv[1])
    print(f'Fixations: {len(FixDF)}  |  Saccades: {len(SacDF)}')
    print(f'Trials: {sorted(FixDF["trial_id"].unique())}')
    print(f'Eyes: {sorted(FixDF["eye"].unique())}')
    for tr in sorted(FixDF['trial_id'].unique()):
        for eye in sorted(FixDF['eye'].unique()):
            n = ((FixDF['trial_id'] == tr) & (FixDF['eye'] == eye)).sum()
            print(f'  trial {tr} {eye}: {n} fixations')
