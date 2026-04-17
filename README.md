# pyemread

**A Python package for multi-line text reading eye-movement experiments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)](https://github.com/gtojty/pyemread/releases)

`pyemread` automates the design and analysis of multi-line text reading eye-movement experiments. It generates pixel-accurate stimulus bitmaps with matching word-level regions of interest, extracts saccades and fixations from **EyeLink ASC files or from raw `[t, x, y]` samples produced by any other tracker**, classifies events into text lines using a temporally-informed heuristic, and computes 18 per-word eye-movement metrics.

---

## What's new in v2.1

- **Tracker-agnostic raw-gaze input.** A new `ext_generic` module reads raw `[t, x, y]` samples from any eye tracker and performs event detection with I-VT (velocity-threshold) or I-DT (dispersion-threshold) algorithms. Format-specific loaders are provided for Tobii Pro Lab, Pupil Labs, SMI RED/iView, and WebGazer.js.
- **Parameter tuning and edge-case handling.** A new `ext_robust` module supplies automatic parameter recommendation, pre-scan detection, orphan-fixation reclassification, and parameter-sensitivity reports.
- **Cross-platform animations.** MP4 video generation via `matplotlib.animation` + FFmpeg replaces the Windows-only `turtle` + `winsound` backend used before v2.0.

See the [CHANGELOG](#changelog) for details.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Package structure](#package-structure)
4. [Typical workflow](#typical-workflow)
5. [Supported languages](#supported-languages)
6. [Supported trackers](#supported-trackers)
7. [Eye-movement metrics](#eye-movement-metrics)
8. [Parameter tuning](#parameter-tuning-and-edge-cases)
9. [Examples](#examples)
10. [Citation](#citation)
11. [Changelog](#changelog)
12. [License](#license)

---

## Installation

### From PyPI

```bash
pip install pyemread
```

### From source

```bash
git clone https://github.com/gtojty/pyemread.git
cd pyemread
pip install -e .
```

### Dependencies

`pyemread` requires Python 3.8 or later and the following packages:

| Package      | Purpose                                           |
|--------------|---------------------------------------------------|
| `numpy`      | Numerical computing                               |
| `pandas`     | DataFrame I/O and manipulation                    |
| `Pillow`     | Stimulus bitmap generation                        |
| `matplotlib` | Static visualisations and MP4 animation frames    |
| `moviepy`    | Audio-synchronised video generation               |

For MP4 video output, FFmpeg must be installed and available on the system path. On Ubuntu:

```bash
sudo apt install ffmpeg
```

On macOS with Homebrew:

```bash
brew install ffmpeg
```

Liberation Mono (or any other monospaced font) and Noto Sans CJK are recommended for alphabetic and East Asian stimuli, respectively.

---

## Quick start

```python
from pyemread import gen, ext, cal

# 1. Generate a bitmap and region file from plain text
gen.praster(
    direct='./output',
    fontpath='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    st_pos='TopLeft',
    lang_type='English',
    text=['Alex and Tim are lost.',
          'They went hiking today.'],
    ID='story01',
)

# 2. Extract saccades and fixations from an EyeLink ASC file
SacDF, FixDF = ext.read_SRRasc('./data', '1950138', 'RP')

# 3. Classify cross-line saccades and fixations
ext.cal_write_SacFix_crlSacFix(
    './data', '1950138',
    ['story01.region.csv'],
    'RP',
)

# 4. Compute 18 per-word eye-movement metrics
cal.cal_write_EM('./data', '1950138', ['story01.region.csv'])
```

For non-EyeLink data, use the new `ext_generic` module:

```python
from pyemread import ext_generic as eg, ext

# Load raw samples from any tracker
samples = eg.tobii_to_raw('P01_tobii.tsv', eye='right')

# Detect fixations and saccades
SacDF, FixDF = eg.detect_events(
    samples, method='IVT',
    velocity_threshold=30.0,     # deg/s
    px_per_deg=30,
    min_fix_duration=50,
)

# Hand off to the standard pyemread pipeline
eg.write_events('./out', 'P01', SacDF, FixDF)
```

---

## Package structure

```
pyemread/
├── __init__.py            # Package init (v2.1.0)
├── gen.py                 # Stimulus bitmap and region-file generation,
│                          # static visualisations, and MP4 animations.
├── ext.py                 # SR-Research EyeLink ASC parser and
│                          # cross-line classifier.
├── ext_generic.py         # NEW in v2.1.  Raw-gaze [t, x, y] input,
│                          # I-VT / I-DT event detection, and loaders
│                          # for Tobii, SMI, Pupil Labs, WebGazer.
├── ext_robust.py          # NEW in v2.1.  Parameter tuning, pre-scan
│                          # detection, orphan reclassification, and
│                          # sensitivity reports.
└── cal.py                 # Computation of 18 per-word metrics
                           # (first-pass, regression-path, second-pass).
```

### Module summary

| Module         | Role                                   | Status              |
|----------------|----------------------------------------|---------------------|
| `gen`          | Stimulus generation and visualisation  | Stable              |
| `ext`          | EyeLink ASC parsing                    | Stable              |
| `ext_generic`  | Tracker-agnostic raw-gaze input        | **New in v2.1**     |
| `ext_robust`   | Parameter tuning and edge-case helpers | **New in v2.1**     |
| `cal`          | Metric computation                     | Stable              |

---

## Typical workflow

The figure below (see `figures/fig1_architecture.png` in the repository) summarises the data flow. Input comes from one of three sources — plain-text stimuli for bitmap generation, EyeLink ASC files for legacy pipelines, or raw `[t, x, y]` samples for any other tracker. Events then pass through the same cross-line classifier and metric computation, producing CSV-based outputs that slot directly into pandas, R, or statistical packages.

```
         plain text           EyeLink ASC         raw [t, x, y]
             │                     │                    │
             ▼                     ▼                    ▼
          gen.py               ext.py            ext_generic.py
             │                     │                    │
             │                     └────────┬───────────┘
             │                              ▼
             │                       ext.cal_crlSacFix
             │                    (+ ext_robust helpers)
             │                              │
             ▼                              ▼
   bitmap + region file          classified events CSVs
                                            │
                                            ▼
                                         cal.py
                                            │
                                            ▼
                                  per-word metrics CSV
                                  MP4 animations (optional)
```

---

## Supported languages

`gen.praster` generates bitmaps and region files in:

| Family            | Languages                                                    |
|-------------------|--------------------------------------------------------------|
| Alphabetic        | English, German, French, Dutch, Italian, Spanish, Greek, … |
| East Asian        | Chinese, Japanese, Korean                                   |

**Input conventions:**

- **Alphabetic languages:** whitespace separates words. Leading and trailing punctuation is attached to the neighbouring word.
- **Chinese:** insert a vertical bar (`|`) between words in the input text (e.g. `"我|喜欢|读书"`); the bars are stripped from the rendered bitmap but used to populate the region file.
- **Japanese and Korean:** inter-word spaces are optional. If present, each space-separated token becomes an AOI; if absent, each character becomes an AOI.

Rendered multilingual examples are available in `examples/textRasters/`.

---

## Supported trackers

| Tracker                     | Entry point                        | Notes                                     |
|-----------------------------|------------------------------------|-------------------------------------------|
| SR-Research EyeLink         | `ext.read_SRRasc`                  | ASC files from `edf2asc`                  |
| Tobii Pro Lab               | `ext_generic.tobii_to_raw`         | TSV exports                               |
| Pupil Labs                  | `ext_generic.pupil_labs_to_raw`    | `gaze_positions.csv` + screen resolution  |
| SMI RED / iView             | `ext_generic.smi_to_raw`           | Sample reports                            |
| WebGazer.js                 | `ext_generic.webgazer_to_raw`      | JSON-lines logs                           |
| **Any other tracker**       | `ext_generic.read_raw_csv`         | Generic CSV/TSV with user-specified columns |

The four format-specific loaders return the same DataFrame schema, after which event detection and line classification proceed identically regardless of the source tracker.

### Event detection algorithms

```python
SacDF, FixDF = ext_generic.detect_events(
    samples,
    method='IVT',                  # or 'IDT'
    velocity_threshold=30.0,       # deg/s (with px_per_deg) or px/s
    dispersion_threshold=35.0,     # pixels (for I-DT)
    min_fix_duration=50.0,         # ms
    min_sac_duration=10.0,         # ms
    smooth_window=3,               # samples of moving-average smoothing
    px_per_deg=None,               # optional deg→pixel conversion
)
```

Both algorithms follow the formulations of Salvucci and Goldberg (2000).

---

## Eye-movement metrics

`cal.cal_write_EM` computes 18 per-word metrics grouped by processing stage:

### Whole-text metrics

| Name       | Definition                                                    |
|------------|---------------------------------------------------------------|
| `tffixos`  | Character offset of each word's first-pass fixation from the start of the first sentence |
| `tffixurt` | Total duration of each word's first-pass fixation             |
| `tffixcnt` | Total count of valid fixations in the trial                   |
| `tregrcnt` | Total count of regressive saccades in the trial               |

### First-pass fixation metrics

| Name        | Definition                                                    |
|-------------|---------------------------------------------------------------|
| `fpurt`     | First-pass reading time (sum of first-pass fixation durations)|
| `fpcount`   | Number of first-pass fixations                                |
| `fpregres`  | 1 if a regression starts from this word, 0 otherwise          |
| `fpregreg`  | Target word-region index of the first-pass regression         |
| `fpregchr`  | Character offset of the first-pass regression target          |
| `ffos`      | Character offset of the first fixation from the region's start|
| `ffixurt`   | Duration of the first first-pass fixation                     |
| `spilover`  | Duration of the first fixation that falls beyond the region   |

### Regression-path metrics

| Name        | Definition                                                    |
|-------------|---------------------------------------------------------------|
| `rpurt`     | Sum of fixation durations in the regression path              |
| `rpcount`   | Number of fixations in the regression path                    |
| `rpregreg`  | Smallest word-region index visited inside the regression path |
| `rpregchr`  | Character offset inside that smallest region                  |

### Second-pass fixation metrics

| Name       | Definition                                                    |
|------------|---------------------------------------------------------------|
| `spurt`    | Sum of fixation durations on the word after the regression path closes |
| `spcount`  | Number of second-pass fixations                               |

See the accompanying manuscript (or `Pyemread_SoftwareX_Supplement.docx`) for formal definitions with examples.

---

## Parameter tuning and edge cases

The cross-line classifier is controlled by two parameters: `diff_ratio` (fraction of a line's horizontal extent that between-fixation displacements must exceed to count as cross-line) and `frontrange_ratio` (fraction of the line treated as its "front"). Defaults of `0.6` and `0.2` reproduce the values used by Gong and Shuai (2023), but `ext_robust` lets you tune them from your own data:

```python
from pyemread import ext_robust as er
import pandas as pd

reg = pd.read_csv('story01.region.csv')
FixDF = pd.read_csv('./data/P01_Fix.csv')

# 1.  Data-driven recommendation
rec = er.recommend_diff_ratio(reg, FixDF)
print(rec)
# {'diff_ratio': 0.61, 'frontrange_ratio': 0.27, 'line_extent_px': 667, ...}

# 2.  Parameter-sensitivity report (Cohen's kappa vs default)
def classify(diff_ratio, frontrange_ratio):
    _, _, _, crlFix = ext.cal_crlSacFix(
        direct='./data', subjID='P01',
        regfileNameList=['story01.region.csv'],
        ExpType='RP',
        diff_ratio=diff_ratio,
        frontrange_ratio=frontrange_ratio,
    )
    return FixDF.merge(crlFix, on='start', how='left')['line_no']

report = er.sensitivity_report(classify)
print(report.head())

# 3.  Pre-scan detection (readers who scan the whole passage before reading)
mask = er.detect_prescan(FixDF, reg)
FixDF_clean = er.mask_prescan_fixations(FixDF, reg)

# 4.  Orphan-fixation reclassification (y-coordinate > one line-space off)
FixDF_fixed = er.reclassify_orphans(FixDF_clean, reg)
```

See Figure 5 of the accompanying manuscript for an example sensitivity heatmap.

---

## Examples

The `examples/` directory contains:

| Path                            | Contents                                                    |
|---------------------------------|-------------------------------------------------------------|
| `examples/oralReading/`         | Three English passages (story01–story03) with matching EyeLink ASC files and oral-reading WAV recordings for subjects 1950138, 1950149, and 1950168. |
| `examples/textRasters/`         | Rendered bitmaps and region files for English, Chinese, Japanese, and Korean passages. |
| `examples/example_workflow.py`  | Complete end-to-end demonstration of the pipeline.         |
| `figures/`                      | Six figures used in the accompanying manuscript.           |

To run the worked example:

```bash
cd examples
python example_workflow.py
```

---

## Citation

If you use `pyemread` in your research, please cite:

```bibtex
@article{gong2026pyemread,
  title   = {pyemread: a Python package for multi-line text reading
             eye-movement experiments with tracker-agnostic gaze input},
  author  = {Gong, Tao},
  journal = {SoftwareX},
  year    = {2026},
  note    = {In review},
  url     = {https://github.com/gtojty/pyemread},
}
```

### Related publications

- Gong, T., & Shuai, L. (2023). Segmented relations between online reading behaviors, text properties, and reader–text interactions: An eye-movement experiment. *Frontiers in Psychology, 13*, 1006662.

---

## Changelog

### v2.1.0 (April 2026)

**New**

- `ext_generic` module: tracker-agnostic raw-gaze input with I-VT and I-DT event detection; format loaders for Tobii Pro Lab, Pupil Labs, SMI, and WebGazer.js.
- `ext_robust` module: `recommend_diff_ratio`, `detect_prescan`, `mask_prescan_fixations`, `reclassify_orphans`, and `sensitivity_report` helpers.
- Documentation for parameter tuning and edge-case handling.

**Changed**

- `__init__.py` now uses graceful fallback imports so lightweight installations without FFmpeg or ASC-parser dependencies still load `ext_generic` and `ext_robust`.

**Unchanged**

- `gen.py`, `ext.py`, and `cal.py` are unchanged from v2.0. Existing v2.0 user code continues to work without modification.

### v2.0.0 (2024)

- Python 3.8+ modernisation (removed `reload(sys)`, `setdefaultencoding`, Python 2-only idioms).
- FFmpeg-based MP4 animation replacing Windows-only `turtle` + `winsound`.
- PEP 8 function naming (snake_case) with CamelCase aliases preserved for backward compatibility.
- Validated event classification (96.2 % forward, 93.8 % backward; see Gong and Shuai 2023).

### v0.1.0 (2017)

- Initial release (Python 2.7).

---

## License

`pyemread` is released under the [MIT License](LICENSE). Example data in `examples/oralReading/` are included with permission from D. Braze (Haskins Laboratories).

---

## Contact

**Tao Gong** — <gtojty@gmail.com>
School of Foreign Languages, Zhejiang University of Finance and Economics, Hangzhou, China.

Bug reports and feature requests: <https://github.com/gtojty/pyemread/issues>
