# pyemread

**A Python package for multi-line text reading eye-movement experiments — with a temporally-informed cross-line classifier and tracker-agnostic gaze input.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)](https://github.com/gtojty/pyemread/releases)

`pyemread` takes multi-line reading experiments end-to-end: it generates pixel-accurate stimulus bitmaps with matching word-level regions of interest, parses **EyeLink ASC files or raw `[t, x, y]` samples from any other tracker**, classifies events into text lines using a temporally-informed heuristic, and computes 18 per-word eye-movement metrics. The cross-line classifier follows the temporal sequence of fixations rather than fitting geometry to a fixation cloud, recovering 96.2 % of forward and 93.8 % of backward cross-line moves on hand-coded data (Gong & Shuai, 2023).

For the complete API listing — every function with parameters, return values, and examples — see **[API_REFERENCE.md](API_REFERENCE.md)**.

---

## What's new in v2.1

- **Tracker-agnostic raw-gaze input.** A new `ext_generic` module reads raw `[t, x, y]` samples from any eye tracker and performs event detection with I-VT (velocity-threshold) or I-DT (dispersion-threshold) algorithms. Format-specific loaders are provided for Tobii Pro Lab, Pupil Labs, SMI RED/iView, and WebGazer.js.
- **Parameter tuning and edge-case handling.** A new `ext_robust` module supplies automatic parameter recommendation, pre-scan detection, orphan-fixation reclassification, and parameter-sensitivity reports.
- **Real-data validated.** Sensitivity analysis on 1,195 real fixations (3 subjects × 2 passages from Gong & Shuai 2023) confirms the defaults are safe across an 8-percent parameter band; data-driven recommender returns `diff_ratio = 0.68` on this corpus.
- **Cross-platform animations.** MP4 video generation via `matplotlib.animation` + FFmpeg replaces the Windows-only `turtle` + `winsound` backend used before v2.0.

See the [Changelog](#changelog) for the full list.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Package structure](#package-structure)
4. [Typical workflow](#typical-workflow)
5. [Supported languages](#supported-languages)
6. [Supported trackers](#supported-trackers)
7. [Cross-line classifier and parameter tuning](#cross-line-classifier-and-parameter-tuning)
8. [Eye-movement metrics](#eye-movement-metrics)
9. [Examples](#examples)
10. [API reference](#api-reference)
11. [Citation](#citation)
12. [Changelog](#changelog)
13. [License and contact](#license-and-contact)

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

| Package      | Minimum version | Purpose                                           |
|--------------|-----------------|---------------------------------------------------|
| `numpy`      | 1.20            | Numerical computing                               |
| `pandas`     | 1.3             | DataFrame I/O and manipulation                    |
| `Pillow`     | 9.0             | Stimulus bitmap generation                        |
| `matplotlib` | 3.4             | Static visualisations and MP4 animation frames    |
| `moviepy`    | 1.0.3           | Audio-synchronised video generation (optional)    |

For MP4 video output, FFmpeg ≥ 4.0 must be installed and available on the system path:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg   # via Chocolatey
```

Liberation Mono (or any other monospaced font) and Noto Sans CJK are recommended for alphabetic and East Asian stimuli, respectively.

---

## Quick start

### EyeLink pipeline (the original v2.0 workflow)

```python
from pyemread import gen, ext, cal

# 1. Generate a bitmap and region file from plain text
gen.Praster(
    direct='./output',
    fontpath='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    stPos='TopLeft',
    langType='English',
    text=['Alex and Tim are lost.',
          'They went hiking today.'],
    ID='story01',
)

# 2. Parse the EyeLink ASC file
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

### Tracker-agnostic pipeline (new in v2.1)

```python
from pyemread import ext_generic as eg, ext

# 1. Load raw samples from any tracker
samples = eg.tobii_to_raw('P01_tobii.tsv', eye='right')
# Or for a generic CSV / TSV:
# samples = eg.read_raw_csv('P01.csv', time_col='t_ms',
#                           x_col='gaze_x', y_col='gaze_y',
#                           time_unit='ms')

# 2. Detect fixations and saccades from the samples
SacDF, FixDF = eg.detect_events(
    samples, method='IVT',
    velocity_threshold=30.0,     # deg/s
    px_per_deg=30,
    min_fix_duration=50,
)

# 3. Hand off to the standard pyemread pipeline
eg.write_events('./out', 'P01', SacDF, FixDF)
# From here ext.cal_crlSacFix and cal.cal_write_EM work
# exactly as for EyeLink data.
```

### Parameter tuning from the data itself

```python
from pyemread import ext_robust as er
import pandas as pd

reg = pd.read_csv('./output/story01.region.csv')
FixDF = pd.read_csv('./data/P01_Fix.csv')

# Data-driven parameter recommendation
rec = er.recommend_diff_ratio(reg, FixDF)
print(rec)
# {'diff_ratio': 0.68, 'frontrange_ratio': 0.25, ...}
```

---

## Package structure

```
pyemread/
├── __init__.py            # Package init (v2.1.0)
├── gen.py                 # Stimulus bitmap and region-file generation,
│                          # static visualisations, and MP4 animations
├── ext.py                 # SR-Research EyeLink ASC parser and
│                          # cross-line classifier
├── ext_generic.py         # NEW in v2.1.  Raw-gaze [t, x, y] input,
│                          # I-VT / I-DT event detection, and loaders
│                          # for Tobii, SMI, Pupil Labs, WebGazer
├── ext_robust.py          # NEW in v2.1.  Parameter tuning, pre-scan
│                          # detection, orphan reclassification, and
│                          # sensitivity reports
└── cal.py                 # Computation of 18 per-word metrics
                           # (first-pass, regression-path, second-pass)
```

### Module summary

| Module         | Role                                   | Status              |
|----------------|----------------------------------------|---------------------|
| `gen`          | Stimulus generation and visualisation  | Stable (v2.0)       |
| `ext`          | EyeLink ASC parsing                    | Stable (v2.0)       |
| `ext_generic`  | Tracker-agnostic raw-gaze input        | **New in v2.1**     |
| `ext_robust`   | Parameter tuning and edge-case helpers | **New in v2.1**     |
| `cal`          | Metric computation                     | Stable (v2.0)       |

---

## Typical workflow

Three input channels feed the same downstream pipeline:

```
         plain text           EyeLink ASC         raw [t, x, y]
             │                     │                    │
             ▼                     ▼                    ▼
        gen.Praster          ext.read_SRRasc    ext_generic.read_raw_csv
                                   │                    │
                                   │            ext_generic.detect_events
                                   │                    │
                                   └──────────┬─────────┘
                                              ▼
                                    ext.cal_crlSacFix
                                  (+ ext_robust helpers)
                                              │
                                              ▼
                                    classified events CSVs
                                              │
                                              ▼
                                       cal.cal_write_EM
                                              │
                                              ▼
                                      per-word metrics CSV
                                  → import into R / pandas
                                  → fit lme4 / lmerTest models
```

The same script can also call `gen.draw_SacFix` to overlay events on the stimulus bitmap (Figure 3 of the paper) or `gen.animate` to produce an MP4 animation synchronised with an oral-reading audio track.

---

## Supported languages

`gen.Praster` and `gen.Gen_Bitmap_RegFile` produce bitmaps and region files in:

| Family       | Languages                                                           |
|--------------|---------------------------------------------------------------------|
| Alphabetic   | English, German, French, Dutch, Italian, Spanish, Portuguese, Polish, Greek |
| East Asian   | Chinese, Japanese, Korean                                          |

**Input conventions:**

- **Alphabetic languages:** whitespace separates words. Leading and trailing punctuation is attached to the neighbouring word.
- **Chinese:** insert a vertical bar (`|`) between words in the input text (e.g. `"我|喜欢|读书"`); the bars are stripped from the rendered bitmap but used to populate the region file.
- **Japanese and Korean:** inter-word spaces are optional. If present, each space-separated token becomes an AOI; if absent, each character becomes an AOI.

Rendered multilingual examples are available in `examples/textRasters/`.

---

## Supported trackers

| Tracker                     | Entry point                        | Notes                                       |
|-----------------------------|------------------------------------|---------------------------------------------|
| SR-Research EyeLink         | `ext.read_SRRasc`                  | ASC files from `edf2asc`                    |
| Tobii Pro Lab               | `ext_generic.tobii_to_raw`         | TSV exports                                 |
| Pupil Labs                  | `ext_generic.pupil_labs_to_raw`    | `gaze_positions.csv` + screen resolution    |
| SMI RED / iView             | `ext_generic.smi_to_raw`           | Sample reports                              |
| WebGazer.js                 | `ext_generic.webgazer_to_raw`      | JSON-lines logs                             |
| **Any other tracker**       | `ext_generic.read_raw_csv`         | Generic CSV/TSV with user-specified columns |

The four format-specific loaders return the same DataFrame schema; after that, event detection (`ext_generic.detect_events`) and line classification (`ext.cal_crlSacFix`) proceed identically regardless of the source tracker.

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

Both algorithms follow the formulations of Salvucci and Goldberg (2000). On a standard laptop, `detect_events` processes a 30-minute 1 kHz raw-gaze trace (1.8 million samples) in approximately 2.2 seconds.

---

## Cross-line classifier and parameter tuning

`ext.cal_crlSacFix` walks through fixations in temporal order and promotes/demotes a running line counter whenever the horizontal displacement between consecutive fixations exceeds an adaptive threshold. Two parameters control the behaviour:

- **`diff_ratio`** (default `0.6`) — fraction of a line's horizontal extent that a between-fixation jump must exceed to count as cross-line.
- **`frontrange_ratio`** (default `0.2`) — fraction of a line treated as its "front", required for backward cross-line detection.

Defaults reproduce Gong and Shuai (2023). The new `ext_robust` module lets you tune them from your own data:

```python
from pyemread import ext, ext_robust as er
import pandas as pd

reg = pd.read_csv('story01.region.csv')
FixDF = pd.read_csv('./data/P01_Fix.csv')

# 1. Data-driven recommendation
rec = er.recommend_diff_ratio(reg, FixDF)
# {'diff_ratio': 0.68, 'frontrange_ratio': 0.25, 'line_extent_px': 667, ...}

# 2. Parameter-sensitivity report (Cohen's kappa vs default)
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

# 3. Pre-scan detection (readers who scan the whole passage before reading)
mask = er.detect_prescan(FixDF, reg)
FixDF_clean = er.mask_prescan_fixations(FixDF, reg)

# 4. Orphan-fixation reclassification (y > one line-space off)
FixDF_fixed = er.reclassify_orphans(FixDF_clean, reg)
```

On 1,195 pooled real fixations from three subjects of Gong and Shuai (2023), Cohen's κ stays ≥ 0.80 across 67 % of the parameter grid; the data-driven recommender returns `diff_ratio = 0.68` and `frontrange_ratio = 0.25`, slightly more conservative than the defaults but inside the high-κ band. See Figure 5 of the accompanying SoftwareX paper.

---

## Eye-movement metrics

`cal.cal_write_EM` computes 18 per-word metrics grouped by processing stage. End-of-sentence and end-of-paragraph wrap-up effects are preserved in `spurt` and `spcount`; only whole-passage re-scans (flagged by the wrap-up heuristic or by `ext_robust.mask_prescan_fixations`) are excluded from analysis. Missing values are written as `NaN`.

### Whole-text metrics

| Name       | Definition                                                              |
|------------|-------------------------------------------------------------------------|
| `tffixos`  | Character offset of each word's first-pass fixation from the start of the first sentence |
| `tffixurt` | Total duration of each word's first-pass fixation                       |
| `tffixcnt` | Total count of valid fixations in the trial                             |
| `tregrcnt` | Total count of regressive saccades in the trial                         |

### First-pass fixation metrics

| Name        | Definition                                                              |
|-------------|-------------------------------------------------------------------------|
| `fpurt`     | First-pass reading time (sum of first-pass fixation durations)          |
| `fpcount`   | Number of first-pass fixations                                          |
| `fpregres`  | 1 if a regression starts from this word, 0 otherwise                    |
| `fpregreg`  | Target word-region index of the first-pass regression                   |
| `fpregchr`  | Character offset of the first-pass regression target                    |
| `ffos`      | Character offset of the first fixation from the region's start          |
| `ffixurt`   | Duration of the first first-pass fixation                               |
| `spilover`  | Duration of the first fixation that falls beyond the region (spillover) |

### Regression-path metrics

| Name        | Definition                                                              |
|-------------|-------------------------------------------------------------------------|
| `rpurt`     | Sum of fixation durations in the regression path                        |
| `rpcount`   | Number of fixations in the regression path                              |
| `rpregreg`  | Smallest word-region index visited inside the regression path           |
| `rpregchr`  | Character offset inside that smallest region                            |

### Second-pass fixation metrics

| Name       | Definition                                                              |
|------------|-------------------------------------------------------------------------|
| `spurt`    | Sum of fixation durations on the word after the regression path closes  |
| `spcount`  | Number of second-pass fixations                                         |

For formal definitions with worked examples, see the SoftwareX manuscript supplement (Appendix A) or [API_REFERENCE.md](API_REFERENCE.md).

---

## Examples

The `examples/` directory contains:

| Path                            | Contents                                                              |
|---------------------------------|-----------------------------------------------------------------------|
| `examples/oralReading/`         | Three English passages (story01–story03) with matching EyeLink ASC files and oral-reading WAV recordings for subjects 1950138, 1950149, 1950168. |
| `examples/textRasters/`         | Rendered bitmaps and region files for English, Chinese, Japanese, and Korean passages. |
| `examples/example_workflow.py`  | Complete end-to-end demonstration of the pipeline (reproduces Figures 2, 3, and 6 of the SoftwareX paper). |
| `figures/`                      | Six 300 dpi figures from the SoftwareX paper.                         |

To reproduce the worked example:

```bash
cd examples
python example_workflow.py
```

---

## API reference

The complete API documentation, with signatures, parameters, return types, and examples for every public function across all five modules, lives in **[API_REFERENCE.md](API_REFERENCE.md)** at the repository root.

A few high-frequency entry points to get you started:

| Function                                  | Purpose                                                |
|-------------------------------------------|--------------------------------------------------------|
| `gen.Praster(...)`                        | Generate one stimulus bitmap + region file             |
| `ext.read_SRRasc(direct, subjID, ExpType)`| Parse an EyeLink ASC file → `(SacDF, FixDF)`           |
| `ext_generic.read_raw_csv(path, ...)`     | Generic raw-gaze CSV/TSV loader                        |
| `ext_generic.detect_events(samples, ...)` | I-VT or I-DT event detection from raw samples          |
| `ext_generic.tobii_to_raw(path)`          | Tobii Pro Lab TSV loader                               |
| `ext.cal_crlSacFix(...)`                  | Cross-line classifier (the methodological novelty)     |
| `ext_robust.recommend_diff_ratio(...)`    | Data-driven parameter recommendation                   |
| `ext_robust.sensitivity_report(...)`      | κ-agreement heatmap over a parameter grid              |
| `cal.cal_write_EM(...)`                   | Compute the 18 per-word eye-movement metrics           |

---

## Citation

If you use `pyemread` in your research, please cite:

```bibtex
@article{gong2026pyemread,
  title   = {pyemread: a Python package for multi-line text reading
             eye-movement experiments with a temporally-informed
             cross-line classifier and tracker-agnostic gaze input},
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
- `API_REFERENCE.md`: complete API documentation for every public function.
- Real-data validation on 1,195 fixations from three subjects of Gong and Shuai (2023); see Figure 5 of the SoftwareX paper.
- Speed benchmarks: ASC parsing ≈ 0.1 s per subject; 30-minute 1 kHz raw-gaze trace processed in ≈ 2.2 s.

**Changed**

- `__init__.py` uses graceful fallback imports so lightweight installations without FFmpeg or ASC-parser dependencies still load `ext_generic` and `ext_robust`.

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

## License and contact

`pyemread` is released under the [MIT License](LICENSE).

Example data in `examples/oralReading/` are included with the permission of D. Braze (Haskins Laboratories).

**Tao Gong** — <gtojty@gmail.com>
School of Foreign Languages, Zhejiang University of Finance and Economics, Hangzhou, China.

Bug reports and feature requests: <https://github.com/gtojty/pyemread/issues>
