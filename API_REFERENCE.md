# pyemread API Reference

**Version:** v2.1.0 · **License:** MIT · **Repository:** <https://github.com/gtojty/pyemread>

This document is a complete reference for the public API of `pyemread` v2.1. Each module section lists the user-facing functions together with their signatures, parameters, return values and short examples. Private helpers (names starting with `_`) are omitted. For a conceptual introduction see the `README.md`; for worked examples see `examples/example_workflow.py`.

---

## Table of contents

- [Module `gen`](#module-gen) — stimulus bitmaps, region files, visualisation, MP4 animations
- [Module `ext`](#module-ext) — SR-Research EyeLink ASC parsing and cross-line classification
- [Module `ext_generic`](#module-ext_generic) — tracker-agnostic raw-gaze input (**new in v2.1**)
- [Module `ext_robust`](#module-ext_robust) — parameter tuning and edge-case helpers (**new in v2.1**)
- [Module `cal`](#module-cal) — computation of 18 per-word eye-movement metrics
- [Conventions](#conventions)

---

## Module `gen`

Stimulus generation, static visualisation, and animation.

### `Praster(direct, fontpath, stPos, langType, codeMethod='utf_8', text=None, dim=(1280, 1024), fg=(0, 0, 0), bg=(232, 232, 232), lmargin=215, tmargin=86, linespace=65, fht=18, fwd=11, bbox=False, bbox_big=False, addspace=18, ID=None)`

Generate a single bitmap (PNG) and region file (CSV) from one line-broken text.

| Parameter | Description |
|---|---|
| `direct` | Output folder (created if absent). |
| `fontpath` | Absolute path to a monospace `.ttf`/`.otf` (Liberation Mono for Latin scripts; Noto Sans CJK for East Asian). |
| `stPos` | `'TopLeft'`, `'Center'`, or `'Auto'`. |
| `langType` | `'English'`, `'German'`, `'French'`, `'Dutch'`, `'Italian'`, `'Spanish'`, `'Greek'`, `'Chinese'`, `'Japanese'`, or `'Korean'`. |
| `text` | List of strings, one per text line. For Chinese, insert `|` between words. |
| `dim` | `(screen_width, screen_height)` in pixels. |
| `fg`, `bg` | Foreground/background RGB. |
| `lmargin`, `tmargin` | Left and top margins in pixels. |
| `linespace` | Vertical space between baselines (pixels). |
| `fht`, `fwd` | Font height and per-character width (pixels). |
| `bbox`, `bbox_big` | If `True`, draw per-word bounding boxes on the bitmap. |
| `addspace` | Pixels added to the left/right of the first/last word in each line for overshoot capture. |
| `ID` | Filename stem (e.g., `'story01'` produces `story01.png` and `story01.region.csv`). |

**Returns:** writes `<direct>/<ID>.png` and `<direct>/<ID>.region.csv`. No return value.

```python
from pyemread import gen
gen.Praster(
    direct='./output',
    fontpath='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    stPos='TopLeft', langType='English',
    text=['Alex and Tim are lost.', 'They went hiking today.'],
    ID='story01',
)
```

### `Gen_Bitmap_RegFile(direct, fontName, stPos, langType, textFileNameList, genmethod=2, ...)`

Batch wrapper that calls `Praster` for every text file in a folder.

- `genmethod=1` treats the whole file as one stimulus; `genmethod=2` treats each blank-line-delimited block as a separate stimulus.
- Lines starting with `#` are treated as meta-comments and stripped.
- Produces one bitmap and region file per stimulus, named after the input text file.

### `draw_SacFix(direct, subjID, regfileNameList, bitmapNameList, drawType, max_FixRadius=30, drawFinal=False, showFixDur=False, PNGopt=0)`

Overlay saccades and fixations on the stimulus bitmap.

| Parameter | Description |
|---|---|
| `drawType` | `'ALL'`, `'Fix'`, `'Sac'`, or `'crlFix'` / `'crlSac'` to draw only cross-line events. |
| `max_FixRadius` | Maximum circle radius in pixels; fixation durations are scaled to this. |
| `drawFinal` | If `True`, fixations after the wrap-up break are included. |
| `showFixDur` | If `True`, the duration in ms is printed next to each circle. |
| `PNGopt` | `0` = render text from the region file; `1` = use the original bitmap as background. |

Left-eye fixations are drawn as green outline circles; right-eye fixations as red filled circles.

### `draw_SacFix_b(...)`

Batch version: applies `draw_SacFix` to every subject folder under `direct`.

### `animate(direct, subjID, trialID)` · `animate_TimeStamp(direct, subjID, trialID)`

Generate an MP4 animation of fixations (or time-stamped samples) synchronised with the oral-reading `.wav` file stored next to the data. Uses `matplotlib.animation` + MoviePy/FFmpeg.

- `animate` renders one circle per fixation, appearing at its `start` and disappearing at its `end`.
- `animate_TimeStamp` renders a moving cursor at sample rate; recommended only when the tracker rate is ≤ the screen refresh rate.

**Returns:** writes `<direct>/<subjID>_trial<N>.mp4`.

### `changePNG2GIF(direct)`

Convert every `.png` in `direct` to `.gif`. Retained for backward compatibility with the pre-v2.0 `turtle` animation backend; most users can ignore it.

### `updReg(direct, regfileNameList, addspace)`

Expand each word's bounding box left/right by `addspace` pixels for overshoot capture. Use when the original region file was generated with `addspace=0`.

---

## Module `ext`

SR-Research EyeLink ASC parser, event extractor, and cross-line classifier. Input ASC files are produced by SR Research's `edf2asc` utility.

### `read_SRRasc(direct, subjID, ExpType, rec_lastFix=False, lump_Fix=True, ln=50, zn=50, mn=50)`

Parse an ASC file and return `(SacDF, FixDF)` — two pandas DataFrames of saccades and fixations across all trials.

| Parameter | Description |
|---|---|
| `direct` | Folder containing `<subjID>.asc`. |
| `subjID` | Subject identifier. |
| `ExpType` | `'RAN'` (single-line rapid-naming) or `'RP'` (multi-line paragraph reading). |
| `rec_lastFix` | If `False`, the last fixation of each trial is marked invalid. |
| `lump_Fix` | If `True`, fixations shorter than `ln` ms are merged with neighbours. |
| `ln`, `zn`, `mn` | Minimum duration thresholds (ms) for lumping (`ln`), zero-duration flagging (`zn`), and validity (`mn`). |

**DataFrame schemas:**

- `SacDF`: `subj`, `trial_id`, `eye`, `start`, `end`, `dur`, `x1`, `y1`, `x2`, `y2`, `ampl`, `pv`, `line_no` (post-classification)
- `FixDF`: `subj`, `trial_id`, `eye`, `start`, `end`, `dur`, `x`, `y`, `valid`, `line_no`, `region_no`

Monocular files (left-only or right-only) and binocular files are both supported. Custom `MSG` lines are captured in an auxiliary `msg_df` attribute.

### `read_TimeStamp(direct, subjID, ExpType)` · `read_Stamp(direct, subjID, ExpType)`

Return a DataFrame of sample-level data (`StampDF`): `subj`, `trial_id`, `eye`, `time`, `x`, `y`, `pupil_size`. `read_Stamp` is a light-weight variant that skips blink parsing.

### `write_Sac_Report(direct, subjID, SacDF)` · `write_Fix_Report(direct, subjID, FixDF)` · `write_TimeStamp_Report(direct, subjID, StampDF)`

Write each DataFrame to `<direct>/<subjID>_Sac.csv`, `..._Fix.csv`, or `..._Stamp.csv`.

### `read_write_SRRasc(direct, subjID, ExpType, ...)` and `read_write_SRRasc_b(direct, ExpType, ...)`

One-shot: parse the ASC file and write the three CSV reports. The `_b` suffix indicates a batch function that iterates over all subject folders under `direct`.

### `cal_crlSacFix(direct, subjID, regfileNameList, ExpType, classify_method='DIFF', recStatus=True, diff_ratio=0.6, frontrange_ratio=0.2, y_range=60, addCharSp=1)`

The core cross-line classifier. Returns `(SacDF, crlSac, FixDF, crlFix)`: the classified saccade/fixation DataFrames plus two sub-DataFrames containing only the cross-line events.

| Parameter | Description |
|---|---|
| `regfileNameList` | List of region-file basenames, one per trial. The ordering matches `trial_id`. |
| `classify_method` | `'DIFF'` uses between-fixation horizontal displacements; `'SAC'` uses explicitly detected saccades. |
| `recStatus` | If `True`, a `log.txt` is written listing questionable trials. |
| `diff_ratio` | Fraction of a line's horizontal extent that a jump must exceed to count as cross-line (default 0.6). |
| `frontrange_ratio` | Fraction of the line treated as its "front" (default 0.2). |
| `y_range` | Vertical pixel threshold for wrap-up detection (default 60). |
| `addCharSp` | Characters of padding added to the left/right bounds of the first/last word in each line, for overshoot capture. |

Default `diff_ratio = 0.6` and `frontrange_ratio = 0.2` reproduce the values used by Gong and Shuai (2023). Use `pyemread.ext_robust.recommend_diff_ratio` to tune these from your own data.

### `write_Sac_crlSac(direct, subjID, SacDF, crlSac)` · `write_Fix_crlFix(direct, subjID, FixDF, crlFix)`

Write classified DataFrames to CSV.

### `cal_write_SacFix_crlSacFix(...)` · `cal_write_SacFix_crlSacFix_b(...)`

One-shot read + classify + write. Batch version applies to every subject folder under `direct`.

### `read_cal_SRRasc(...)` · `read_cal_write_SRRasc(...)` · `read_cal_write_SRRasc_b(...)`

End-to-end pipelines combining ASC parsing, cross-line classification, and CSV output.

### `cal_TimeStamp(direct, subjID, regfileNameList, ExpType, align_method, addCharSp=1)`

Classify sample-level time-stamped data into text lines. `align_method` is one of `'FIX'` (align to fixations in `_Fix.csv`) or `'SAC'` (align to saccades).

### `cal_write_TimeStamp(...)` · `cal_write_TimeStamp_b(...)` · `read_cal_TimeStamp(...)` · `read_cal_write_TimeStamp(...)` · `read_cal_write_TimeStamp_b(...)`

Convenience wrappers combining the above.

---

## Module `ext_generic`

Tracker-agnostic raw-gaze processing (new in v2.1). Ingests `[t, x, y]` samples from any tracker and produces DataFrames in the same schema as `ext.read_SRRasc`, so the downstream classifier and metric computation are tracker-independent.

### `read_raw_csv(path, time_col='time', x_col='x', y_col='y', eye_col=None, trial_col=None, time_unit='ms', sep=',', encoding='utf-8')`

Generic CSV/TSV reader for raw gaze samples.

| Parameter | Description |
|---|---|
| `path` | File path. |
| `time_col`, `x_col`, `y_col` | Column names for time, x-pixel, y-pixel. |
| `eye_col` | Optional column with `'L'`/`'R'` labels. Absent → `'R'`. |
| `trial_col` | Optional column with integer trial index. Absent → single trial. |
| `time_unit` | `'ms'`, `'s'`, `'us'`, or `'ns'`. Converted internally to ms. |
| `sep` | Field separator. |
| `encoding` | File encoding. |

**Returns:** DataFrame with columns `['time', 'x', 'y', 'eye', 'trial_id']`, time normalised to start at 0.

### `read_raw_tsv(path, **kwargs)`

Convenience wrapper with `sep='\t'`.

### `detect_events(samples, method='IVT', velocity_threshold=30.0, dispersion_threshold=35.0, min_fix_duration=50.0, min_sac_duration=10.0, smooth_window=3, px_per_deg=None)`

Fixation/saccade detection from raw samples.

| Parameter | Description |
|---|---|
| `samples` | DataFrame from `read_raw_csv`. |
| `method` | `'IVT'` (velocity-threshold, Salvucci & Goldberg 2000) or `'IDT'` (dispersion-threshold). |
| `velocity_threshold` | Cutoff. If `px_per_deg=None` interpreted in pixels/second; otherwise in degrees/second. |
| `dispersion_threshold` | For I-DT: maximum x+y range (pixels) within a candidate fixation. |
| `min_fix_duration`, `min_sac_duration` | Minimum event duration (ms). |
| `smooth_window` | Samples of centred moving-average smoothing (`1` = none). |
| `px_per_deg` | Pixels per visual degree. When given, `velocity_threshold` is read as deg/s. |

**Returns:** `(SacDF, FixDF)` with columns matching `ext.read_SRRasc`.

**Benchmark:** on a standard laptop a 30-minute 1 kHz raw-gaze trace (1.8 million samples) is processed in approximately 2.2 seconds.

```python
from pyemread import ext_generic as eg
samples = eg.tobii_to_raw('P01_tobii.tsv')
SacDF, FixDF = eg.detect_events(
    samples, method='IVT',
    velocity_threshold=30.0,   # deg/s
    px_per_deg=30,
    min_fix_duration=50,
)
```

### `classify_lines(SacDF, FixDF, reg_df, method='DIFF', diff_ratio=0.6, frontrange_ratio=0.2, y_range=60, handle_prescan=True)`

In-memory wrapper round `ext.cal_crlSacFix`. Use when you want to skip CSV I/O.

- `reg_df` is a region DataFrame (loaded from `*_region.csv`).
- `handle_prescan=True` applies `ext_robust.mask_prescan_fixations` before classification.

**Returns:** `(SacDF_out, crlSac, FixDF_out, crlFix)`.

### `write_events(direct, subjID, SacDF, FixDF)`

Write the two DataFrames to `<direct>/<subjID>_Sac.csv` and `..._Fix.csv`, matching the schema of `ext.write_Sac_Report` / `ext.write_Fix_Report`.

### Tracker-specific loaders

Each loader returns a DataFrame in the same schema as `read_raw_csv` (`time`, `x`, `y`, `eye`, `trial_id`).

**`tobii_to_raw(path, eye='average')`**

Tobii Pro Lab TSV export. `eye` ∈ {`'left'`, `'right'`, `'average'`}. Times are converted from µs.

**`pupil_labs_to_raw(gaze_positions_csv, screen_w, screen_h)`**

Pupil Labs `gaze_positions.csv`. Coordinates are de-normalised from 0–1 to pixels, with y-up flipped to screen-down convention.

**`smi_to_raw(path)`**

SMI RED / iView sample-report `.txt`. Detects both `L POR X [px]` and `R POR X [px]` columns and produces binocular output.

**`webgazer_to_raw(path)`**

WebGazer.js JSON-lines log (one `{t, x, y, trial?}` object per line).

---

## Module `ext_robust`

Parameter tuning and edge-case handling (new in v2.1).

### `recommend_diff_ratio(reg_df, FixDF)`

Data-driven recommendation for the `diff_ratio` and `frontrange_ratio` parameters of the cross-line classifier.

- `reg_df` : region DataFrame (from `*_region.csv`).
- `FixDF` : fixation DataFrame with `x` and `trial_id`.

**Returns:** dict with keys `diff_ratio`, `frontrange_ratio`, `line_extent_px`, `n_fixations`, `note`.

**On the Gong & Shuai (2023) data** (1,195 pooled right-eye fixations from three subjects reading story01 and story02), this function returns `diff_ratio = 0.68` and `frontrange_ratio = 0.25`, both inside the high-κ band around the defaults (see Figure 5 of the manuscript).

```python
from pyemread import ext_robust as er
rec = er.recommend_diff_ratio(reg_df, FixDF)
# {'diff_ratio': 0.68, 'frontrange_ratio': 0.25, ...}
```

### `detect_prescan(FixDF, reg_df, min_scan_fixations=4, y_tolerance=1.5)`

Return a boolean Series marking fixations that form a pre-reading scan of the whole passage.

A pre-scan is a contiguous block at the start of the trial containing at least `min_scan_fixations` fixations with vertical span > `y_tolerance × linespace`.

### `mask_prescan_fixations(FixDF, reg_df, **kwargs)`

Return a copy of `FixDF` with `valid=0` on fixations flagged by `detect_prescan`.

### `reclassify_orphans(FixDF, reg_df, line_tolerance=None)`

Reassign any fixation whose `y`-coordinate lies more than `line_tolerance` pixels from its line's baseline to the nearest line. Defaults to one line-space.

**Returns:** copy of `FixDF` with possibly-updated `line_no` and a new boolean `reclassified` column.

### `sensitivity_report(classify_fn, grid_diff_ratio=(0.4, 0.5, 0.6, 0.7, 0.8), grid_frontrange=(0.15, 0.20, 0.25, 0.30), reference=(0.6, 0.2))`

Run `classify_fn(diff_ratio, frontrange_ratio)` over a parameter grid and report Cohen's κ agreement with the classification produced at `reference`.

`classify_fn` must return a vector of line assignments (one per fixation). The user supplies this closure so the report is agnostic to whether classification is in-memory or disk-based.

**Returns:** DataFrame with columns `diff_ratio`, `frontrange_ratio`, `kappa_vs_reference`, `pct_agreement`.

**The sensitivity-report idiom** is not specific to reading research: any downstream classifier with tunable parameters can be profiled the same way by wrapping it in an appropriate closure.

```python
def classify(dr, fr):
    _, _, _, crl = ext.cal_crlSacFix(
        './data', 'P01', ['story01.region.csv'],
        'RP', diff_ratio=dr, frontrange_ratio=fr)
    return FixDF.merge(crl, on='start', how='left')['line_no']

report = er.sensitivity_report(classify)
```

---

## Module `cal`

Computation of 18 per-word eye-movement metrics.

### `cal_write_EM(direct, subjID, regfileNameList, addCharSp=1)`

Compute the metrics for one subject and write them to `<direct>/<subjID>_EM_trial<N>.csv` (one file per trial). For binocular data, separate `_L.csv` and `_R.csv` files are written.

**Metric groups and columns:**

- **Whole-text:** `tffixos`, `tffixurt`, `tffixcnt`, `tregrcnt`
- **First-pass:** `fpurt`, `fpcount`, `fpregres`, `fpregreg`, `fpregchr`, `ffos`, `ffixurt`, `spilover`
- **Regression path:** `rpurt`, `rpcount`, `rpregreg`, `rpregchr`
- **Second-pass:** `spurt`, `spcount`

Missing values are `NaN`. `spurt` and `spcount` preserve end-of-sentence and end-of-paragraph wrap-up effects — only whole-passage re-scans (flagged by `ext.cal_crlSacFix`'s wrap-up heuristic, or by `ext_robust.mask_prescan_fixations`) are excluded.

See the accompanying Supplementary Material (Appendix A) for formal definitions of every metric.

### `cal_write_EM_b(direct, regfileNameList, addCharSp=1)`

Batch version applying `cal_write_EM` to every subject folder under `direct`.

### `mergeCSV(direct, regfileNameList, subjID)` · `mergeCSV_b(direct, regfileNameList)`

Concatenate per-trial metric CSVs into a single subject-level (or study-level) CSV suitable for mixed-effects modelling (e.g. in `lme4`/`lmerTest`):

```r
library(lme4)
d <- read.csv("study_level_EM.csv")
m <- lmer(fpurt ~ word_length + word_freq + (1 | subj), data = d)
```

---

## Conventions

### File naming

All functions that read from or write to disk follow the pattern `<subjID>_<type>.csv` inside a single subject folder. The suffixes are:

| Suffix | Content |
|---|---|
| `_Sac.csv` | Per-saccade records |
| `_Fix.csv` | Per-fixation records |
| `_Stamp.csv` | Sample-level records |
| `_crlSac.csv` | Cross-line saccades only |
| `_crlFix.csv` | Cross-line fixations only |
| `_EM_trial<N>.csv` | Per-word eye-movement metrics for trial `N` |
| `_EM_trial<N>_L.csv` / `_R.csv` | Per-word metrics for binocular data, per eye |

### Coordinate system

All coordinates are in screen pixels with the origin at the top-left corner. Positive `x` runs rightward, positive `y` runs downward. `baseline` in the region file is the `y`-coordinate of a line's text baseline; `y1_pos` and `y2_pos` delimit the bounding-box top and bottom.

### Time units

All times are in milliseconds (ms) in the saccade/fixation/sample DataFrames. The `ext_generic.read_raw_csv` loader auto-converts from `'s'`, `'us'`, or `'ns'` at load time.

### Missing values

Functions that compute per-word metrics emit `NaN` when a word was never fixated (skipped) or when the metric is otherwise undefined (e.g. `fpregreg` when there is no first-pass regression).

### Backward compatibility

Function names use CamelCase (matching the original v0.1 and v2.0 API). All functions can also be accessed via snake_case aliases (e.g. `read_srr_asc` = `read_SRRasc`), introduced in v2.0 for PEP 8 compliance. Either form is supported for the life of the v2.x series.

---

## Minimal end-to-end example

```python
from pyemread import gen, ext, cal, ext_robust as er
import pandas as pd

# 1. Generate stimulus
gen.Praster(
    direct='./output',
    fontpath='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    stPos='TopLeft', langType='English',
    text=['Alex and Tim are lost.', 'They went hiking today.'],
    ID='story01',
)

# 2. Parse ASC file
SacDF, FixDF = ext.read_SRRasc('./data', '1950138', 'RP')
ext.write_Sac_Report('./data', '1950138', SacDF)
ext.write_Fix_Report('./data', '1950138', FixDF)

# 3. Classify cross-line events
ext.cal_write_SacFix_crlSacFix(
    './data', '1950138', ['story01.region.csv'], 'RP',
)

# 4. Optional: tune parameters from the data itself
reg = pd.read_csv('./output/story01.region.csv')
rec = er.recommend_diff_ratio(reg, FixDF)
print(rec)   # {'diff_ratio': 0.68, 'frontrange_ratio': 0.25, ...}

# 5. Compute 18 per-word metrics
cal.cal_write_EM('./data', '1950138', ['story01.region.csv'])
```

---

_For questions or bug reports, please open an issue at <https://github.com/gtojty/pyemread/issues> or email Tao Gong at <gtojty@gmail.com>._
