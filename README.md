# Pyemread

**A Python Package for Multi-line Text Reading Eye Movement Experiments**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Pyemread is a comprehensive Python package designed for conducting and analyzing eye movement experiments involving single and multi-line text reading. It provides tools for:

- **Frontend**: Generating pixel-level bitmaps and word-wise region files for experimental stimuli
- **Backend**: Extracting, classifying, and analyzing eye movement data from SR Research EyeLink trackers

## Features

### Stimulus Generation (gen.py)
- Generate high-quality bitmaps for single/multi-line text displays
- Create word-wise region files with precise bounding boxes
- Support for multiple languages: English, German, French, Dutch, Italian, Spanish, Greek, Chinese, Korean, and Japanese
- Customizable fonts, colors, margins, and line spacing

### Data Extraction (ext.py)
- Parse SR Research EyeLink ASCII (.asc) files
- Extract saccades, fixations, and blinks with temporal information
- Lump short fixations using configurable thresholds
- Classify saccades/fixations into text lines using heuristic algorithms
- Identify cross-line eye movements (forward and backward)

### Measure Calculation (cal.py)
- Calculate comprehensive eye movement metrics:
  - **First-pass measures**: fpurt, fpcount, ffos, ffixurt, spilover
  - **Regression measures**: fpregres, fpregreg, fpregchr
  - **Regression path measures**: rpurt, rpcount, rpregreg, rpregchr
  - **Second-pass measures**: spurt, spcount
  - **Trial-level measures**: tffixos, tffixurt, tfixcnt, tregrcnt

### Visualization (gen.py)
- Draw saccades and fixations on bitmap backgrounds
- Create MP4 video animations with synchronized audio playback
- Support for single-eye and binocular data visualization

## Installation

### Requirements
- Python 3.8 or higher
- Required packages:
  ```
  numpy>=1.20.0
  pandas>=1.3.0
  Pillow>=8.0.0
  matplotlib>=3.4.0
  ```
- Optional for animations:
  ```
  ffmpeg (system installation)
  ```

### Install from source
```bash
git clone https://github.com/gtojty/pyemread.git
cd pyemread
pip install -e .
```

### Install dependencies
```bash
pip install numpy pandas Pillow matplotlib
```

For animation support, install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Quick Start

### 1. Generate Bitmap and Region File

```python
import pyemread as pr

# Define text content (each string is one line)
text_lines = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy tabby cat sleeps in the sun all afternoon."
]

# Generate bitmap and region file
pr.gen.praster(
    direct='/path/to/output',
    fontpath='/path/to/font.ttf',
    st_pos='TopLeft',
    lang_type='English',
    text=text_lines,
    dim=(1280, 1024),
    fht=18,
    id_name='story01',
    regfile=True,
    bbox=False
)
```

### 2. Extract Eye Movement Data

```python
import pyemread as pr

# Read and extract saccades/fixations from ASCII file
sac_df, fix_df = pr.ext.read_srrasc(
    direct='/path/to/data',
    subj_id='subject01',
    exp_type='RAN',
    lump_fix=True,
    ln=50,  # Lump threshold (ms)
    mn=50   # Minimum valid duration (ms)
)

# Write to CSV files
pr.ext.write_sac_report(direct, subj_id, sac_df)
pr.ext.write_fix_report(direct, subj_id, fix_df)
```

### 3. Classify Cross-line Eye Movements

```python
import pyemread as pr

# Calculate cross-line saccades and fixations
sac_df, crl_sac, fix_df, crl_fix = pr.ext.cal_crl_sac_fix(
    direct='/path/to/data',
    subj_id='subject01',
    regfile_list=['story01.region.csv'],
    exp_type='RAN',
    classify_method='SAC',
    diff_ratio=0.6,
    frontrange_ratio=0.2,
    y_range=60
)
```

### 4. Calculate Eye Movement Measures

```python
import pyemread as pr

# Calculate and save EM measures
pr.cal.cal_write_em(
    direct='/path/to/data',
    subj_id='subject01',
    regfile_list=['story01.region.csv'],
    add_char_sp=1
)
```

### 5. Visualize Eye Movements

```python
import pyemread as pr

# Draw saccades and fixations on bitmap
pr.gen.draw_sac_fix(
    direct='/path/to/data',
    subj_id='subject01',
    regfile_list=['story01.region.csv'],
    bitmap_list=['story01.png'],
    draw_type='ALL',
    max_fix_radius=30,
    show_fix_dur=True
)

# Create animation video
pr.gen.create_animation_video(
    fix_df=fix_df,
    bitmap_path='/path/to/story01.png',
    output_path='/path/to/animation.mp4',
    audio_path='/path/to/audio.wav',  # Optional
    fps=15,
    max_radius=30
)
```

## Eye Movement Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **First-pass** | fpurt | First-pass fixation time (sum of durations) |
| | fpcount | Number of first-pass fixations |
| | ffos | Offset of first fixation from word start (characters) |
| | ffixurt | Duration of first first-pass fixation |
| | spilover | Duration of first fixation beyond the region |
| **Regression** | fpregres | Whether first-pass regression occurred (0/1) |
| | fpregreg | Word region where regression ended |
| | fpregchr | Character offset where regression ended |
| **Regression path** | rpurt | Total duration of regression path |
| | rpcount | Number of fixations in regression path |
| | rpregreg | Smallest region index visited in path |
| | rpregchr | Character offset in smallest region |
| **Second-pass** | spurt | Second-pass fixation time |
| | spcount | Number of second-pass fixations |
| **Trial-level** | tffixos | Total first-fixation offset from text start |
| | tffixurt | Total first-fixation duration |
| | tfixcnt | Total number of valid fixations |
| | tregrcnt | Total number of regressive saccades |

## Language Support

| Language | Code | Word Separation |
|----------|------|-----------------|
| English | 'English' | Space-separated |
| German | 'German' | Space-separated |
| French | 'French' | Space-separated |
| Dutch | 'Dutch' | Space-separated |
| Italian | 'Italian' | Space-separated |
| Spanish | 'Spanish' | Space-separated |
| Greek | 'Greek' | Space-separated |
| Chinese | 'Chinese' | Use `\|` for word boundaries |
| Korean | 'Korean' | Optional spaces |
| Japanese | 'Japanese' | Optional spaces |

## File Formats

### Region File (.region.csv)
| Column | Description |
|--------|-------------|
| Name | Stimulus identifier |
| Language | Language type |
| WordID | Word index (1-based) |
| Word | Word text |
| length | Word length in characters |
| height | Word height in pixels |
| baseline | Baseline y-coordinate |
| line_no | Line number (1-based) |
| x1_pos, y1_pos | Top-left corner of word box |
| x2_pos, y2_pos | Bottom-right corner of word box |
| b_x1, b_y1 | Top-left of expanded box |
| b_x2, b_y2 | Bottom-right of expanded box |

### Fixation Report (_Fix.csv)
Contains columns for subject, trial, timing, position, pupil size, validity, and region assignment.

### Saccade Report (_Sac.csv)
Contains columns for subject, trial, timing, start/end positions, amplitude, peak velocity, and line classification.

## Examples

The `examples/` directory contains:
- `oralReading/`: Sample eye tracking data from oral reading experiments
- `textRasters/`: Sample bitmaps and region files for multiple languages

## Citation

If you use Pyemread in your research, please cite:

```bibtex
@article{gong2024pyemread,
  title={Pyemread: A Python Package for Multi-line Text Reading Eye Movement Experiments},
  author={Gong, Tao},
  journal={Behavior Research Methods},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- **Tao Gong** - [gtojty@gmail.com](mailto:gtojty@gmail.com)
- **David Braze** - [davebraze@gmail.com](mailto:davebraze@gmail.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 2.1.1 (2024)
- Modernized codebase for Python 3.8+
- Replaced turtle-based animation with FFmpeg video generation
- Fixed pandas compatibility issues
- Improved cross-platform support
- Enhanced documentation and examples

### Version 0.1.0 (2017)
- Initial release with basic functionality
