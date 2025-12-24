# Pyemread v2.1.1 Update Summary

## Overview

This document summarizes the comprehensive updates made to the **pyemread** package, transforming it from a Python 2.7 codebase to a modern Python 3.8+ package with enhanced functionality, improved code quality, and professional documentation suitable for publication in *Behavior Research Methods*.

---

## Major Changes

### 1. Python 3 Modernization

**Original Issues:**
- Used Python 2 syntax (`print` statements, `reload(sys)`, `has_key()`)
- Relied on deprecated modules (`winsound`, `turtle`)
- Used legacy string formatting
- Lacked type annotations

**Updates:**
- Full Python 3.8+ compatibility
- Modern type annotations throughout
- f-string formatting
- Removed `reload(sys); sys.setdefaultencoding("utf-8")` pattern
- PEP 8 compliant naming conventions (snake_case)

### 2. Animation System Replacement

**Original Issues:**
- Used `turtle` graphics (platform-specific, limited functionality)
- Required `winsound` (Windows-only)
- Could not save animations
- No audio synchronization capability

**Updates:**
- Replaced with **FFmpeg**-based video generation
- Cross-platform compatibility (Windows, Linux, macOS)
- Generates MP4 files with H.264 encoding
- Synchronized audio support (WAV files)
- Duration-scaled fixation circles
- Color-coded eyes (green=left, red=right)

**New Function: `create_animation_video()`**
```python
gen.create_animation_video(
    direct='./data',
    subj_id='1950138',
    trial_id=0,
    output_file='reading_animation.mp4',
    fps=15,
    max_fix_radius=30
)
```

### 3. Code Organization

**Function Naming:**
- Changed from `camelCase` to `snake_case` (PEP 8)
- Examples:
  - `Praster` → `praster`
  - `Gen_Bitmap_RegFile` → `gen_bitmap_regfile`
  - `read_SRRasc` → `read_srr_asc`
  - `cal_crlSacFix` → `cal_crl_sac_fix`

**Module Structure:**
- Proper `__init__.py` with version info
- `setup.py` for pip installation
- Clear separation of public and private functions

### 4. Enhanced Documentation

**Docstrings:**
- Comprehensive Google-style docstrings
- Type annotations for all parameters
- Clear return value descriptions
- Usage examples

**README.md:**
- Installation instructions
- Quick start guide
- Complete API reference
- Eye movement metrics table
- Language support matrix

### 5. Error Handling

**Improvements:**
- Explicit exceptions with informative messages
- Input validation for all public functions
- Graceful handling of missing files
- Warning system for questionable data

---

## File Changes

### gen.py (1,780 lines)
| Feature | Status |
|---------|--------|
| `praster()` | ✓ Updated |
| `gen_bitmap_regfile()` | ✓ Updated |
| `draw_sac_fix()` | ✓ Updated |
| `create_animation_video()` | ✓ **NEW** |
| `change_png2gif()` | ✓ Updated |
| `FontDict` class | ✓ Updated |

### ext.py (2,500 lines)
| Feature | Status |
|---------|--------|
| `read_srr_asc()` | ✓ Updated |
| `cal_crl_sac_fix()` | ✓ Updated |
| `read_cal_srr_asc()` | ✓ Updated |
| Fixation lumping algorithm | ✓ Updated |
| Cross-line classification | ✓ Updated |

### cal.py (800 lines)
| Feature | Status |
|---------|--------|
| `cal_write_em()` | ✓ Updated |
| All EM metrics | ✓ Verified |
| `merge_csv()` | ✓ Updated |

---

## Eye Movement Metrics

The package calculates 18 validated metrics across four categories:

### Trial-Level Metrics
| Metric | Description |
|--------|-------------|
| `tffixos` | Total first-fixation offset from text start |
| `tffixurt` | Total first-pass fixation duration |
| `tfixcnt` | Total valid fixation count |
| `tregrcnt` | Total regressive saccade count |

### First-Pass Fixation Metrics
| Metric | Description |
|--------|-------------|
| `fpurt` | First-pass reading time |
| `fpcount` | First-pass fixation count |
| `ffos` | First fixation offset |
| `ffixurt` | First fixation duration |
| `spilover` | Spillover duration |

### Regression Metrics
| Metric | Description |
|--------|-------------|
| `fpregres` | First-pass regression (0/1) |
| `fpregreg` | Regression target region |
| `fpregchr` | Regression target character |

### Regression Path Metrics
| Metric | Description |
|--------|-------------|
| `rpurt` | Regression path duration |
| `rpcount` | Regression path fixation count |
| `rpregreg` | Leftmost region in path |
| `rpregchr` | Leftmost character in path |

### Second-Pass Metrics
| Metric | Description |
|--------|-------------|
| `spurt` | Second-pass reading time |
| `spcount` | Second-pass fixation count |

---

## Language Support

| Language | Status | Font Requirement |
|----------|--------|------------------|
| English | ✓ | System fonts |
| German | ✓ | System fonts |
| French | ✓ | System fonts |
| Dutch | ✓ | System fonts |
| Italian | ✓ | System fonts |
| Spanish | ✓ | System fonts |
| Greek | ✓ | System fonts |
| Chinese | ✓ | SimHei.ttf or similar |
| Korean | ✓ | Malgun Gothic or similar |
| Japanese | ✓ | MS Gothic or similar |

---

## Dependencies

### Required
```
numpy>=1.20.0
pandas>=1.3.0
Pillow>=8.0.0
matplotlib>=3.4.0
```

### Optional (for animations)
```
FFmpeg (command-line tool)
```

---

## Installation

```bash
# From source
git clone https://github.com/gtojty/pyemread.git
cd pyemread
pip install -e .

# Or directly
pip install .
```

---

## Backward Compatibility

While function signatures have changed, the core functionality remains compatible. Users should update their code as follows:

```python
# Old (v0.1.0)
from pyemread import gen
gen.Praster(direct, fontpath, stPos, langType, ...)

# New (v2.1.1)
from pyemread import gen
gen.praster(direct, fontpath, st_pos, lang_type, ...)
```

---

## Testing

All modules have been tested with:
- Sample eye tracking data (EyeLink 1000, 250 Hz)
- English, Chinese, Korean, and Japanese texts
- Both single-eye and binocular recordings
- Animation generation with audio synchronization

---

## Manuscript Updates

The accompanying manuscript (`Pyemread_Manuscript.docx`) has been revised to:
1. Reflect all code changes and new features
2. Follow *Behavior Research Methods* style guidelines
3. Address reviewer comments about:
   - User-goal oriented organization
   - Algorithm validation methodology
   - Alternative approaches discussion
   - Video generation improvements

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2017 | Initial release (Python 2.7) |
| 2.1.1 | 2024 | Complete modernization (Python 3.8+) |

---

## Authors

- **Tao Gong** - gtojty@gmail.com
- **David Braze** - davebraze@gmail.com

## License

MIT License - See LICENSE file for details

## Citation

```bibtex
@article{gong2024pyemread,
  title={Pyemread: A Python Package for Multi-line Text Reading Eye Movement Experiments},
  author={Gong, Tao and Braze, David},
  journal={Behavior Research Methods},
  year={2024},
  doi={TBD}
}
```
