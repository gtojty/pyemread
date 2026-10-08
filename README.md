# pyemread

**Multi-line text reading eye-movement experiments in Python: stimuli with word-level areas of interest, fixation-to-line assignment and reading measures.**

[![tests](https://github.com/gtojty/pyemread/actions/workflows/tests.yml/badge.svg)](https://github.com/gtojty/pyemread/actions/workflows/tests.yml)
[![docs](https://github.com/gtojty/pyemread/actions/workflows/docs.yml/badge.svg)](https://gtojty.github.io/pyemread/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19622178.svg)](https://doi.org/10.5281/zenodo.19622178)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gtojty/pyemread/blob/master/examples/tutorial.ipynb)

pyemread takes a reading study with multi-line passages from the text to an analysis-ready table of word-level measures:

1. **Stimuli.** `gen.praster` renders a passage as a bitmap and writes a region file with the bounding box and line number of every word, for alphabetic scripts and for Chinese, Japanese and Korean.
2. **Eye-movement input.** EyeLink ASC files (`ext.process_subject`), fixation tables exported by other software (`ext_generic.read_fixations`), Eye-Tracking-BIDS recordings (`ext_generic.read_bids_eyetrack`) and raw samples with I-VT/I-DT event detection (`ext_generic.detect_events`). Continuous recordings can be cut into trials (`ext_generic.epoch`).
3. **Fixation-to-line assignment.** The default is the *slice* algorithm of Glandorf and Schroeder (2021), adapted from Carr et al. (2022). The original pyemread heuristic (`DIFF`), nearest-line assignment and line numbers corrected elsewhere are also available. `ext_robust.trial_report` flags trials on which the methods disagree, and `gen.draw_line_assignment` plots them for inspection.
4. **Reading measures.** `cal.cal_write_em` computes first-fixation duration, gaze duration, regression-path duration, re-reading and total time and other word-level measures, and `cal.cal_write_sm` computes sentence-level measures.

Documentation: <https://gtojty.github.io/pyemread/>

## Installation

```bash
pip install git+https://github.com/gtojty/pyemread.git
```

Python 3.9 or later. The dependencies (NumPy, pandas, Pillow, matplotlib) are installed automatically; they are listed in `pyproject.toml` and `requirements.txt`. FFmpeg is needed only for animations. pyemread is not distributed on PyPI.

## Quick start

The example data in `examples/oralReading/` are EyeLink recordings of three participants (`1950138`, `1950149`, `1950168`) reading three English passages aloud (Gong & Shuai, 2023). Clone the repository to use them:

```bash
git clone https://github.com/gtojty/pyemread.git
cd pyemread
pip install -e ".[test]"
python examples/example_workflow.py      # the complete tutorial
pytest                                   # the test suite
```

The core of the tutorial:

```python
from pyemread import cal, ext, ext_robust

regions = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]
# expects data/story0*.region.csv and data/1950138/1950138.asc
sac, crl_sac, fix, crl_fix = ext.process_subject("data", "1950138", regions)
report = ext_robust.trial_report(fix, ext.load_regions("data", regions))
print(report[["trial_id", "eye", "agree_slice_diff", "flag"]])   # trials to inspect
em = cal.cal_write_em("data", "1950138", regions)                 # word-level measures
```

`examples/tutorial.ipynb` runs the same steps as a notebook (also on Google Colab), and `examples/bids_example/` shows the route for recordings from other trackers.

## Line-assignment accuracy

`validation/` contains the scripts that evaluate the line-assignment methods on the 48 human-corrected trials of Carr et al. (2022) and on simulated reading, and that compare the sample-based event detector with the EyeLink parser on the example data. The results are reported in the accompanying article.

## Modules

| Module (alias) | Purpose |
|---|---|
| `gen` (`stimuli`) | stimulus bitmaps, region files, plots, animations |
| `ext` (`eyelink`) | EyeLink ASC parsing; `process_subject`, `classify_fixations` |
| `ext_generic` (`gaze`) | fixation tables, Eye-Tracking-BIDS, raw samples, epoching; experimental Tobii Pro Lab and Pupil Labs readers |
| `lines` | line-assignment methods and word assignment |
| `ext_robust` (`diagnostics`) | trial-level diagnostics, pre-scan detection |
| `cal` (`measures`) | word- and sentence-level measures |

Names from earlier versions (e.g. `ext.cal_crlSacFix`, `cal.cal_write_EM`, `gen.Praster`) remain available as aliases. See [CHANGELOG.md](CHANGELOG.md).

## Citation

Please cite the accompanying article (Gong, under review, *Behavior Research Methods*) and the software archive: Gong, T. pyemread (Version 2.2.0) [Computer software]. https://doi.org/10.5281/zenodo.19622178. See [CITATION.cff](CITATION.cff) and [AUTHORS.md](AUTHORS.md).

## License

MIT; see [LICENSE](LICENSE). The `slice` implementation is adapted from https://github.com/jwcarr/drift (CC BY 4.0).
