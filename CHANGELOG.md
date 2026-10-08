# Changelog

## 2.2.0 (2026)

Installation and maintenance
- The modules are now in a `pyemread/` package directory, so `pip install git+https://github.com/gtojty/pyemread.git` installs importable code (earlier releases installed metadata only). Packaging uses `pyproject.toml`; `requirements.txt` lists the runtime dependencies. pyemread is not on PyPI.
- Compatible with pandas 1.5–3.x and Python 3.9–3.13; tested in continuous integration, including the complete tutorial on the example data.
- API documentation is generated from the docstrings with Sphinx; `API_REFERENCE.md` and `Summary_of_Changes.md` were removed.
- Missing input files raise `FileNotFoundError` with the expected path instead of printing a message; progress messages use the `logging` module.

Line assignment (`pyemread.lines`, `ext.classify_fixations`)
- New default method `SLICE` (Glandorf & Schroeder, 2021; adapted from Carr et al., 2022). The original heuristic remains available as `DIFF` and `SAC`; `NEAREST` and `PROVIDED` (line numbers corrected elsewhere) were added. The method can be chosen per trial.
- Trials on which `SLICE` and `DIFF` disagree on more than 10% of fixations are flagged (`ext_robust.trial_report`, `gen.draw_line_assignment`).
- `ext.process_subject` reads, classifies and writes a participant's data in one call. `ext.cal_crl_sac_fix(..., write=True)` writes its results; previously the classified tables were returned but never written, so `cal.cal_write_EM` read unclassified fixations and returned empty measures.

Reading measures (`pyemread.cal`)
- `cal_write_em` raises an error if the fixations have not been assigned to lines, and accepts in-memory tables (`fix_df=`, `sac_df=`).
- Fixed a `KeyError` when a trial ended on the first fixation on a word.
- First pass now follows the standard definition: a word fixated only after the eye had moved past it has no first pass (`strict_first_pass=False` restores the earlier behaviour). Brief glances at later lines are ignored.
- `rpcount` now includes the first-pass fixations, consistent with `rpurt`; `fpregchr`, `rpregchr` and `tffixos` count characters from the beginning of the text (off-by-one errors fixed); `tffixurt` was written as `ttfixurt` and is fixed.
- New word measures `tfurt` and `tfcount` (total fixation time and count) and sentence-level measures (`cal_write_sm`).

Input from other trackers (`pyemread.ext_generic`)
- New readers for fixation tables (`read_fixations`), Eye-Tracking-BIDS recordings (`read_bids_eyetrack`, `bids_fixations`), and experimental readers for Tobii Pro Lab fixations and samples and Pupil Labs Core and Neon surface-mapped fixations and gaze. `smi_to_raw` and `webgazer_to_raw` were removed.
- `detect_events` requires explicit threshold units, estimates velocity with a Savitzky–Golay filter, interpolates short gaps, splits at long gaps, merges nearby fixations, applies a minimum duration and returns tables in the pyemread schema.
- `epoch` and `trials_from_messages` cut continuous recordings into trials.
- `write_events` writes to `<direct>/<subj_id>/`, where the rest of the pipeline reads.

Diagnostics (`pyemread.ext_robust`)
- `trial_report` replaces `recommend_diff_ratio` and `reclassify_orphans`, which were removed.

Stimuli and animations (`pyemread.gen`)
- `praster` creates its output folder. `animate` finds the bitmap and audio of a trial by name, and videos are written to the participant's folder. MoviePy is no longer used.

## 2.1.1 and earlier

See the release notes on GitHub and Zenodo (https://doi.org/10.5281/zenodo.19622178).
