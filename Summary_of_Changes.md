# pyemread v2.1 — Summary of Changes (revision 2)

This document summarises everything that changed relative to the v2.0 code and the manuscript reviewed at *Behavior Research Methods* (ID BR-Org-25-1119), *including* the revisions made in response to the reviewer-style quality check that followed the first SoftwareX draft.

---

## 1. Rationale

The BRM review produced a rejection with a constructive recommendation to re-submit to an open-software venue. A self-reviewer pass on the first SoftwareX draft then identified twelve items to tighten the submission. The **second revision** (current release) addresses all of those except two (the Sphinx-HTML API site and a unit-test folder, which are deferred to the GitHub release workflow).

See `SoftwareX_Reviewer_Report.md` for the full reviewer checklist and `Pyemread_Response_to_Reviewers.docx` for the point-by-point response to BRM.

---

## 2. What changed in this revision

### 2.1 Title and abstract — **novelty foregrounded**

The title now reads *"pyemread: a Python package for multi-line text reading eye-movement experiments with a **temporally-informed cross-line classifier** and tracker-agnostic gaze input."* The abstract opens with the classifier as the methodological contribution, not with a capability list, and cites the 96.2 % / 93.8 % accuracy numbers from the Gong and Shuai (2023) validation.

### 2.2 Section 1 (Motivation) — **reframed around the classifier**

Now contrasts geometric correction methods (Cohen 2013; Carr et al. 2022 *eyekit*) with pyemread's temporal approach. Adds a domain-independence sentence: the core primitive is "AOI-aligned fixation-sequence classification" and is applicable to code reading, sheet-music reading, L2 assessment, clinical dyslexia screening, etc.

### 2.3 Figure 3 — **real data**

Re-rendered using real EyeLink ASC data from subject 1950138 reading story01 (right eye, 207 fixations). Twelve cross-line moves are cleanly identified (11 forward return sweeps plus 1 backward re-read).

### 2.4 Figure 5 — **real-data sensitivity analysis**

Rebuilt using 1,195 pooled right-eye fixations from three subjects (1950138, 1950149, 1950168) × two passages (story01, story02), all from the Gong and Shuai (2023) corpus.

| Statistic | Value |
|---|---|
| Total fixations | 1,195 |
| κ at the default (0.60, 0.20) | 1.00 |
| κ range across the 9 × 5 grid | 0.57 – 1.00 |
| Fraction of grid with κ ≥ 0.80 | 67 % |
| Recommended diff_ratio (data-driven) | 0.68 |
| Recommended frontrange_ratio | 0.25 |

A notable new finding: `diff_ratio` is the dominant parameter on this corpus; `frontrange_ratio` is essentially inert because backward cross-line moves are rare.

### 2.5 Table 1 — **competitive comparison**

New table in §4 comparing `pyemread` to `eyekit` (Carr et al. 2022), `saccades` (von der Malsburg 2015), SR DataViewer, and PsychoPy+pandas on eight feature dimensions.

### 2.6 Code metadata table — **three gaps closed**

- **C3** now contains a Zenodo DOI placeholder (`https://doi.org/10.5281/zenodo.pyemread-2-1-0`, to be minted on acceptance) rather than "N/A".
- **C6** lists pinned minimum versions: `pandas ≥ 1.3`, `NumPy ≥ 1.20`, `Pillow ≥ 9.0`, `matplotlib ≥ 3.4`, `MoviePy ≥ 1.0.3`, `FFmpeg ≥ 4.0`.
- **C8** now cites `API_REFERENCE.md` (shipped in the repository root) rather than only `README.md`.

### 2.7 Benchmark numbers

Added to §2.2: ASC parsing ≈ **0.1 s** per subject; 30-minute 1 kHz raw-gaze trace processed by I-VT in **≈ 2.2 s** (≈ 0.8 M samples/s).

### 2.8 Numbered code listings

Three inline code blocks are now labelled **Listing 1**, **Listing 2**, **Listing 3** and can be cross-referenced from the body text, following SoftwareX convention.

### 2.9 Expanded reference list

The reference list grows from 6 to 10 items. New citations:

- Carr et al. (2022) — *eyekit*, the most direct competitor.
- Kassner, Patera & Bulling (2014) — Pupil Labs.
- Papoutsaki et al. (2016) — WebGazer.
- SR Research Data Viewer (2021) — commercial baseline.

### 2.10 Wrap-up-effects clarification

The §2.2 "Metric computation" paragraph now explicitly states that `spurt` and `spcount` preserve end-of-sentence and end-of-paragraph wrap-up effects; only whole-passage re-scans are excluded. This closes a concern raised by BRM Reviewer #1 that was previously addressed only in the response letter.

### 2.11 Code-listing rendering fix

Each code listing is now rendered as one shaded paragraph per line rather than one shaded paragraph with embedded `\n` characters that Word treats as soft wraps. All three listings render correctly.

### 2.12 Honest labelling of synthetic data

Figure 4 is now explicitly labelled as a synthetic 1 kHz trace with known ground-truth event boundaries, and the caption notes that real-tracker validation on Tobii / SMI / Pupil Labs / WebGazer is planned for v2.2. This trades a stronger but unsupported claim for a weaker but defensible one, at the request of the reviewer-style check.

---

## 3. New files in the repository

| File | Purpose |
|---|---|
| `API_REFERENCE.md` | 428-line complete API reference (new) |
| `asc_parser.py` | Lightweight Python-3 ASC parser for Figure 3 / 5 regeneration (new) |
| `figures/fig3_emovements.png` | **Regenerated from real EyeLink data** |
| `figures/fig5_sensitivity.png` | **Regenerated from real EyeLink data** |

## 4. Code modules (no changes from revision 1)

- `ext_generic.py` — raw-gaze support (unchanged from rev 1)
- `ext_robust.py` — parameter tuning (unchanged from rev 1)
- `__init__.py` — package init (unchanged from rev 1)
- `gen.py`, `ext.py`, `cal.py` — unchanged from v2.0

## 5. Word-count audit

| Metric | Value |
|---|---|
| Total extracted text (including title, affiliations, metadata table, references) | 3,282 words |
| SoftwareX-countable words (excluding title/authors/metadata table/references) | **2,907 words** |
| SoftwareX limit | 3,000 words |
| **Margin under limit** | **93 words** |

| Metric | Value |
|---|---|
| Figures | 6 (at the SoftwareX limit) |
| Tables | 2 (metadata table + Table 1) |
| Code listings | 3 (numbered) |

## 6. Items *not* addressed in this revision, and why

| Item | Why deferred |
|---|---|
| Ship a `tests/` folder | Belongs on the GitHub release workflow, not the paper. Unit tests exist as smoke tests; formal `pytest` organisation would take a full day. |
| Sphinx HTML API site | `API_REFERENCE.md` in the repo root is the equivalent at a fraction of the effort. Can be added post-publication as a GitHub Pages site. |
| Real-tracker (Tobii, Pupil Labs, WebGazer) validation of `ext_generic` | Requires actual non-EyeLink data not currently available. Figure 4 now explicitly discloses that it uses synthetic 1 kHz data; planned as v2.2. |

## 7. Deliverables

| File | Role |
|---|---|
| `Pyemread_SoftwareX_Manuscript_v2.docx` | Revised main paper (2 KB figures, 2,907 countable words, 6 figures, 2 tables, 3 listings) |
| `Pyemread_SoftwareX_Supplement.docx` | Appendices A–E (unchanged from rev 1) |
| `Pyemread_SoftwareX_Cover_Letter.docx` | Cover letter (unchanged from rev 1) |
| `Pyemread_Response_to_Reviewers.docx` | Point-by-point response to BRM (unchanged from rev 1) |
| `SoftwareX_Reviewer_Report.md` | Self-reviewer quality check that drove this revision |
| `API_REFERENCE.md` | New — complete API reference, cited in metadata C8 |
| `README.md` | Unchanged from earlier update; remains accurate |
| `asc_parser.py` | New — Python-3 ASC parser used to regenerate Figures 3 and 5 |
| `ext_generic.py`, `ext_robust.py`, `__init__.py` | Python modules new in v2.1 |
| `figures/fig1–fig6` | 300 dpi PNG source files, Figures 3 and 5 regenerated from real data |
| `Summary_of_Changes.md` | This document |
