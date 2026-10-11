"""Word- and sentence-level measures on hand-constructed fixation sequences."""

import numpy as np
import pandas as pd
import pytest

from pyemread import cal, lines


def region(words_per_line=(5,), width=40, nchar=4, y0=100, spacing=60):
    rows, wid = [], 1
    for ln, n in enumerate(words_per_line, start=1):
        for k in range(n):
            rows.append({"WordID": wid, "Word": f"w{wid}", "length": nchar, "line_no": ln,
                         "x1_pos": 100 + width * k, "x2_pos": 100 + width * (k + 1),
                         "y1_pos": y0 + spacing * (ln - 1) - 10, "y2_pos": y0 + spacing * (ln - 1) + 10})
            wid += 1
    return pd.DataFrame(rows)


def fixations(xs, ys, durs, reg):
    t = np.cumsum([0] + list(durs[:-1])) + 20 * np.arange(len(durs))
    fix = pd.DataFrame({"subj": "S", "trial_id": 1, "trial_type": "t", "eye": "R",
                        "start_time": t, "end_time": t + np.array(durs), "duration": durs,
                        "x_pos": xs, "y_pos": ys, "valid": "yes"})
    geo = lines.text_geometry(reg)
    fix["line_no"] = lines.assign_lines_nearest(fix, reg)["line_no"]
    return lines.assign_regions(fix, reg)


def em_for(xs, durs, reg=None, ys=None, **kw):
    reg = region() if reg is None else reg
    ys = [100] * len(xs) if ys is None else ys
    return cal.compute_word_measures(fixations(xs, ys, durs, reg), reg, **kw).set_index("region")


def test_first_pass_regression_path_and_second_pass():
    em = em_for([120, 160, 200, 160, 210, 240, 280], [200, 210, 220, 230, 240, 250, 260])
    assert em.loc[1, ["fpurt", "fpcount", "fpregres", "rpurt", "rpcount"]].tolist() == [200, 1, 0, 200, 1]
    assert em.loc[2, "fpurt"] == 210 and em.loc[2, "spurt"] == 230 and em.loc[2, "tfurt"] == 440
    w3 = em.loc[3]
    assert (w3.fpurt, w3.fpregres, w3.fpregreg) == (220, 1, 2)
    assert (w3.rpurt, w3.rpcount, w3.rpregreg) == (690, 3, 2)
    assert w3.spurt == 240 and w3.spilover == 230
    assert em.loc[5, "fpurt"] == 260 and np.isnan(em.loc[5, "spilover"])
    assert em["tfixcnt"].iloc[0] == 7
    assert em["tregrcnt"].iloc[0] == 1
    assert em["tffixurt"].iloc[0] == 200 + 210 + 220 + 250 + 260


def test_character_positions_are_text_relative():
    em = em_for([120, 160, 200, 165, 240], [200] * 5)
    # regression from w3 lands on w2 at x=165 -> character 2 of w2 (0-based),
    # preceded by the 4 characters of w1
    assert em.loc[3, "fpregchr"] == 4 + 2
    assert em.loc[1, "ffos"] == 2          # x=120 in padded [90,140]: ceil(30/50*4) - 1


def test_word_skipped_then_regressed_to_has_no_first_pass():
    xs, d = [120, 200, 160, 240], [200, 210, 220, 230]
    strict = em_for(xs, d)
    assert np.isnan(strict.loc[2, "fpurt"]) and strict.loc[2, "spurt"] == 220
    lenient = em_for(xs, d, strict_first_pass=False)
    assert lenient.loc[2, "fpurt"] == 220


def test_trial_ending_on_first_fixation_of_a_word():
    # This sequence raised a KeyError in pyemread <= 2.1.
    em = em_for([120, 160, 200], [200, 210, 220])
    assert em.loc[3, "fpurt"] == 220 and em.loc[3, "fpregres"] == 0
    assert em.loc[3, "rpurt"] == 220 and np.isnan(em.loc[3, "spilover"])


def test_regression_to_previous_line():
    reg = region(words_per_line=(5, 5))
    xs = [120, 160, 200, 240, 280, 120, 160, 200, 200]
    ys = [100] * 5 + [160, 160, 100, 160]
    em = em_for(xs, [200] * 9, reg=reg, ys=ys)
    w7 = em.loc[7]
    assert (w7.fpregres, w7.fpregreg) == (1, 3)
    assert w7.rpurt == 400 and w7.rpcount == 2
    assert em.loc[3, "spurt"] == 200


def test_unassigned_fixations_raise():
    reg = region()
    fix = fixations([120, 160], [100, 100], [200, 200], reg)
    fix["line_no"] = np.nan
    with pytest.raises(ValueError):
        cal.compute_word_measures(fix, reg)


def test_sentence_measures():
    reg = region()
    reg["Word"] = ["The", "cat.", "A", "big", "dog."]
    assert cal.sentence_ids(reg).tolist() == [1, 1, 2, 2, 2]
    fix = fixations([120, 160, 200, 120, 240, 280], [100] * 6, [200, 210, 220, 230, 240, 250], reg)
    sm = cal.compute_sentence_measures(fix, reg).set_index("sentence_id")
    assert sm.loc[1, "fpurt"] == 410 and sm.loc[1, "spurt"] == 230
    assert sm.loc[2, "fpurt"] == 220 and sm.loc[2, "fpregres"] == 1
    # go-past of sentence 2 runs to the end of the trial (no later sentence)
    assert sm.loc[2, "rpurt"] == 220 + 230 + 240 + 250 and sm.loc[2, "tfurt"] == 220 + 240 + 250


def test_glance_to_a_later_line_does_not_end_first_pass():
    reg = region(words_per_line=(5, 5))
    xs = [120, 240, 160, 200, 240, 280]
    ys = [100, 160, 100, 100, 100, 100]          # fixation 2 is a glance at line 2
    em = em_for(xs, [200] * 6, reg=reg, ys=ys)
    assert em.loc[2, "fpurt"] == 200 and em.loc[3, "fpurt"] == 200
    strict = em_for(xs, [200] * 6, reg=reg, ys=ys, ignore_glances=False)
    assert np.isnan(strict.loc[2, "fpurt"])
