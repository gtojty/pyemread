"""Edge cases found in review: partial reading, trials without line changes,
binocular validity, other input routes, CJK input without spaces."""

import numpy as np
import pandas as pd
import pytest

from pyemread import cal, ext, ext_generic as eg, gen, lines
from conftest import EXAMPLES, ROOT, SUBJECT, cjk_font
from test_lines import passage
from test_measures import em_for, region

RASTERS = ROOT / "examples" / "textRasters"


def _lines_read(reg, n_read, per_line=8):
    geo = lines.text_geometry(reg)
    xs = np.linspace(reg.x1_pos.min() + 20, reg.x2_pos.max() - 20, per_line)
    fix = pd.DataFrame({"x_pos": np.tile(xs, n_read),
                        "y_pos": np.repeat(geo.centers[:n_read], per_line)})
    return fix, np.repeat(geo.line_ids[:n_read], per_line)


@pytest.mark.parametrize("n_read", [1, 2, 3, 8])
def test_slice_when_only_some_lines_are_read(n_read):
    reg = passage(n_lines=8)
    fix, truth = _lines_read(reg, n_read)
    out = lines.assign_lines(fix, reg, method="slice")
    assert (out["line_no"].to_numpy() == truth).all()


def test_diff_when_no_line_change_is_detected():
    reg = passage(n_lines=8)
    fix, truth = _lines_read(reg, 1)
    out = lines.assign_lines(fix, reg, method="diff")
    assert (out["line_no"].to_numpy() == truth).all()


def _one_line_trial(eye="R"):
    reg = passage(n_lines=4)
    fix, _ = _lines_read(reg, 1)
    n = len(fix)
    fix = fix.assign(subj="S", trial_id=1, trial_type="p", eye=eye,
                     start_time=np.arange(n) * 250.0, end_time=np.arange(n) * 250.0 + 200,
                     duration=200.0, valid="yes", line_no=np.nan, region_no=np.nan)
    return reg, fix


@pytest.mark.parametrize("method", ["DIFF", "SLICE", "NEAREST"])
def test_pipeline_does_not_crash_on_a_one_line_trial(method):
    reg, fix = _one_line_trial()
    sac, crl_sac, out, crl_fix, flagged = ext.classify_fixations(
        pd.DataFrame(columns=ext.SAC_COLUMNS), fix, {"p": reg}, method)
    assert (out["line_no"] == 1).all()
    assert len(crl_fix) == 0


def test_last_fixation_of_each_eye_is_invalid(eyelink_dir):
    _, _, fix, _ = ext.process_subject(str(eyelink_dir), SUBJECT,
                                       ["story01.region.csv", "story02.region.csv",
                                        "story03.region.csv"])
    for (_, eye), g in fix.groupby(["trial_id", "eye"]):
        assert g["valid"].iloc[-1] == "no", eye


def test_generic_route_uses_the_same_validity_rules():
    fix = pd.DataFrame({"trial_id": 1, "eye": "R", "x_pos": [100, 110, 300, 500],
                        "y_pos": 100, "start_time": [0, 210, 260, 600],
                        "end_time": [200, 250, 580, 800], "duration": [200, 40, 320, 200],
                        "valid": "yes", "sampfreq": 250})
    out = ext.apply_fixation_validity(fix)
    assert len(out) == 3                      # the 40-ms fixation is merged with its neighbor
    assert out["valid"].tolist() == ["yes", "yes", "no"]   # the last fixation is invalid


def test_first_pass_survives_an_initial_glance_at_a_later_line():
    reg = region(words_per_line=(5, 5, 5))
    # two fixations on line 3 (e.g. a central fixation target), then reading of line 1
    em = em_for([200, 220, 120, 160, 200, 240, 280], [200] * 7, reg=reg,
                ys=[220, 220, 100, 100, 100, 100, 100])
    assert em.loc[[1, 2, 3, 4, 5], "fpurt"].notna().all()


def test_detector_error_names_the_parameters_in_degrees():
    smp = pd.DataFrame({"time": np.arange(100) * 4.0, "x": 100.0, "y": 100.0,
                        "eye": "R", "trial_id": 1})
    with pytest.raises(ValueError, match="dispersion_unit"):
        eg.detect_events(smp, velocity_threshold=1200, velocity_unit="px/s")


def test_praster_strips_byte_order_mark(tmp_path):
    from matplotlib import font_manager
    font = font_manager.findfont("DejaVu Sans Mono")
    gen.praster(str(tmp_path), font, "TopLeft", "English",
                text=["﻿The quick brown fox", "jumps over the dog."], ID="bom")
    reg = pd.read_csv(tmp_path / "bom.region.csv")
    assert reg["Word"].iloc[0] == "The" and reg["length"].iloc[0] == 3


@pytest.mark.skipif(cjk_font("simsun.ttc") is None or cjk_font("batang.ttc") is None,
                    reason="CJK fonts not available")
@pytest.mark.parametrize("lang", ["Japanese", "Korean"])
def test_cjk_without_spaces_gives_one_aoi_per_character(tmp_path, lang):
    text = ["今日は良い天気です。", "明日も晴れる。"] if lang == "Japanese" else ["오늘은좋은날", "내일도맑음。"]
    font = cjk_font("simsun.ttc" if lang == "Japanese" else "batang.ttc")
    gen.praster(str(tmp_path), font, "TopLeft", lang, text=text, ID="cjk")
    reg = pd.read_csv(tmp_path / "cjk.region.csv")
    n_punct = sum(t.count("。") for t in text)
    assert len(reg) == sum(len(t) for t in text) - n_punct   # full stops join their character


def test_prescan_before_reading_is_detected_and_masked():
    from pyemread import ext_robust
    reg = passage(n_lines=8)
    geo = lines.text_geometry(reg)
    scan_y = [geo.centers[6], geo.centers[4], geo.centers[2], geo.centers[5]]
    read_x = np.linspace(130, 650, 6)
    fix = pd.DataFrame({"trial_id": 1, "eye": "R",
                        "x_pos": [400, 300, 500, 200] + list(read_x),
                        "y_pos": scan_y + [geo.centers[0]] * 6, "valid": "yes"})
    mask = ext_robust.detect_prescan(fix, reg)
    assert mask.tolist() == [True] * 4 + [False] * 6
    assert (ext_robust.mask_prescan_fixations(fix, reg)["valid"][:4] == "no").all()


def test_classify_lines_single_stimulus():
    reg = passage(n_lines=4)
    fix, truth = _lines_read(reg, 4)
    n = len(fix)
    fix = fix.assign(subj="S", trial_id=1, eye="R", start_time=np.arange(n) * 250.0,
                     end_time=np.arange(n) * 250.0 + 200, duration=200.0, valid="yes")
    sac, crl_sac, out, crl_fix = eg.classify_lines(pd.DataFrame(columns=ext.SAC_COLUMNS),
                                                   fix, reg)
    assert (out["line_no"].to_numpy() == truth).all()
    assert out["region_no"].notna().all()
    assert len(crl_fix) == 3


def test_classify_method_keys_are_checked():
    reg, fix = _one_line_trial()
    sac = pd.DataFrame(columns=ext.SAC_COLUMNS)
    # a lowercase eye is accepted
    _, _, out, _, _ = ext.classify_fixations(sac, fix, {"p": reg}, {(1, "r"): "NEAREST"})
    assert (out["line_no"] == 1).all()
    # a key that matches no trial (here a string trial id) is reported
    with pytest.warns(UserWarning, match="matches no trial"):
        ext.classify_fixations(sac, fix, {"p": reg}, {("1", "R"): "DIFF", "default": "SLICE"})
