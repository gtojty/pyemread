"""Line assignment on synthetic reading with known lines."""

import numpy as np
import pandas as pd
import pytest

from pyemread import lines


def passage(n_lines=8, n_words=10, word_w=60, spacing=50, x0=100, y0=100):
    rows, wid = [], 1
    for ln in range(1, n_lines + 1):
        for k in range(n_words):
            rows.append({"WordID": wid, "Word": "word", "length": 4, "line_no": ln,
                         "x1_pos": x0 + k * word_w, "x2_pos": x0 + (k + 1) * word_w,
                         "y1_pos": y0 + (ln - 1) * spacing - 10,
                         "y2_pos": y0 + (ln - 1) * spacing + 10})
            wid += 1
    return pd.DataFrame(rows)


def read(reg, drift_per_fix=0.0, noise=4.0, within=(), between=(), seed=1):
    """One fixation per word; optional long within-line regressions (after
    word index in ``within``) and regressions to the previous line (``between``)."""
    rng = np.random.default_rng(seed)
    geo = lines.text_geometry(reg)
    xs, ys, truth = [], [], []
    for ln_i, y in enumerate(geo.centers, start=1):
        words = reg[reg.line_no == ln_i]
        for k, (_, w) in enumerate(words.iterrows()):
            xs.append((w.x1_pos + w.x2_pos) / 2); ys.append(y); truth.append(ln_i)
            if (ln_i, k) in within:                # back to the start of the line
                xs.append(words.x1_pos.iloc[0] + 10); ys.append(y); truth.append(ln_i)
            if (ln_i, k) in between and ln_i > 1:  # up to the previous line and back
                xs.append(words.x1_pos.iloc[3] + 10); ys.append(y - geo.spacing); truth.append(ln_i - 1)
    n = len(xs)
    ys = np.array(ys) + rng.normal(0, noise, n) + drift_per_fix * np.arange(n)
    return pd.DataFrame({"x_pos": xs, "y_pos": ys}), np.array(truth)


def acc(method, fix, reg, truth):
    out = lines.assign_lines(fix, reg, method=method)
    return np.mean(out["line_no"].to_numpy() == truth)


def test_geometry():
    geo = lines.text_geometry(passage())
    assert geo.spacing == 50 and len(geo.centers) == 8
    assert geo.width == pytest.approx(540)


def test_clean_reading_all_methods():
    reg = passage()
    fix, truth = read(reg)
    for m in ("slice", "diff", "nearest"):
        assert acc(m, fix, reg, truth) > 0.98


def test_slow_drift_slice_and_diff_robust_nearest_not():
    reg = passage()
    fix, truth = read(reg, drift_per_fix=-0.5)       # ~ -40 px (0.8 lines) by the end
    assert acc("slice", fix, reg, truth) > 0.97
    assert acc("diff", fix, reg, truth) > 0.97
    assert acc("nearest", fix, reg, truth) < 0.9


def test_long_within_line_regression():
    reg = passage()
    fix, truth = read(reg, within={(3, 8), (5, 9)})
    assert acc("slice", fix, reg, truth) > 0.97
    assert acc("diff", fix, reg, truth) < acc("slice", fix, reg, truth)


def test_regression_to_previous_line():
    reg = passage()
    fix, truth = read(reg, between={(4, 6)})
    assert acc("slice", fix, reg, truth) > 0.97


def test_regions():
    reg = passage()
    fix, truth = read(reg)
    out = lines.assign_lines_slice(fix, reg)
    out = lines.assign_regions(out, reg)
    assert out["region_no"].notna().all()
    assert (out["region_no"].diff().dropna() == 1).mean() > 0.95


def test_report_flags_disagreement():
    reg = passage()
    # regressions to the line start in mid-line: "diff" counts them as return sweeps
    fix, truth = read(reg, within={(2, 4), (4, 5)})
    assert acc("diff", fix, reg, truth) < 0.8 and acc("slice", fix, reg, truth) > 0.97
    rep = lines.line_assignment_report(fix, reg)
    assert rep["flag"] and rep["agree_slice_diff"] < 0.9


def test_single_line():
    reg = passage(n_lines=1)
    fix = pd.DataFrame({"x_pos": [130, 190, 250], "y_pos": [100, 103, 98]})
    assert (lines.assign_lines(fix, reg)["line_no"] == 1).all()
