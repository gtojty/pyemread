"""End-to-end checks on the EyeLink example data distributed with the package."""

import numpy as np
import pandas as pd
import pytest

from pyemread import cal, ext
from conftest import REGIONS, SUBJECT


@pytest.mark.parametrize("method", ["SLICE", "DIFF"])
def test_process_subject_then_measures(eyelink_dir, method):
    d = str(eyelink_dir)
    sac, crl_sac, fix, crl_fix = ext.process_subject(d, SUBJECT, REGIONS, classify_method=method)
    for suffix in ("_Sac", "_Fix", "_crlSac", "_crlFix"):
        assert (eyelink_dir / SUBJECT / f"{SUBJECT}{suffix}.csv").exists()
    assert fix["line_no"].notna().mean() > 0.95
    assert fix["region_no"].notna().mean() > 0.9
    em = cal.cal_write_em(d, SUBJECT, REGIONS)
    # 3 passages x 2 eyes, one row per word
    n_words = sum(len(pd.read_csv(eyelink_dir / r)) for r in REGIONS)
    assert len(em) == 2 * n_words
    assert set(em["trial_type"]) == {"story01", "story02", "story03"}
    # measures are populated: most words are fixated and many have a first pass
    assert em["tfurt"].notna().mean() > 0.7
    assert em["fpurt"].notna().mean() > 0.5
    assert (em["ffixurt"] <= em["fpurt"]).all() or em["ffixurt"].isna().any()
    ok = em["fpurt"].notna()
    assert (em.loc[ok, "rpurt"] >= em.loc[ok, "fpurt"]).all()
    assert (em.loc[ok, "rpcount"] >= em.loc[ok, "fpcount"]).all()
    for tt in ("story01", "story02", "story03"):
        for eye in ("L", "R"):
            assert (eyelink_dir / SUBJECT / f"{SUBJECT}_EM_{tt}_{eye}.csv").exists()


def test_unclassified_fixations_raise(eyelink_dir):
    d = str(eyelink_dir)
    ext.read_write_srr_asc(d, SUBJECT, "RP")          # parse only, no line assignment
    with pytest.raises(ValueError, match="not been assigned to text lines"):
        cal.cal_write_em(d, SUBJECT, REGIONS)


def test_in_memory_route_matches_written_route(eyelink_dir):
    d = str(eyelink_dir)
    ext.read_write_srr_asc(d, SUBJECT, "RP")
    sac, crl_sac, fix, crl_fix = ext.cal_crl_sac_fix(d, SUBJECT, REGIONS)
    em_mem = cal.cal_write_em(d, SUBJECT, REGIONS, fix_df=fix, sac_df=sac, write=False)
    ext.cal_crl_sac_fix(d, SUBJECT, REGIONS, write=True)
    em_disk = cal.cal_write_em(d, SUBJECT, REGIONS, write=False)
    pd.testing.assert_frame_equal(em_mem.reset_index(drop=True), em_disk.reset_index(drop=True),
                                  check_dtype=False)


def test_missing_subject_is_a_clear_error(eyelink_dir):
    with pytest.raises(FileNotFoundError, match="P01"):
        ext.process_subject(str(eyelink_dir), "P01", REGIONS)


def test_sentence_measures(eyelink_dir):
    d = str(eyelink_dir)
    ext.process_subject(d, SUBJECT, REGIONS)
    sm = cal.cal_write_sm(d, SUBJECT, REGIONS)
    assert len(sm) > 0 and sm["tfurt"].notna().mean() > 0.9
    assert (sm["rpurt"].dropna() >= 0).all()


def test_provided_line_assignment(eyelink_dir):
    """Line numbers corrected elsewhere (e.g. by hand) can be used as they are."""
    d = str(eyelink_dir)
    sac, fix = ext.read_srr_asc(d, SUBJECT, "RP")
    regions = ext.load_regions(d, REGIONS)
    _, _, ref, _, _ = ext.classify_fixations(sac, fix, regions, "DIFF", check=False)
    fix["line_no"] = ref["line_no"].to_numpy()
    _, _, out, _, _ = ext.classify_fixations(sac, fix, regions, "PROVIDED", check=False)
    assert out["line_no"].equals(ref["line_no"])
    assert out["region_no"].notna().sum() == ref["region_no"].notna().sum()
