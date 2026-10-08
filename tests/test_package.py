"""Installation-level checks: imports, aliases and version metadata."""

import re
from pathlib import Path

import pyemread


def test_modules_and_aliases():
    from pyemread import cal, ext, ext_generic, ext_robust, gen, lines  # noqa: F401
    from pyemread import diagnostics, eyelink, gaze, measures, stimuli
    assert measures is cal and stimuli is gen and eyelink is ext
    assert gaze is ext_generic and diagnostics is ext_robust
    import pyemread.measures as m
    assert m is cal


def test_version_is_consistent():
    toml = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', toml, re.M).group(1) == pyemread.__version__


def test_legacy_names_are_aliases():
    from pyemread import cal, ext, gen
    assert ext.cal_crlSacFix is ext.cal_crl_sac_fix
    assert ext.read_SRRasc is ext.read_srr_asc
    assert cal.cal_write_EM is cal.cal_write_em
    assert callable(gen.Praster)
