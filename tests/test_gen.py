"""Stimulus generation, drawing and animation."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import font_manager

from pyemread import ext, gen
from conftest import EXAMPLES, REGIONS, ROOT, SUBJECT

RASTERS = ROOT / "examples" / "textRasters"
MONO = font_manager.findfont("DejaVu Sans Mono")


def test_praster_english(tmp_path):
    text = [ln.strip() for ln in open(RASTERS / "story01.txt", encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    gen.praster(direct=str(tmp_path), fontpath=MONO, st_pos="TopLeft", lang_type="English",
                text=text, ID="story01")
    reg = pd.read_csv(tmp_path / "story01.region.csv")
    assert (tmp_path / "story01.png").exists()
    assert reg["line_no"].max() == len(text)
    assert len(reg) == sum(len(t.split()) for t in text)
    assert (reg["x2_pos"] > reg["x1_pos"]).all() and (reg["y2_pos"] > reg["y1_pos"]).all()


@pytest.mark.skipif(not (RASTERS / "simsun.ttc").exists(), reason="CJK font not available")
def test_praster_chinese(tmp_path):
    text = [ln.strip() for ln in open(RASTERS / "ChineseStory.txt", encoding="utf-8")
            if ln.strip() and not ln.startswith("#")]
    gen.praster(direct=str(tmp_path), fontpath=str(RASTERS / "simsun.ttc"), st_pos="TopLeft",
                lang_type="Chinese", text=text, ID="zh")
    reg = pd.read_csv(tmp_path / "zh.region.csv")
    assert len(reg) > 10 and reg["line_no"].max() >= 2


def test_draw_and_animate(eyelink_dir):
    d = str(eyelink_dir)
    shutil.copy(RASTERS / "story01.png", eyelink_dir / "story01.png")
    ext.process_subject(d, SUBJECT, REGIONS)
    gen.draw_sac_fix(d, SUBJECT, REGIONS, ["story01.png"], "ALL")
    assert any(p.suffix == ".png" for p in (eyelink_dir / SUBJECT).iterdir())
    if not gen.FFMPEG_AVAILABLE:
        pytest.skip("FFmpeg not installed")
    shutil.copy(EXAMPLES / SUBJECT / f"{SUBJECT}-1.wav", eyelink_dir / SUBJECT)
    out = gen.animate(d, SUBJECT, 1, eye="R")
    assert out is not None and Path(out).exists() and Path(out).stat().st_size > 10_000
