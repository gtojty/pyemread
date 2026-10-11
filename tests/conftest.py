"""Shared fixtures: copies of the example data in a temporary directory."""

import logging
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "oralReading"
REGIONS = ["story01.region.csv", "story02.region.csv", "story03.region.csv"]
SUBJECT = "1950138"


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("pyemread").setLevel(logging.ERROR)


@pytest.fixture
def eyelink_dir(tmp_path):
    """<tmp>/story0*.region.csv and <tmp>/1950138/1950138.asc."""
    for r in REGIONS:
        shutil.copy(EXAMPLES / r, tmp_path / r)
    (tmp_path / SUBJECT).mkdir()
    shutil.copy(EXAMPLES / SUBJECT / f"{SUBJECT}.asc", tmp_path / SUBJECT)
    return tmp_path


def cjk_font(name):
    """Path of a CJK font that is not distributed with pyemread (e.g. 'simsun.ttc'
    or 'batang.ttc', installed with Windows), or None if it cannot be found.
    Set PYEMREAD_CJK_FONTS to a folder that contains the fonts to use another
    location."""
    import os
    folders = [os.environ.get("PYEMREAD_CJK_FONTS", ""), r"C:\Windows\Fonts",
               str(ROOT / "examples" / "textRasters")]
    for folder in folders:
        if folder and (Path(folder) / name).exists():
            return str(Path(folder) / name)
    # a free alternative (e.g. Debian/Ubuntu package fonts-noto-cjk)
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return path
    return None
