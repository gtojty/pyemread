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
