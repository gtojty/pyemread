# -*- coding: utf-8 -*-
"""
pyemread: multi-line text reading eye-movement experiments in Python.

Modules (descriptive alias in brackets):

``gen`` (``stimuli``)
    Stimulus bitmaps and word-level region (AOI) files for alphabetic and
    East Asian scripts; plots and animations of recorded eye movements.
``ext`` (``eyelink``)
    EyeLink ASC parsing, line and word assignment, and the one-call
    pipeline :func:`pyemread.ext.process_subject`.
``ext_generic`` (``gaze``)
    Input from other trackers: fixation tables, Eye-Tracking-BIDS
    recordings, raw samples with I-VT/I-DT event detection, and epoching
    of continuous recordings.
``lines``
    Fixation-to-line assignment methods (``"slice"``, ``"diff"``,
    ``"nearest"``) and word assignment.
``ext_robust`` (``diagnostics``)
    Trial-level diagnostics of line assignment and pre-scan detection.
``cal`` (``measures``)
    Word-level and sentence-level reading measures.

Both names can be used, e.g. ``from pyemread import measures`` is the same
module as ``from pyemread import cal``.
"""

import sys as _sys

__version__ = "2.2.1"
__author__ = "Tao Gong"
__email__ = "gtojty@gmail.com"
__license__ = "MIT"

from . import lines, ext, ext_generic, ext_robust, cal, gen  # noqa: E402

_ALIASES = {"stimuli": gen, "eyelink": ext, "gaze": ext_generic,
            "diagnostics": ext_robust, "measures": cal}
for _name, _mod in _ALIASES.items():
    globals()[_name] = _mod
    _sys.modules[f"{__name__}.{_name}"] = _mod

__all__ = ["gen", "ext", "ext_generic", "lines", "ext_robust", "cal",
           "stimuli", "eyelink", "gaze", "diagnostics", "measures", "__version__"]
