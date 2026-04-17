# -*- coding: utf-8 -*-
"""
pyemread — Python package for multi-line text reading eye-movement experiments.

Modules
-------
gen           : stimulus bitmaps, region files, and visualisations.
ext           : SR-Research (EyeLink) ASC parsing and event extraction.
ext_generic   : tracker-agnostic [t, x, y] raw-gaze processing (I-VT / I-DT);
                direct loaders for Tobii, SMI, Pupil Labs, and WebGazer.
ext_robust    : parameter tuning and edge-case handling for the cross-line
                classifier (pre-scan detection, orphan reclassification,
                sensitivity reports).
cal           : computation of first-pass, regression-path and second-pass
                eye-movement metrics.
"""

__version__ = "2.1.0"
__author__ = "Tao Gong"
__email__ = "gtojty@gmail.com"
__license__ = "MIT"

__all__ = ["gen", "ext", "ext_generic", "ext_robust", "cal"]

# Import sub-modules, but don't let an import failure in one block the others.
# This matters for environments that install the lightweight raw-gaze side of
# pyemread without the ASC-parsing / ffmpeg dependencies.
import warnings as _warnings

for _name in __all__:
    try:
        __import__(f"{__name__}.{_name}")
    except Exception as _exc:                               # pragma: no cover
        _warnings.warn(f"pyemread.{_name} could not be imported: {_exc}")
