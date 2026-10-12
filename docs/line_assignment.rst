Line assignment
===============

Vertical drift and the geometry of multi-line text make it necessary to
assign each fixation to the line being read before word-level measures can
be computed. :mod:`pyemread.lines` provides three methods:

``"slice"`` (default)
    The slice algorithm of Glandorf and Schroeder (2021), adapted from the
    Python implementation of Carr et al. (2022).
``"diff"``
    The original pyemread heuristic (Gong & Shuai, 2023), which counts line
    changes in temporal order from horizontal displacements.
``"nearest"``
    Nearest line center (no drift correction).

Line numbers corrected in another program can be used with
``classify_method="PROVIDED"``, and the method can be chosen per trial with
a dictionary, e.g. ``{(1, "R"): "DIFF", "default": "SLICE"}``.

Because the methods fail in different situations,
:func:`pyemread.ext_robust.trial_report` flags trials on which ``"slice"``
and ``"diff"`` disagree on more than 10% of fixations;
:func:`pyemread.gen.draw_line_assignment` plots both assignments for
inspection. The ``validation/`` folder of the repository contains the scripts
that evaluate the methods against the human-corrected data of Carr et al.
(2022) and on simulated reading.
