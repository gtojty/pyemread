Input formats
=============

All routes produce a *fixation table*: one row per fixation with the columns
in :data:`pyemread.ext.FIX_COLUMNS`. The columns used by line assignment and
the measures are ``trial_id``, ``trial_type`` (the stimulus, which names its
region file ``<trial_type>.region.csv``), ``eye``, ``start_time``,
``end_time`` and ``duration`` (ms), ``x_pos`` and ``y_pos`` (screen pixels,
origin at the top left), and ``valid`` (``"yes"``/``"no"``).

The same validity rules are applied to every route
(:func:`pyemread.ext.apply_fixation_validity`): per trial and eye, fixations
of 50 ms or less are merged with a neighbour within 50 pixels, fixations still
shorter than 50 ms are invalid, and the last fixation is invalid because it
is usually cut short by the end of the trial. For the generic routes, this
step is part of :func:`pyemread.ext_generic.process_fixations` and can be
switched off (``apply_validity=False``).

EyeLink
    :func:`pyemread.ext.process_subject` reads ``<data>/<subj>/<subj>.asc``
    (converted with SR Research's ``edf2asc``). Trials are delimited by the
    ``TRIALID`` and ``TRIAL_RESULT`` messages, and the stimulus of a trial is
    taken from the trial variable named by :data:`pyemread.ext.TRIAL_TYPE_VAR`.

Fixation tables from any software
    :func:`pyemread.ext_generic.read_fixations` with a ``column_map``.
    ``examples/fixation_table`` is an example file; processed with the same
    line-assignment choices, it gives the same word-level measures as the
    EyeLink file it was exported from.

Eye-Tracking-BIDS
    :func:`pyemread.ext_generic.read_bids_eyetrack`,
    :func:`pyemread.ext_generic.bids_fixations`,
    :func:`pyemread.ext_generic.trials_from_messages` and
    :func:`pyemread.ext_generic.epoch`. ``examples/bids_example`` is a BIDS
    version of one example recording.

Raw samples
    :func:`pyemread.ext_generic.read_samples` and
    :func:`pyemread.ext_generic.detect_events` (I-VT, I-DT), for head-fixed,
    screen-based recordings. Thresholds are given in degrees of visual angle
    together with ``px_per_deg``.

Vendor exports (experimental)
    :func:`pyemread.ext_generic.tobii_fixations` (Tobii Pro Lab),
    :func:`pyemread.ext_generic.pupil_core_surface_fixations` (Pupil Core,
    surface-mapped) and :func:`pyemread.ext_generic.neon_surface_fixations`
    (Pupil Labs Neon, Marker Mapper enrichment). These readers follow the
    vendors' documented export formats and are tested on synthetic files
    only; check their output against the vendor software. For head-mounted
    trackers, use the fixations detected by the vendor's software, which
    accounts for head movement.

Continuous recordings
    :func:`pyemread.ext_generic.epoch` cuts samples or fixations into trials
    with a table of trial onsets and offsets, which can be built from the
    recording's messages with :func:`pyemread.ext_generic.trials_from_messages`.
