API reference
=============

Each module can also be imported under a descriptive alias:
``stimuli`` (gen), ``eyelink`` (ext), ``gaze`` (ext_generic),
``diagnostics`` (ext_robust) and ``measures`` (cal). Names used in earlier
versions (e.g. ``ext.cal_crlSacFix``, ``cal.cal_write_EM``,
``gen.Praster``) remain available as aliases.

.. autosummary::
   :toctree: generated
   :recursive:

   pyemread.gen
   pyemread.ext
   pyemread.ext_generic
   pyemread.lines
   pyemread.ext_robust
   pyemread.cal
