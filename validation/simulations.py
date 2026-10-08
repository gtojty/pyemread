"""Line-assignment accuracy on simulated reading with known ground truth.

The simulations reuse the ``ReadingScenario`` class of Carr et al. (2022,
``code/simulation.py``) unchanged. A random passage of 8 to 12 lines is
generated, and one fixation per word is placed along each line, with
optional within-line and between-line regressions. The fixations are then
distorted by one of five factors, varied one at a time over the ranges
used by Carr et al.:

* ``noise`` - Gaussian vertical noise (SD 0 to 40 px; line spacing 64 px);
* ``slope`` - a within-line slope (-0.1 to 0.1);
* ``shift`` - a vertical shift proportional to line position (-0.2 to 0.2);
* ``regression_within`` - probability of a within-line regression after
  each word (0 to 1);
* ``regression_between`` - probability of a regression to an earlier line
  on each line (0 to 1).

Every simulated trial is corrected by Carr et al.'s algorithms and by
pyemread's ``diff`` and ``slice`` methods. Accuracy is the percentage of
fixations assigned to the line on which they were generated.

Output: ``results/simulations.csv``.

Usage: ``python simulations.py [n_sims_per_level]`` (default 50).
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np
import pandas as pd
import eyekit

from common import (CARR_ALGORITHMS, DRIFT, RESULTS, install_carr_algorithms,
                    passage_to_region, run_pyemread)

FACTORS = {
    "noise": np.linspace(0, 40, 9),
    "slope": np.linspace(-0.1, 0.1, 9),
    "shift": np.linspace(-0.2, 0.2, 9),
    "regression_within": np.linspace(0, 1, 9),
    "regression_between": np.linspace(0, 1, 9),
}


def main(n_sims: int = 50, seed: int = 2026):
    # eyekit >= 0.6 no longer provides TextBlock.word_centers(), used by
    # Carr et al.'s 'compare' and 'warp' implementations.
    if not hasattr(eyekit.TextBlock, "word_centers"):
        eyekit.TextBlock.word_centers = lambda self: [w.center for w in self.words()]
    sys.path.insert(0, str(DRIFT / "code"))
    # Carr et al.'s core.py imports cairosvg for figure export only.
    import types
    sys.modules.setdefault("cairosvg", types.ModuleType("cairosvg"))
    algorithms = install_carr_algorithms()
    from simulation import ReadingScenario  # Carr et al.'s simulator

    rng_seed = seed
    rows = []
    t_start = time.time()
    for factor, levels in FACTORS.items():
        for level in levels:
            np.random.seed(rng_seed)
            random.seed(rng_seed)          # lorem (passage text) uses the random module
            rng_seed += 1
            scenario = ReadingScenario(**{factor: float(level)})
            for sim in range(n_sims):
                passage, xy, truth = scenario.simulate()
                truth = np.asarray(truth) + 1
                reg = passage_to_region(passage)
                for a in CARR_ALGORITHMS:
                    lab = algorithms.correct_drift(a, xy, passage, return_line_assignments=True)
                    rows.append((factor, level, sim, a, 100 * np.mean(np.asarray(lab) + 1 == truth)))
                for m in ("diff", "slice"):
                    lab = run_pyemread(m, xy[:, 0], xy[:, 1], reg)
                    rows.append((factor, level, sim, f"pyemread_{m}", 100 * np.mean(lab == truth)))
            print(f"{factor}={level:.3f} done ({time.time() - t_start:.0f} s)", flush=True)
    out = pd.DataFrame(rows, columns=["factor", "level", "sim", "method", "accuracy"])
    out.to_csv(RESULTS / "simulations.csv", index=False)
    print(out.groupby(["factor", "method"])["accuracy"].mean().unstack(0).round(1).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
