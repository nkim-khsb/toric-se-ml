"""Re-score the (4,3), (5,4), (7,3) Abreu deviations on the 240x240 grid.

WHY.  Section 3.3 of the paper now quotes grid worst/mean pairs, from
ypq_reproduce_and_grid.log beside this file (the *.log rule in .gitignore has an
exception for it, since it is the data behind two published columns):
    Y^{2,1} deg-8   0.0151% worst / 0.00141% mean   (MC-400 max was 0.0059%)
    Y^{3,2} deg-12  0.0078% worst / 0.00048% mean   (MC-400 max was 0.0033%)
Section 3.4's five-member table still carries MC-400 sample maxima for all five
rows, so it contradicts section 3.3 on the two shared targets and mixes
estimators on the other three.  This script supplies the missing three.

NO CHECKPOINTS EXIST for these three, and reconstructing psi from a bare
coefficient vector does not work in any case: whiten_poly's basis is defined ON
the training sample, so the vector alone does not determine a function (see the
docstring of revisions_Hoseob/experiments/ypq_reproduce_and_grid.py, which
measured held-out 5e-2 instead of 6e-13 when that was attempted).  We therefore
retrain with sweep_extreme_lambda.train_case, which is the same configuration
that produced the table, and report the MC-400 maximum alongside so that the
reproduction of the published number is visible before the new one is believed.

PRE-REGISTERED (fixed before running):
  (0) the MC-400 max must reproduce the paper's value for each member to the
      digits the paper quotes; if it does not, the retrain is not the same
      solution and nothing below is reported.
  (1) grid max is expected to exceed the MC max by a factor 1.5-3, matching the
      2.0-2.6 measured on dP3, Y^{2,1} and Y^{3,2}.
  (2) grid mean is expected to be stable to three digits across n=120/240/360;
      the max is not expected to settle (it chases a boundary supremum).

MEMORY.  abreu_grid vmaps the whole n x n grid; at n >= 240 that exhausted
memory on one machine (fourth-order AD over ~40k points).  A chunked but
otherwise identical estimator is grid_dev in
revisions_Hoseob/experiments/ypq_reproduce_and_grid.py if this dies.

Run:  PYTHONPATH=. python experiments/ypq/abreu_grid_sweep_rescore.py
"""
import sys

import jax, numpy as np, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sugrasol.laplacian import abreu_grid
from sugrasol.ypq import gauge_fixed, ortho_psi, sample_slice
from experiments.ypq.sweep_extreme_lambda import train_case

PAPER_MC = {(4, 3): 0.042, (5, 4): 0.033, (7, 3): 5.7e-5}  # percent, paper table 3

# optional CLI filter, e.g. "7,3", so one member can be re-measured alone
_only = sys.argv[1] if len(sys.argv) > 1 else None
for (P, Q) in [(4, 3), (5, 4), (7, 3)]:
    if _only and _only != f"{P},{Q}":
        continue
    print(f"\n== Y^{{{P},{Q}}} ==", flush=True)
    r = train_case(P, Q)
    chart = r["chart"]
    psi = gauge_fixed(ortho_psi(r["v"][:r["nc"]], r["powers"], r["W"]), chart.anchors)

    # (0) the estimator section 3.4 used: max over a 400-point Monte-Carlo draw
    ss_eval = sample_slice(jax.random.PRNGKey(11), chart, 400, eps=1e-2)
    from sugrasol.laplacian import abreu_scalar
    S = jax.vmap(abreu_scalar(chart, psi))(ss_eval)
    mc = 100 * float(jnp.max(jnp.abs(S / 12.0 - 1.0)))
    print(f"  deg {r['deg']}  held-out {r['l_te']:.2e}   MC-400 max {mc:.4e}%"
          f"   (paper: {PAPER_MC[(P, Q)]})", flush=True)

    for n in (120, 240, 360):
        g = abreu_grid(psi, chart, n=n)
        print(f"  grid n={n:3d}: max {g['max_pct']:.4e}%  mean {g['mean_pct']:.4e}%"
              f"  [{g['n_points']:6d} pts]  = {g['max_pct']/mc:.2f}x the MC max",
              flush=True)
