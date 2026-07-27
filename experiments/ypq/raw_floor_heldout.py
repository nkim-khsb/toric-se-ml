"""Held-out loss AT the raw-monomial conditioning plateau on Y^{3,2}.

Why this exists (review-paper1-20260727.md B1).  The non-existence diagnostic of
the paper rests on a contrast between two kinds of plateau:

  * conditioning artifact (Y^{3,2}, raw monomials): train and held-out sit
    TOGETHER at the floor, and both drop together once the basis is fixed;
  * genuine non-existence (dP2 KE-attempt): held-out is pinned while train pulls
    away from it by overfitting.

The 2026-07-12 raw runs recorded only the TRAIN loss (log.md:199, 6.4e-8 at
degree 10 and 7.0e-8 at degree 12), so the "together" half of the contrast was
an inference, not a measurement.  It needs no retraining: the raw checkpoints
ckpt_y32_deg{10,12}.npy are the stalled minimizers themselves, so we simply
evaluate the SAME loss on the held-out sample.

Sampling reproduces the original raw runs (Y^{3,2}: 1024 points, reproducing the
logged train losses 6.4e-8 / 7.0e-8 to two figures; Y^{2,1}: 4096, matching the
orthonormalized run it is compared against), train PRNGKey(1); the
held-out key is the paper's PRNGKey(7) (PRNGKey(2), used by the older
train_y21.py, is reported too -- the conclusion is insensitive to the choice).

One provenance note.  The Y^{2,1} raw checkpoint has been resumed by later L-BFGS
restarts since the 2026-07-12 log entry, so it now sits at 1.5e-9 rather than the
4.3e-9 recorded then.  The paper quotes the checkpoint, which is what a reader
can rerun; the crawl from 4.3e-9 to 1.5e-9 over thousands of extra iterations is
itself the ill-conditioning, not a contradiction of it.

Usage: PYTHONPATH=. python experiments/ypq/raw_floor_heldout.py
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import b_ypq, ypq
from sugrasol.ypq import (gauge_fixed, loss_fn, poly_psi, sample_slice,
                          slice_chart)

print(f"  {'target':>8} {'deg':>4} {'n':>5} {'params':>7} {'train':>11} "
      f"{'held-out':>11} {'held/train':>11} {'alt held-out':>13}")
for P, Q, DEG, NSAMP in ((3, 2, 10, 1024), (3, 2, 12, 1024), (2, 1, 8, 4096)):
    chart = slice_chart(ypq(P, Q), b_ypq(P, Q))
    ss = sample_slice(jax.random.PRNGKey(1), chart, NSAMP, eps=2e-3)      # train
    ss_te = sample_slice(jax.random.PRNGKey(7), chart, NSAMP, eps=2e-3)   # held-out
    ss_te2 = sample_slice(jax.random.PRNGKey(2), chart, NSAMP, eps=2e-3)  # alt.
    v = np.load(f"experiments/ypq/ckpt_y{P}{Q}_deg{DEG}.npy")
    nc = len(v) - 1
    psi = gauge_fixed(poly_psi(jnp.asarray(v[:nc]), DEG), chart.anchors)
    c = float(v[nc])
    l_tr = float(loss_fn(chart, psi, c, ss))
    l_te = float(loss_fn(chart, psi, c, ss_te))
    l_te2 = float(loss_fn(chart, psi, c, ss_te2))
    print(f"  Y^{{{P},{Q}}} {DEG:4d} {NSAMP:5d} {nc:7d} {l_tr:11.3e} {l_te:11.3e} "
          f"{l_te / l_tr:11.2f} {l_te2:13.3e}")

print("\n  Reference points for the contrast:")
print("    Y^{3,2} degree 12, ORTHONORMALIZED: train 2.0e-14, held-out 2.0e-14")
print("    Y^{2,1} degree  8, ORTHONORMALIZED: train 5.6e-13, held-out 6.1e-13")
print("    dP2 KE-attempt degree 16 (no solution): train 1.0e-2, held-out 4.2e-2")
print("  => at a conditioning plateau held/train = O(1); at the dP2 obstruction")
print("     the same ratio is ~4 and grows with the parameter count.")
