"""Is the symmetry-free dP3 network's 4e-7 a floor, or an unreleased budget?

Appendix C reports that raising the L-BFGS cap from 6000 to 18,000 moves the
held-out residual from 4.82e-7 to 4.18e-7 and that raising it ninefold to
54,000 "changes nothing at all", the two runs agreeing bit for bit.  They agree
because neither reaches its cap: nn_dp3_nosym.py sets maxiter but not maxfun,
so scipy's default maxfun = 15,000 stops both after 13,739 iterations.  The
lever that was pulled is not the one that binds, and the total budget has never
been raised.  This raises it.

Same chart, same sampling, same MLP, same Adam warm-up, same seed as
nn_dp3_nosym.py -- only the optimizer budget differs.  The exit reason is
printed, since it is what separates "hit the new cap" from "converged".

PRE-REGISTERED (fixed before running):
  (A) OPTIMIZATION.  held-out falls below 1e-7, i.e. more than a factor 4 below
      the 4.18e-7 the maxfun wall gave.  Then 4e-7 is a budget, appendix C's
      "the floor is not its iteration budget" is wrong, and the gap to the
      D6-informed network's 3.3e-10 is partly optimizer.
  (B) FLOOR.  held-out stalls within a factor 2 of 4.18e-7 AND the run exits on
      ftol/gtol rather than on a cap.  Then appendix C's claim is true for the
      first time, and can be stated on this evidence instead.
  (C) NEITHER.  still falling at the new cap: the answer is "more than this
      budget", and the number to report is where it got and what it cost.
Expressivity is not among these: the D6-informed network is a restriction of
this same function class and reaches 3.3e-10, so the class contains the answer.

Usage: PYTHONPATH=. python experiments/dp3/nn_budget_probe.py [maxfun]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from sugrasol.cone import B_DP3, dp3
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (gauge_fixed, loss_fn, residual_on_slice,
                          sample_slice, slice_chart)

jax.config.update("jax_enable_x64", True)

MAXFUN = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
PAPER_HELD = 4.18e-7        # what the maxfun=15,000 wall gives

chart = slice_chart(dp3(), B_DP3)
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)
ss_te = sample_slice(jax.random.PRNGKey(7), chart, 1024, eps=2e-3)


def make_psi(mlp_params):
    return gauge_fixed(lambda s: mlp(mlp_params, s), chart.anchors)


def total_loss(params, batch):
    mlp_params, c = params
    return loss_fn(chart, make_psi(mlp_params), c, batch)


key = jax.random.PRNGKey(0)
params = (init_mlp(key, sizes=(2, 20, 20, 1), scale=1e-2), None)
r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
params = (params[0], -float(jnp.mean(r0)))

opt = optax.adam(5e-3)
state = opt.init(params)
loss_te = jax.jit(lambda p: total_loss(p, ss_te))


@jax.jit
def step(params, state, batch):
    l, g = jax.value_and_grad(total_loss)(params, batch)
    upd, state = opt.update(g, state)
    return optax.apply_updates(params, upd), state, l


t0 = time.time()
for it in range(1201):
    params, state, l = step(params, state, ss)
print(f"  adam 1200: held {float(loss_te(params)):.3e}  ({time.time()-t0:.0f}s)",
      flush=True)

flat0, unravel = jax.flatten_util.ravel_pytree(params)
vg = jax.jit(jax.value_and_grad(lambda fv: total_loss(unravel(fv), ss)))
print(f"  L-BFGS with maxfun = {MAXFUN:,} (nn_dp3_nosym.py leaves it at scipy's "
      f"15,000)", flush=True)
TRACE_EVERY = max(1, MAXFUN // 200)
trace = []


def _cb(xk):
    """Held-out every TRACE_EVERY iterations.  The first run of this file gave
    the scaling from two budgets, which appendix C had to quote with that
    caveat; recording the trajectory replaces the two points with a curve."""
    _cb.k += 1
    if _cb.k % TRACE_EVERY == 0:
        h = float(loss_te(unravel(jnp.asarray(xk))))
        trace.append((_cb.k, h))
        print(f"    it {_cb.k:>9,}  held-out {h:.4e}", flush=True)


_cb.k = 0

res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
               np.asarray(flat0), jac=True, method="L-BFGS-B",
               callback=_cb,
               options=dict(maxiter=10 * MAXFUN, maxfun=MAXFUN,
                            ftol=1e-18, gtol=1e-16))
held = float(loss_te(unravel(jnp.asarray(res.x))))
print(f"\n  train {res.fun:.4e}   held-out {held:.4e}   "
      f"(paper at the old wall: {PAPER_HELD:.2e}, ratio {held/PAPER_HELD:.3f})")
print(f"  iterations {res.nit:,}   function evals {res.nfev:,}   "
      f"status {res.status}   {time.time()-t0:.0f}s")
print(f"  exit: {res.message}")

capped = res.nfev >= MAXFUN or res.status == 1
if len(trace) >= 4:
    t = np.array(trace)
    keep = t[:, 0] > t[-1, 0] / 20           # drop the transient
    sl = np.polyfit(np.log10(t[keep, 0]), np.log10(t[keep, 1]), 1)[0]
    cc = np.corrcoef(np.log10(t[keep, 0]), np.log10(t[keep, 1]))[0, 1]
    print(f"  trajectory: held-out ~ iterations^{sl:.2f} over the last "
          f"{int(keep.sum())} of {len(trace)} samples  (corr {cc:.4f})")
    np.savez(Path(__file__).resolve().parent / "nn_budget_trace.npz",
             iters=t[:, 0], held=t[:, 1], maxfun=MAXFUN, slope=sl, corr=cc)
if held < 1e-7:
    print("\n  => (A) OPTIMIZATION: the 4e-7 was budget, not floor.")
elif held > PAPER_HELD / 2 and not capped:
    print("\n  => (B) FLOOR: converged on its own test, and stayed there.")
else:
    print("\n  => (C) NEITHER: still bounded by the budget at this cost.")
