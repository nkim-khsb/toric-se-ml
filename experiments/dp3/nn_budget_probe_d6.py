"""The D6-informed dP3 network's floor: 3.3e-10 at 7770 iterations, with a log.

Section 4.4 and the discussion both quote "the D6-informed network meets its
convergence test after 7770 iterations at 3.3e-10", and that pair is what the
factor 35 cost of withholding the symmetry is measured against.  No committed
artifact held it: nn_dp3.py stops at maxiter = 3000 with maxfun at scipy's
default, so the production run reports 8.8e-8 (tab:threeway) and the floor was
recorded only in prose.  This is the same run carried to its convergence test,
logged.

Same chart, samples, MLP, D6 input symmetrization, Adam warm-up and seeds as
nn_dp3.py -- only the optimizer budget differs.  This is the D6 counterpart of
nn_budget_probe.py, which did the same for the symmetry-free network and found
698,344 iterations and 1.16e-8.

PRE-REGISTERED (fixed before running):
  (A) REPRODUCED.  exits on ftol/gtol (status 0, not a cap), held-out within a
      factor 2 of 3.3e-10, and iterations within a factor 2 of 7770.  Then
      section 4.4's pair is confirmed and has its artifact.
  (B) WRONG VALUE.  converges on its own test, but held-out is off by more than
      a factor 2.  Then the quoted 3.3e-10 needs correcting, and the factor 35
      against the symmetry-free 1.16e-8 moves with it.
  (C) NEVER CONVERGES.  reaches the cap.  Then "meets its convergence test" is
      wrong, both floors are budgets, and the comparison in section 4.4 has to
      be restated the way appendix C restated the symmetry-free one.
Every seed is fixed, so a rerun is a check on the published pair and not a
replacement for it.

Usage: PYTHONPATH=. python experiments/dp3/nn_budget_probe_d6.py [maxfun]
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
from sugrasol.ypq import (dihedral_matrices, gauge_fixed, loss_fn,
                          residual_on_slice, sample_slice, slice_chart)

jax.config.update("jax_enable_x64", True)

MAXFUN = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
PAPER_HELD, PAPER_NIT = 3.3e-10, 7770      # section 4.4
PROD_HELD = 8.8e-8                         # tab:threeway, at maxiter = 3000

chart = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(chart.verts_s)   # D6, linear on centred s
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)
ss_te = sample_slice(jax.random.PRNGKey(7), chart, 1024, eps=2e-3)


def make_psi(mlp_params):
    """D6 by input averaging over the twelve-element orbit, as nn_dp3.py.

    Note this is NOT wrapped in gauge_fixed: a D6-invariant psi carries no
    linear gauge mode to remove, and nn_dp3.py does not remove one either.
    """
    def sym_psi(s):
        gs = jnp.einsum("gij,j->gi", group, s)
        return jnp.mean(jax.vmap(lambda x: mlp(mlp_params, x))(gs))
    return sym_psi


def total_loss(params, batch):
    mlp_params, c = params
    return loss_fn(chart, make_psi(mlp_params), c, batch)


mlp_params = init_mlp(jax.random.PRNGKey(0), sizes=(2, 20, 20, 1), scale=1e-2)
r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
params = (mlp_params, -float(jnp.mean(r0)))
npar = int(sum(np.size(x) for x in jax.tree.leaves(params[0])))

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
print(f"  {npar} params, D6-symmetrized, 1024 samples", flush=True)
print(f"  adam 1200: held {float(loss_te(params)):.3e}  ({time.time()-t0:.0f}s)",
      flush=True)

flat0, unravel = jax.flatten_util.ravel_pytree(params)
vg = jax.jit(jax.value_and_grad(lambda fv: total_loss(unravel(fv), ss)))
print(f"  L-BFGS with maxfun = {MAXFUN:,} (nn_dp3.py: maxiter 3000, maxfun at "
      f"scipy's 15,000)", flush=True)
trace = []


def _cb(xk):
    _cb.k += 1
    if _cb.k % 500 == 0:
        h = float(loss_te(unravel(jnp.asarray(xk))))
        trace.append((_cb.k, h))
        print(f"    it {_cb.k:>7,}  held-out {h:.4e}", flush=True)


_cb.k = 0

res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
               np.asarray(flat0), jac=True, method="L-BFGS-B", callback=_cb,
               options=dict(maxiter=10 * MAXFUN, maxfun=MAXFUN,
                            ftol=1e-18, gtol=1e-16))
held = float(loss_te(unravel(jnp.asarray(res.x))))
capped = res.nfev >= MAXFUN or res.status == 1
print(f"\n  train {res.fun:.4e}   held-out {held:.4e}")
print(f"  paper (sec 4.4): {PAPER_HELD:.2e} at {PAPER_NIT:,} it   "
      f"-> ratio {held/PAPER_HELD:.3f}, iterations {res.nit/PAPER_NIT:.3f}x")
print(f"  production (tab:threeway, maxiter 3000): {PROD_HELD:.2e}   "
      f"-> released by {PROD_HELD/held:.1f}x")
print(f"  iterations {res.nit:,}   function evals {res.nfev:,}   "
      f"status {res.status}   {time.time()-t0:.0f}s")
print(f"  exit: {res.message}")

np.savez(Path(__file__).resolve().parent / "nn_budget_d6.npz",
         held=held, train=float(res.fun), nit=int(res.nit), nfev=int(res.nfev),
         status=int(res.status), capped=bool(capped), maxfun=MAXFUN,
         params=npar, seconds=time.time()-t0,
         trace_iters=np.array([t[0] for t in trace]),
         trace_held=np.array([t[1] for t in trace]))

within2 = 0.5 <= held / PAPER_HELD <= 2.0
nit2 = 0.5 <= res.nit / PAPER_NIT <= 2.0
if not capped and within2 and nit2:
    print("\n  => (A) REPRODUCED: section 4.4's pair now has its artifact.")
elif not capped:
    print(f"\n  => (B) WRONG VALUE: converged, but held-out {held:.3e} and "
          f"{res.nit:,} it do not match the quoted pair.")
else:
    print("\n  => (C) NEVER CONVERGES: this floor is a budget too.")
