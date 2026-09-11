"""Wall clock for the dP2 polynomial, on the same machine and build as the
network probes, so the two can be put in one table.

The paper's cost paragraph says only "minutes on one core".  The comparison a
reader actually wants -- what each function class costs to reach what precision
on the same target -- needs the two measured under the same conditions.  The
polynomial timings quoted so far came from revisions_Hoseob's robustness log,
a different session; these replace them.

Reported the way nn_floor_budget_probe.py reports: seconds from before the fit
(so compilation is inside the number, as it is for the network), iterations,
function evaluations, and the exit message, which is the part that matters --
at b* the polynomial stops on its own convergence test and the network does
not.

The last row deliberately gives the polynomial the network's released budget at
the regular Reeb vector, maxfun = 100,000, so the two floors are compared at
equal cost rather than at whatever default each happened to hit.

Usage: PYTHONPATH=. python experiments/dp2/poly_timing.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import B_DP2_REG, dp2, minimize_reeb
from sugrasol.ypq import (gauge_fixed, loss_fn, ortho_psi, residual_on_slice,
                          sample_slice, slice_chart, whiten_poly)

jax.config.update("jax_enable_x64", True)

cone = dp2()
b_se = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)


def fit(label, b, deg, maxfun=None):
    t0 = time.time()
    ch = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), ch, 3000, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(7), ch, 3000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
    powers, W = whiten_poly(deg, ss)
    nc = len(powers)

    def L(v, batch):
        psi = gauge_fixed(ortho_psi(v[:nc], powers, W), ch.anchors)
        return loss_fn(ch, psi, v[nc], batch)

    vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
    L_te = jax.jit(lambda v: L(v, ss_te))
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    opts = dict(maxiter=20000, ftol=1e-18, gtol=1e-16)
    if maxfun is not None:
        opts.update(maxiter=10 * maxfun, maxfun=maxfun)
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B", options=opts)
    held = float(L_te(jnp.asarray(res.x)))
    dt = time.time() - t0
    print(f"  {label:9s} deg {deg:2d} ({nc:3d} params): train {res.fun:.3e}  "
          f"held-out {held:.3e}", flush=True)
    print(f"            {res.nit:,} it, {res.nfev:,} evals, {dt:.0f}s   "
          f"exit: {res.message}", flush=True)
    return held, dt, res.nit


if __name__ == "__main__":
    print("== dP2 polynomial, wall clock on this machine ==", flush=True)
    print("   (network, same machine: regular width 16 -> 5.45e-2 in 1067s, CAPPED)\n",
          flush=True)
    for deg in (12, 14, 16):
        fit("b*", b_se, deg)
    print(flush=True)
    fit("regular", B_DP2_REG, 16)
    print("\n  and at the network's released budget, for an equal-cost floor:",
          flush=True)
    fit("regular", B_DP2_REG, 16, maxfun=100_000)
