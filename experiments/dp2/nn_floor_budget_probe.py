"""Is the dP2 network's held-out floor the geometry, or its optimizer budget?

Section 5.5 of paper 1 answers the referee question "is the 4e-2 floor the
obstruction or the ceiling of your polynomial basis?" by escalating a network
instead, and reporting that the floor survives.  But negative_control_nn.py
sets maxiter and not maxfun, so every one of those runs stopped at scipy's
default 15,000 function evaluations -- the same unreleased budget that, on the
dP3 network, turned out to be worth a factor 17.6 (nn_budget_probe.log).  The
claim that carries section 5.5 therefore rests on runs whose budget was never
released.  This releases it.

Same pentagon, same sampling, same gauge, same MLP and Adam warm-up as
negative_control_nn.py; only maxfun changes.  Both Reeb vectors are run,
because the b* column is what rules out "the network simply cannot optimize on
this pentagon".

PRE-REGISTERED (fixed before running):
  (1) At the regular Reeb the held-out residual stays above 1e-2 and the run
      exits on a cap or on ftol/gtol with the residual pinned.  Then the floor
      is not the budget, section 5.5 stands, and stands on better evidence.
  (2) At the regular Reeb the held-out residual falls below 1e-2, i.e. a
      factor 4 below the polynomial's 4.2e-2.  Then the network floor was
      partly budget and section 5.5 must be qualified.  Note that it cannot
      fall below a real obstruction: a drop that stops near the polynomial's
      4.2e-2 is consistent with the claim, a drop far past it is not.
  (3) At b* the residual keeps improving on the released budget, as it did on
      dP3.  Anything else would mean the two columns are not comparable and
      neither number means what section 5.5 says.

Usage: PYTHONPATH=. python experiments/dp2/nn_floor_budget_probe.py [maxfun]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from sugrasol.cone import B_DP2_REG, dp2, minimize_reeb
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (gauge_fixed, loss_fn, residual_on_slice,
                          sample_slice, slice_chart)

jax.config.update("jax_enable_x64", True)

MAXFUN = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
ADAM_IT = 1200
POLY_FLOOR = 4.2e-2          # the polynomial's held-out floor at regular b
PAPER_NN = {"regular": 6.1e-2, "b*": 3.1e-8}   # what the old wall gave

cone = dp2()
b_se = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)


def probe(label, b, h):
    chart = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), chart, 3000, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(7), chart, 3000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)

    def make_psi(mp):
        return gauge_fixed(lambda s: mlp(mp, s), chart.anchors)

    def total(params, batch):
        mp, c = params
        return loss_fn(chart, make_psi(mp), c, batch)

    params = (init_mlp(jax.random.PRNGKey(0), sizes=(2, h, h, 1), scale=1e-2),
              -float(jnp.mean(r0)))
    nparam = int(sum(np.size(x) for x in jax.tree.leaves(params)))
    opt = optax.adam(5e-3)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        l, g = jax.value_and_grad(total)(params, batch)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l

    t0 = time.time()
    for _ in range(ADAM_IT):
        params, state, _ = step(params, state, ss)
    flat0, unravel = jax.flatten_util.ravel_pytree(params)
    vg = jax.jit(jax.value_and_grad(lambda fv: total(unravel(fv), ss)))
    res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
                   np.asarray(flat0), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=10 * MAXFUN, maxfun=MAXFUN,
                                ftol=1e-18, gtol=1e-16))
    params = unravel(jnp.asarray(res.x))
    held = float(total(params, ss_te))
    capped = res.nfev >= MAXFUN or res.status == 1
    print(f"  {label:9s} width {h:2d} ({nparam:5d} params): train {res.fun:.3e}  "
          f"held-out {held:.3e}", flush=True)
    print(f"            {res.nit:,} it, {res.nfev:,} evals, "
          f"{'CAPPED' if capped else 'exited on its own test'}, "
          f"{time.time()-t0:.0f}s", flush=True)
    return held, capped


if __name__ == "__main__":
    print(f"== dP2 network floor, maxfun = {MAXFUN:,} "
          f"(negative_control_nn.py left it at 15,000) ==", flush=True)
    print(f"   polynomial held-out floor at regular b: {POLY_FLOOR:.1e}\n", flush=True)
    out = {}
    for label, b, h in (("regular", B_DP2_REG, 16), ("regular", B_DP2_REG, 32),
                        ("b*", b_se, 16)):
        out[(label, h)] = probe(label, b, h)

    reg = min(v[0] for (l, _), v in out.items() if l == "regular")
    print(f"\n  best regular-b held-out at this budget: {reg:.3e}  "
          f"(old wall {PAPER_NN['regular']:.1e}, polynomial {POLY_FLOOR:.1e})")
    if reg > 1e-2:
        print("  => (1) the floor is NOT the budget: section 5.5 stands.")
    else:
        print("  => (2) the floor moved below 1e-2: section 5.5 needs qualifying.")
    print(f"  b* at this budget: {out[('b*', 16)][0]:.3e}  "
          f"(old wall {PAPER_NN['b*']:.1e})")
