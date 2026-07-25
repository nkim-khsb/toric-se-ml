"""dP2 negative control with a NEURAL ansatz: is the held-out floor a property
of the GEOMETRY or of our polynomial basis?

The polynomial negative control (negative_control.py) shows the held-out
residual pinned at ~4e-2 at the regular Reeb b=(3,0,0), where dP2 would need a
Kaehler-Einstein metric it cannot have, while the training residual overfits
away from it.  The obvious referee question is whether that floor is the
obstruction or merely the ceiling of the polynomial function class -- the paper
itself shows (Y^{3,2}) that a 'floor' can be a basis artifact.

This run answers it with a completely different function class.  Same pentagon,
same two Reeb vectors, same 3-anchor affine gauge fixing, same residual-only
loss; the ansatz is a plain MLP (no symmetry imposed -- dP2 has only Z_2
anyway), and the analogue of degree escalation is WIDTH escalation, up to
~1200 parameters, well beyond the 150 of the polynomial run.

Read the two columns together: the SE Reeb is the positive contrast (a solution
exists there by Futaki-Ono-Wang), so it controls for 'the NN just cannot
optimize on this pentagon'.

Usage: PYTHONPATH=. python experiments/dp2/negative_control_nn.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from sugrasol.cone import B_DP2_REG, dp2, minimize_reeb
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (
    gauge_fixed, loss_fn, residual_on_slice, sample_slice, slice_chart,
)

jax.config.update("jax_enable_x64", True)

cone = dp2()
b_se = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)
print(f"Reeb vectors: regular {tuple(np.array(B_DP2_REG))}, "
      f"SE (volume-min) {tuple(np.round(np.array(b_se), 6))}\n", flush=True)

WIDTHS = (8, 16, 24, 32)          # capacity ladder, ~105 -> ~1185 parameters
N_SAMP = 3000                      # matches the polynomial run
ADAM_IT, LBFGS_IT = 800, 5000


def run(label, b):
    chart = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), chart, N_SAMP, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(7), chart, N_SAMP, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
    print(f"== {label} ==  psi=0 residual spread "
          f"{float(jnp.max(r0) - jnp.min(r0)):.3f}", flush=True)

    for h in WIDTHS:
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
                       options=dict(maxiter=LBFGS_IT, ftol=1e-18, gtol=1e-16))
        params = unravel(jnp.asarray(res.x))
        held = float(total(params, ss_te))
        print(f"   width {h:2d} ({nparam:5d} params): train {res.fun:.3e}  "
              f"held-out {held:.3e}   ({res.nit} it, {time.time() - t0:.0f}s)", flush=True)
    print(flush=True)


run("KE-ATTEMPT  regular b=(3,0,0)  [no KE on dP2 => expect FLOOR]", B_DP2_REG)
run("SE  volume-min irregular b     [FOW: exists => expect CONVERGE]", b_se)
