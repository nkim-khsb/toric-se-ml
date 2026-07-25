"""dP2 negative control: attempt KE (must fail) vs SE (must converge).

Same pentagon cone, two Reeb vectors:
  - REGULAR b=(3,0,0): base = smooth dP2. dP2 has no KE -> the transverse
    Monge-Ampere has NO solution. Honest negative result = a residual FLOOR that
    does NOT drop as we raise the ansatz degree (contrast the Y^{3,2} conditioning
    artifact, which DID drop with the right basis).
  - IRREGULAR b = volume-minimizer: the genuine Sasaki-Einstein metric exists
    (Futaki-Ono-Wang) -> residual drops to machine floor.

Same machinery for both (ortho polynomial basis + 3-anchor gauge fixing, as for
Y^{p,q}; no symmetry assumed). The discriminator is: does the loss reach ~0?

Usage: PYTHONPATH=. python experiments/dp2/negative_control.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import B_DP2_REG, dp2, minimize_reeb, vol_Y
from sugrasol.ypq import (
    gauge_fixed, loss_fn, ortho_psi, residual_on_slice, sample_slice,
    slice_chart, whiten_poly,
)

jax.config.update("jax_enable_x64", True)

cone = dp2()
b_se = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)
print(f"vol(Y): regular (3,0,0) = {float(vol_Y(cone, B_DP2_REG)) / np.pi**3:.5f} pi^3, "
      f"SE {tuple(np.round(np.array(b_se), 4))} = {float(vol_Y(cone, b_se)) / np.pi**3:.5f} pi^3 "
      f"(SE < regular confirms (3,0,0) is not the minimizer)\n")


def run(label, b, degrees):
    chart = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), chart, 3000, eps=2e-3)
    ss_test = sample_slice(jax.random.PRNGKey(7), chart, 3000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
    print(f"== {label} ==  psi=0 residual spread {float(jnp.max(r0) - jnp.min(r0)):.3f}")
    for DEG in degrees:
        powers, W = whiten_poly(DEG, ss)
        nc = len(powers)

        def L(v, batch):
            psi = gauge_fixed(ortho_psi(v[:nc], powers, W), chart.anchors)
            return loss_fn(chart, psi, v[nc], batch)

        vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
        L_te = jax.jit(lambda v: L(v, ss_test))
        v0 = np.zeros(nc + 1)
        v0[nc] = -float(jnp.mean(r0))
        res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                       v0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
        print(f"   deg {DEG:2d} ({nc:3d} params): train {res.fun:.3e}  "
              f"held-out {float(L_te(jnp.asarray(res.x))):.3e}  ({res.nit} it)")
    print()


run("KE-ATTEMPT  regular b=(3,0,0) [base=dP2, no KE => expect FLOOR]",
    B_DP2_REG, [6, 8, 10, 12, 14, 16])
run("SE  volume-min irregular b [FOW exists => expect CONVERGE]",
    b_se, [6, 8, 10, 12, 14])
