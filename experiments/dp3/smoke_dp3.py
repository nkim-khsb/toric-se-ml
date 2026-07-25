"""dP3 smoke: D6-invariant ortho ansatz, cone-MA training, Abreu S=12 grading.

Convention: our SE_5 Ric=4g throughout (cone MA (2.54)); DHHKW reconciliation
is deferred to the Laplacian grader only. No anchor gauge-fixing: the D6-
invariant basis already excludes the affine flat directions.

Usage: PYTHONPATH=. python experiments/dp3/smoke_dp3.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import dp3, B_DP3
from sugrasol.ypq import (
    dihedral_matrices, loss_fn, residual_on_slice, sample_slice, slice_chart,
    sym_ortho_psi, t_of_s, whiten_sym_poly,
)

jax.config.update("jax_enable_x64", True)

chart = slice_chart(dp3(), B_DP3)
verts = chart.verts_s
group = dihedral_matrices(verts)

# --- verify the D6 group ---
A_rot = group[2]  # index 2 = R^1 (mats stored [I, ref, R, R.ref, R^2, ...])
P = jnp.eye(2)
order = next(k for k in range(1, 8)
             if jnp.allclose((P := P @ A_rot), jnp.eye(2), atol=1e-9))
inv_ok = all(
    set(map(lambda p: (round(float(p[0]), 4), round(float(p[1]), 4)),
            jnp.einsum("ij,vj->vi", g, verts)))
    == set(map(lambda p: (round(float(p[0]), 4), round(float(p[1]), 4)), verts))
    for g in group)
print(f"D6 group: {group.shape[0]} elements, rotation order {order}, "
      f"all permute vertices: {inv_ok}")

ss = sample_slice(jax.random.PRNGKey(1), chart, 2048, eps=2e-3)
ss_test = sample_slice(jax.random.PRNGKey(7), chart, 2048, eps=2e-3)

r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
print(f"psi=0 residual: mean {float(jnp.mean(r0)):.4f}, "
      f"spread {float(jnp.max(r0) - jnp.min(r0)):.4f}")


def abreu_S(psi, s):
    def u(ss):
        t = t_of_s(chart, ss)
        lt = chart.cone.ells(t)
        return 0.5 * jnp.sum(lt * jnp.log(lt)) + psi(ss)
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)
    return -jnp.einsum("jkjk->", T)


for DEG in (6, 10, 14, 18):
    powers, W, grp = whiten_sym_poly(DEG, ss, group)
    nc = W.shape[1]

    def L(v, batch):
        psi = sym_ortho_psi(v[:nc], powers, W, grp)  # NO gauge_fixed (D6)
        return loss_fn(chart, psi, v[nc], batch)

    vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
    L_te = jax.jit(lambda v: L(v, ss_test))
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    t0 = time.time()
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
    dt = time.time() - t0
    v = jnp.asarray(res.x)
    psi = sym_ortho_psi(v[:nc], powers, W, grp)
    sup = float(jnp.max(jnp.abs(jax.vmap(psi)(ss_test))))
    ss_e = sample_slice(jax.random.PRNGKey(11), chart, 200, eps=1e-2)
    S = jax.vmap(lambda s: abreu_S(psi, s))(ss_e)
    print(f"deg {DEG:2d}: invariant dim {nc:3d}  loss {res.fun:.2e}  "
          f"held {float(L_te(v)):.2e}  sup|psi| {sup:.3e}  "
          f"AbreuS[{float(jnp.min(S)):.3f},{float(jnp.max(S)):.3f}] "
          f"dev {float(jnp.max(jnp.abs(S - 12)) / 12):.2e}  ({res.nit} it, {dt:.1f}s)")
