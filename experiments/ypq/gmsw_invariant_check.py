"""Gold-standard grading of the learned Y^{p,q} metric against GMSW closed form.

Method (coordinate-free): compute the scalar invariants
    Phi1 = |Riem|^2,  Phi2 = Riem^3
of the 6d Ricci-flat cone at the r=1 slice, from
  (i) the LEARNED toric potential  (2d scatter over the slice polygon),
 (ii) the GMSW closed-form metric  (1d curve in y).
Cohomogeneity-1 of the true solution => the 2d scatter must COLLAPSE onto a
1d curve; correctness => that curve must coincide with GMSW's. The network
was never told about the SU(2): the collapse is emergent.

Usage: PYTHONPATH=. python3 experiments/ypq/gmsw_invariant_check.py [p q]
"""
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

from sugrasol.cone import b_ypq, ypq
from sugrasol.curvature import invariants, toric_metric
from sugrasol.gmsw import cone6, roots_and_a
from sugrasol.ypq import (
    cone_potential, gauge_fixed, loss_fn, n_poly_coeffs, poly_psi,
    sample_slice, slice_chart, t_of_s,
)

jax.config.update("jax_enable_x64", True)

P, Q = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2, 1)
DEG = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
print(f"== Y^{{{P},{Q}}} ==")

chart = slice_chart(ypq(P, Q), b_ypq(P, Q))
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)


def train():
    """Polynomial + trainable c: smooth least-squares problem -> L-BFGS."""
    import numpy as np
    from scipy.optimize import minimize

    nc = n_poly_coeffs(DEG)

    def L(v):
        psi = gauge_fixed(poly_psi(v[:nc], DEG), chart.anchors)
        return loss_fn(chart, psi, v[nc], ss)

    vg = jax.jit(jax.value_and_grad(L))
    from sugrasol.ypq import residual_on_slice
    import os
    ckpt = f"experiments/ypq/ckpt_y{P}{Q}_deg{DEG}.npy"
    if os.path.exists(ckpt):
        v0 = np.load(ckpt)
        print("resuming from", ckpt)
    else:
        r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
        v0 = np.zeros(nc + 1)
        v0[nc] = -float(jnp.mean(r0))

    def fun(v):
        l, g = vg(jnp.asarray(v))
        return float(l), np.asarray(g)

    res = minimize(fun, v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=STEPS, ftol=1e-18, gtol=1e-14))
    print(f"trained loss: {res.fun:.3e}  (L-BFGS iters {res.nit})")
    np.save(ckpt, res.x)
    v = jnp.asarray(res.x)
    return (v[:nc], v[nc])


pc = train()
psi_tr = gauge_fixed(poly_psi(pc[0], DEG), chart.anchors)

if len(sys.argv) > 5 and int(sys.argv[5]) == 0:  # train-only mode
    sys.exit(0)

# ---- invariants of the learned (and untrained) cone at r=1 (l_b = 1/2)
N_EVAL = int(sys.argv[5]) if len(sys.argv) > 5 else 400
SKIP_BEFORE = len(sys.argv) > 5
ss_eval = sample_slice(jax.random.PRNGKey(11), chart, N_EVAL, eps=1e-2)


def scatter_invariants(psi):
    G = cone_potential(chart, psi)
    g6 = toric_metric(G, 3)

    def inv_at(s):
        y = t_of_s(chart, s) / 2.0  # l_b = 1/2  <=>  r = 1
        x6 = jnp.concatenate([y, jnp.zeros(3)])
        return jnp.stack(invariants(g6, x6))

    return jax.vmap(inv_at)(ss_eval)  # (N, 2)

inv_before = None if SKIP_BEFORE else scatter_invariants(lambda s: 0.0)
inv_after = scatter_invariants(psi_tr)

# ---- GMSW closed-form curve
y1, y2, a = roots_and_a(P, Q)
g6_gmsw = cone6(float(a))
ys = jnp.linspace(float(y1) + 1e-4, float(y2) - 1e-4, 200)


def gmsw_inv(yv):
    x6 = jnp.array([1.0, yv, 0.9, 0.3, 0.7, 0.2])
    return jnp.stack(invariants(g6_gmsw, x6))

inv_gmsw = jax.vmap(gmsw_inv)(ys)  # (200, 2)

# ---- deviation: interpolate GMSW Phi2(Phi1) and compare at learned points
order = jnp.argsort(inv_gmsw[:, 0])
xg, yg = inv_gmsw[order, 0], inv_gmsw[order, 1]
phi2_ref = jnp.interp(inv_after[:, 0], xg, yg)
rel = jnp.abs(inv_after[:, 1] - phi2_ref) / jnp.abs(phi2_ref)
inside = (inv_after[:, 0] > float(jnp.min(xg))) & (inv_after[:, 0] < float(jnp.max(xg)))
rel_in = rel[inside]
print(f"points within GMSW Phi1 range: {int(jnp.sum(inside))}/{len(rel)}")
print(f"relative deviation of Phi2(Phi1): median {float(jnp.median(rel_in)):.3e}, "
      f"max {float(jnp.max(rel_in)):.3e}")

# ---- figure
panels = ([(inv_after, r"after training (learned $\psi$)")] if SKIP_BEFORE else
          [(inv_before, r"before training ($\psi=0$)"),
           (inv_after, r"after training (learned $\psi$)")])
fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels), 4.4),
                         squeeze=False)
for ax, (inv, ttl) in zip(axes[0], panels):
    ax.scatter(inv[:, 0], inv[:, 1], s=8, alpha=0.55, color="#c44e52",
               label="learned metric (2d samples)", zorder=3)
    ax.plot(inv_gmsw[:, 0], inv_gmsw[:, 1], color="#4c72b0", lw=2,
            label="GMSW closed form (1d curve)", zorder=2)
    ax.set_xlabel(r"$\Phi_1=|\mathrm{Riem}|^2$")
    ax.set_ylabel(r"$\Phi_2=\mathrm{Riem}^3$")
    ax.set_title(f"$Y^{{{P},{Q}}}$ cone at $r=1$: {ttl}")
    ax.legend(fontsize=8)
fig.tight_layout()
out = f"experiments/ypq/gmsw_collapse_y{P}{Q}.png"
fig.savefig(out, dpi=160)
print("figure:", out)
