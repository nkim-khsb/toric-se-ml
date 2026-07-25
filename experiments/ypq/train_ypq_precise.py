"""Precision run for Y^{p,q}: conditioned (orthonormalized) polynomial ansatz.

Motivation (log.md 2026-07-12): for Y^{3,2} raw monomials of degree 10-12 are
Vandermonde-ill-conditioned on the off-center slice polygon, so L-BFGS stalled
(~7e-8) and deg-12 did NOT beat deg-10.  Here we orthonormalize the SAME
polynomial space over the training samples (ypq.whiten_poly) — same ansatz,
better conditioned — and drive L-BFGS to the residual floor, then grade against
the GMSW closed form with the coordinate-free curvature-invariant collapse.

Usage: PYTHONPATH=. python experiments/ypq/train_ypq_precise.py p q [deg steps nsamp neval]
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import b_ypq, ypq
from sugrasol.curvature import invariants, toric_metric
from sugrasol.gmsw import cone6, roots_and_a
from sugrasol.ypq import (
    cone_potential, gauge_fixed, loss_fn, ortho_psi, residual_on_slice,
    sample_slice, slice_chart, t_of_s, whiten_poly,
)

jax.config.update("jax_enable_x64", True)

P = int(sys.argv[1]) if len(sys.argv) > 1 else 3
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DEG = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
NSAMP = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
NEVAL = int(sys.argv[6]) if len(sys.argv) > 6 else 400
print(f"== Y^{{{P},{Q}}}  deg={DEG}  steps={STEPS}  nsamp={NSAMP} ==")

chart = slice_chart(ypq(P, Q), b_ypq(P, Q))
ss = sample_slice(jax.random.PRNGKey(1), chart, NSAMP, eps=2e-3)
ss_test = sample_slice(jax.random.PRNGKey(7), chart, NSAMP, eps=2e-3)

powers, W = whiten_poly(DEG, ss)
nc = len(powers)
print(f"n_coeffs = {nc}")


def L(v, batch):
    psi = gauge_fixed(ortho_psi(v[:nc], powers, W), chart.anchors)
    return loss_fn(chart, psi, v[nc], batch)


vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
L_test = jax.jit(lambda v: L(v, ss_test))

# init: psi = 0, c = -mean residual at psi=0 (pretrain-then-relax anchor)
ckpt = f"experiments/ypq/ckpt_y{P}{Q}_deg{DEG}_ortho.npy"
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
               options=dict(maxiter=STEPS, ftol=1e-18, gtol=1e-16))
v = jnp.asarray(res.x)
np.save(ckpt, res.x)
l_train, l_test = float(res.fun), float(L_test(v))
print(f"trained loss: {l_train:.4e}  held-out: {l_test:.4e}  "
      f"(L-BFGS iters {res.nit}, {res.message})")

psi_tr = gauge_fixed(ortho_psi(v[:nc], powers, W), chart.anchors)
sup_psi = float(jnp.max(jnp.abs(jax.vmap(psi_tr)(ss_test))))
print(f"sup|psi| = {sup_psi:.4e},  c = {float(v[nc]):.6f}")

# ---- GMSW coordinate-free grading (curvature invariants at r=1 slice) -------
ss_eval = sample_slice(jax.random.PRNGKey(11), chart, NEVAL, eps=1e-2)


def scatter_invariants(psi):
    g6 = toric_metric(cone_potential(chart, psi), 3)

    def inv_at(s):
        y = t_of_s(chart, s) / 2.0  # l_b = 1/2  <=>  r = 1
        x6 = jnp.concatenate([y, jnp.zeros(3)])
        return jnp.stack(invariants(g6, x6))

    return jax.vmap(inv_at)(ss_eval)


inv_after = scatter_invariants(psi_tr)

y1, y2, a = roots_and_a(P, Q)
g6_gmsw = cone6(float(a))
ys = jnp.linspace(float(y1) + 1e-4, float(y2) - 1e-4, 200)
inv_gmsw = jax.vmap(lambda yv: jnp.stack(
    invariants(g6_gmsw, jnp.array([1.0, yv, 0.9, 0.3, 0.7, 0.2]))))(ys)

order = jnp.argsort(inv_gmsw[:, 0])
xg, yg = inv_gmsw[order, 0], inv_gmsw[order, 1]
phi2_ref = jnp.interp(inv_after[:, 0], xg, yg)
rel = jnp.abs(inv_after[:, 1] - phi2_ref) / jnp.abs(phi2_ref)
inside = (inv_after[:, 0] > float(jnp.min(xg))) & (inv_after[:, 0] < float(jnp.max(xg)))
rel_in = rel[inside]
print(f"points within GMSW range: {int(jnp.sum(inside))}/{len(rel)}")
print(f"GMSW Phi2(Phi1) rel dev: median {float(jnp.median(rel_in)):.3e}  "
      f"max {float(jnp.max(rel_in)):.3e}")

fig, ax = plt.subplots(figsize=(5.7, 4.4))
ax.scatter(inv_after[:, 0], inv_after[:, 1], s=8, alpha=0.55, color="#c44e52",
           label="learned metric (2d samples)", zorder=3)
ax.plot(inv_gmsw[:, 0], inv_gmsw[:, 1], color="#4c72b0", lw=2,
        label="GMSW closed form (1d curve)", zorder=2)
ax.set_xlabel(r"$\Phi_1=|\mathrm{Riem}|^2$")
ax.set_ylabel(r"$\Phi_2=\mathrm{Riem}^3$")
ax.set_title(f"$Y^{{{P},{Q}}}$ cone at $r=1$ (deg {DEG}, ortho basis)")
ax.legend(fontsize=8)
fig.tight_layout()
out = f"experiments/ypq/gmsw_collapse_y{P}{Q}_ortho.png"
fig.savefig(out, dpi=160)
print("figure:", out)
