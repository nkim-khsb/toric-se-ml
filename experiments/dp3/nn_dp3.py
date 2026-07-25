"""NN ansatz on dP3: does a neural psi reproduce the (known) KE metric?

dP3 is polynomial-tractable -- so this is NOT about needing a NN.  It is a
CONTROLLED de-risking of the neural pipeline for the non-toric main line
(deformed conifold, dP3 smoothing) where the polynomial reduction is gone and
the W4a wall was hit.  The boundary lesson from the polynomial work is built in:
Guillemin's l ln l lives entirely in G_can, so the NN represents only the SMOOTH
correction psi -- no boundary singularity for it to fight.  D6 is built in by
input symmetrization (same as the polynomial), matching apples-to-apples.

Ground truth = the degree-18 polynomial solution (dp3_G_deg18.npz).  Grades:
held-out MA loss, Abreu S=12, pointwise psi agreement (up to a constant), and
lambda1 via the Laplacian.

Usage: PYTHONPATH=. python experiments/dp3/nn_dp3.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sugrasol.cone import B_DP3, dp3
from sugrasol.laplacian import laplace_spectrum, polygon_quadrature, slice_potential
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, loss_fn, residual_on_slice,
    sample_slice, slice_chart, sym_ortho_psi,
)

jax.config.update("jax_enable_x64", True)

chart = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(chart.verts_s)          # D6, linear on centred s
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)
ss_te = sample_slice(jax.random.PRNGKey(7), chart, 1024, eps=2e-3)


# ---- D6-symmetrized MLP psi (boundary in G_can, symmetry by input averaging)
def sym_psi(mlp_params, s):
    gs = jnp.einsum("gij,j->gi", group, s)        # (12, 2) orbit of s
    return jnp.mean(jax.vmap(lambda x: mlp(mlp_params, x))(gs))


def make_psi(mlp_params):
    return lambda s: sym_psi(mlp_params, s)


def total_loss(params, batch):
    mlp_params, c = params
    return loss_fn(chart, make_psi(mlp_params), c, batch)


# ---- init at psi ~ 0 (Guillemin), c = -<residual at psi=0>  (as the poly did)
key = jax.random.PRNGKey(0)
mlp_params = init_mlp(key, sizes=(2, 20, 20, 1), scale=1e-2)
r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
params = (mlp_params, -float(jnp.mean(r0)))
print(f"psi=0 residual mean {float(jnp.mean(r0)):.4f}  "
      f"spread {float(jnp.max(r0) - jnp.min(r0)):.4f}", flush=True)

# ---- short Adam warm-up (break MLP symmetry), then L-BFGS as the workhorse
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
    if it % 400 == 0:
        print(f"  adam {it:5d}: train {float(l):.3e}  held {float(loss_te(params)):.3e}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

# ---- L-BFGS polish (scipy) on the flattened pytree
from scipy.optimize import minimize

flat0, unravel = jax.flatten_util.ravel_pytree(params)
vg = jax.jit(jax.value_and_grad(lambda fv: total_loss(unravel(fv), ss)))
res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
               np.asarray(flat0), jac=True, method="L-BFGS-B",
               options=dict(maxiter=3000, ftol=1e-18, gtol=1e-16))
params = unravel(jnp.asarray(res.x))
print(f"  L-BFGS: train {res.fun:.3e}  held {float(loss_te(params)):.3e}  "
      f"({res.nit} it, {time.time() - t0:.0f}s)", flush=True)

# ============================ grade against ground truth =====================
psi_nn = make_psi(params[0])

# ground truth: deg-18 polynomial psi
d = np.load("experiments/dp3smooth/dp3_G_deg18.npz", allow_pickle=True)
psi_poly = sym_ortho_psi(jnp.asarray(d["coeffs"]), _poly_powers(int(d["degree"])),
                         jnp.asarray(d["W"]), jnp.asarray(d["group"]))


def abreu_S(psi, s):
    u = slice_potential(chart, psi)
    Hinv = lambda z: jnp.linalg.inv(jax.hessian(u)(z))
    return -jnp.einsum("jkjk->", jax.jacfwd(jax.jacfwd(Hinv))(s))


ss_e = sample_slice(jax.random.PRNGKey(11), chart, 200, eps=1e-2)
S_nn = jax.vmap(lambda s: abreu_S(psi_nn, s))(ss_e)
print(f"\nAbreu S (NN): [{float(jnp.min(S_nn)):.4f}, {float(jnp.max(S_nn)):.4f}]  "
      f"dev {float(jnp.max(jnp.abs(S_nn - 12)) / 12):.2e}")

# pointwise psi agreement up to a constant (both D6-invariant, no linear gauge)
vnn = jax.vmap(psi_nn)(ss_te)
vpoly = jax.vmap(psi_poly)(ss_te)
dv = (vnn - jnp.mean(vnn)) - (vpoly - jnp.mean(vpoly))
print(f"pointwise |psi_NN - psi_poly| (mean-subtracted): "
      f"rms {float(jnp.sqrt(jnp.mean(dv**2))):.2e}  max {float(jnp.max(jnp.abs(dv))):.2e}  "
      f"(sup|psi_poly| {float(jnp.max(jnp.abs(vpoly - jnp.mean(vpoly)))):.3e})")

# lambda1 via the Laplacian (deterministic quadrature)
nodes, wts = polygon_quadrature(chart.verts_s, ngl=24)
S_mean = float(jnp.mean(S_nn))
powers = _poly_powers(20)


def sym_basis(s):
    gs = jnp.einsum("gij,j->gi", group, s)
    sym = jnp.mean(jax.vmap(lambda g: _monomials(g, powers))(gs), axis=0)
    return jnp.concatenate([jnp.ones(1), sym])


A = jax.vmap(sym_basis)(sample_slice(jax.random.PRNGKey(5), chart, 40000, eps=1e-4))
_, Sv, Vt = jnp.linalg.svd(A, full_matrices=False)
r = int(jnp.sum(Sv > 1e-12 * Sv[0]))
Wb = Vt[:r].T / Sv[:r]
basis = lambda s: sym_basis(s) @ Wb
u_nn = slice_potential(chart, psi_nn)
w = laplace_spectrum(u_nn, basis, nodes, weights=wts)
print(f"lambda1 (NN metric): {float(w[1]) * 4 / S_mean:.5f}  (poly/DHHKW 6.3228)")
