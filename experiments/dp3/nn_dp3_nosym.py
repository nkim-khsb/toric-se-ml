"""NN on dP3 WITHOUT the D6 symmetry: can the network find the (symmetric)
KE metric on its own, and does D6 emerge?

Why this run.  nn_dp3.py built D6 into the ansatz by orbit-averaging, matching
the polynomial apples-to-apples; it tested the pipeline's mechanics (boundary
handling, convergence), not its ability to cope without symmetry.  But the
non-toric main line needs exactly that: the branch-(i) deformation BREAKS D6
(generic epsilon leaves only (C^*)^2 x Z_2, note-w3-symmetry), so the honest
de-risking run is the symmetry-free one.  dP3 is the right place to do it
because the answer is known: the KE solution is D6-symmetric and unique up to
gauge, so a symmetry-free network that converges MUST discover that symmetry.
This is the toric analogue of the SU(2)^2 emergence of the rung-A 6d run.

Gauge, and a trap.  With D6 built in, the affine flat direction psi -> psi +
(a + b.s) was killed for free: no linear function carries a D6 invariant.
Dropping the symmetry REOPENS it, so we must fix it explicitly (rule 4) --
`gauge_fixed` subtracts the affine function matching psi at three anchors.
Without this the optimizer drifts along an exactly flat direction.

Grades: held-out MA loss, Abreu S=12, pointwise psi vs the degree-18 polynomial
solution (both in the same 3-anchor gauge), lambda1, and the D6 ORBIT SPREAD
(max over the 12-element orbit minus min, relative to sup|psi|) as the
emergence measure.

Usage: PYTHONPATH=. python experiments/dp3/nn_dp3_nosym.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from sugrasol.cone import B_DP3, dp3
from sugrasol.laplacian import laplace_spectrum, polygon_quadrature, slice_potential
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, gauge_fixed, loss_fn,
    residual_on_slice, sample_slice, slice_chart, sym_ortho_psi,
)

jax.config.update("jax_enable_x64", True)

chart = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(chart.verts_s)          # only used for GRADING now
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)
ss_te = sample_slice(jax.random.PRNGKey(7), chart, 1024, eps=2e-3)


# ---- raw (NON-symmetrized) MLP psi, affine gauge fixed at three anchors
def make_psi(mlp_params):
    raw = lambda s: mlp(mlp_params, s)
    return gauge_fixed(raw, chart.anchors)


def total_loss(params, batch):
    mlp_params, c = params
    return loss_fn(chart, make_psi(mlp_params), c, batch)


key = jax.random.PRNGKey(0)
mlp_params = init_mlp(key, sizes=(2, 20, 20, 1), scale=1e-2)
r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
params = (mlp_params, -float(jnp.mean(r0)))
print(f"psi=0 residual mean {float(jnp.mean(r0)):.4f}  "
      f"spread {float(jnp.max(r0) - jnp.min(r0)):.4f}  [NO D6 in the ansatz]",
      flush=True)

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

flat0, unravel = jax.flatten_util.ravel_pytree(params)
vg = jax.jit(jax.value_and_grad(lambda fv: total_loss(unravel(fv), ss)))
res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
               np.asarray(flat0), jac=True, method="L-BFGS-B",
               options=dict(maxiter=6000, ftol=1e-18, gtol=1e-16))
params = unravel(jnp.asarray(res.x))
print(f"  L-BFGS: train {res.fun:.3e}  held {float(loss_te(params)):.3e}  "
      f"({res.nit} it, {time.time() - t0:.0f}s)", flush=True)

# ============================ grading ========================================
psi_nn = make_psi(params[0])

d = np.load("experiments/dp3smooth/dp3_G_deg18.npz", allow_pickle=True)
psi_poly_raw = sym_ortho_psi(jnp.asarray(d["coeffs"]), _poly_powers(int(d["degree"])),
                             jnp.asarray(d["W"]), jnp.asarray(d["group"]))
psi_poly = gauge_fixed(psi_poly_raw, chart.anchors)   # SAME gauge as the NN


def abreu_S(psi, s):
    u = slice_potential(chart, psi)
    Hinv = lambda z: jnp.linalg.inv(jax.hessian(u)(z))
    return -jnp.einsum("jkjk->", jax.jacfwd(jax.jacfwd(Hinv))(s))


ss_e = sample_slice(jax.random.PRNGKey(11), chart, 200, eps=1e-2)
S_nn = jax.vmap(lambda s: abreu_S(psi_nn, s))(ss_e)
print(f"\nAbreu S (NN, no D6): [{float(jnp.min(S_nn)):.4f}, {float(jnp.max(S_nn)):.4f}]  "
      f"dev {float(jnp.max(jnp.abs(S_nn - 12)) / 12):.2e}")

vnn = jax.vmap(psi_nn)(ss_te)
vpoly = jax.vmap(psi_poly)(ss_te)
dv = vnn - vpoly
sup_poly = float(jnp.max(jnp.abs(vpoly)))
print(f"pointwise |psi_NN - psi_poly| (same gauge): rms {float(jnp.sqrt(jnp.mean(dv**2))):.2e}"
      f"  max {float(jnp.max(jnp.abs(dv))):.2e}   (sup|psi_poly| {sup_poly:.3e})")

# ---- D6 EMERGENCE, measured GAUGE-INVARIANTLY.
# Trap (hit on the first run): the orbit spread of psi ITSELF is not a symmetry
# measure once the affine gauge is fixed at three non-symmetric anchors -- the
# subtracted affine piece is not D6-invariant, so even the exactly invariant
# polynomial solution shows a spread (both came out at 4.1e-2, revealing the
# flaw rather than a violation).  The Hessian kills affine functions, so it is
# gauge-invariant: psi invariant  <=>  H(g s) = g^{-T} H(s) g^{-1}.
def hess_equivariance(psi, s):
    H = jax.hessian(psi)
    H0 = H(s)
    scale = jnp.linalg.norm(H0)

    def dev(g):
        gi = jnp.linalg.inv(g)
        return jnp.linalg.norm(H(g @ s) - gi.T @ H0 @ gi)

    return jnp.max(jax.vmap(dev)(group)) / scale


eq_nn = jax.vmap(lambda s: hess_equivariance(psi_nn, s))(ss_te)
eq_poly = jax.vmap(lambda s: hess_equivariance(psi_poly, s))(ss_te)
print(f"D6 emergence (gauge-invariant, relative Hessian equivariance defect):")
print(f"    NN (no D6 imposed): mean {float(jnp.mean(eq_nn)):.2e}  max {float(jnp.max(eq_nn)):.2e}")
print(f"    poly (D6 built in): mean {float(jnp.mean(eq_poly)):.2e}  = the numerical floor")

# ---- lambda1
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
w = laplace_spectrum(slice_potential(chart, psi_nn), basis, nodes, weights=wts)
print(f"lambda1 (NN, no D6): {float(w[1]) * 4 / S_mean:.5f}  (poly/DHHKW 6.3228)")
