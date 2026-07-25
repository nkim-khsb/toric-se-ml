"""Grade the learned dP3 metric against DHHKW's Laplacian eigenvalue lambda1.

Step 1 (VALIDATE the tool): conifold transverse base = CP^1 x CP^1 (KE, psi=0).
  Closed form in our doubled-slice normalization (S=12 => each S^2 has K=3):
  lowest torus-invariant eigenvalue lambda1 = 6 exactly. The tool must reproduce
  it before we trust it on dP3.
Step 2 (GRADE dP3): train the D6-invariant metric, compute lambda1, lambda2 in
  our normalization, reconcile via the dimensionless lambda/S, compare to
  DHHKW lambda1 = 6.322, lambda2 = 17.2  (their Ric=g => S=4; ours S=12,
  so lambda_ours * 4/S_ours  should match DHHKW).

Usage: PYTHONPATH=. python experiments/dp3/lambda_compare.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.laplacian import (laplace_spectrum, polygon_quadrature,
                                slice_potential)
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, loss_fn, residual_on_slice,
    sample_slice, slice_chart, sym_ortho_psi, t_of_s, whiten_sym_poly,
)

jax.config.update("jax_enable_x64", True)


def svd_ortho(raw, samples, tol=1e-9):
    """basis(s) = raw(s) @ W with columns orthonormal over `samples`."""
    A = jax.vmap(raw)(samples)
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))
    W = Vt[:r].T / S[:r]
    return (lambda s: raw(s) @ W), r


def poly_basis(degree, samples):
    powers = [(i, j) for tot in range(degree + 1)
              for i in range(tot + 1) for j in [tot - i]]  # incl. const+linear
    raw = lambda s: jnp.stack([s[0] ** i * s[1] ** j for i, j in powers])
    return svd_ortho(raw, samples)


def sym_basis(degree, samples, group, tol=1e-9):
    powers = _poly_powers(degree)  # deg >= 2

    def raw(s):
        gs = jnp.einsum("gij,j->gi", group, s)
        sym = jnp.mean(jax.vmap(lambda g: _monomials(g, powers))(gs), axis=0)
        return jnp.concatenate([jnp.ones(1), sym])  # + constant

    return svd_ortho(raw, samples, tol=tol)


def abreu_S(chart, psi, s):
    u = slice_potential(chart, psi)
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)
    return -jnp.einsum("jkjk->", T)


NGL = 24         # 1d Gauss-Legendre nodes / triangle for the polygon quadrature
CLOUD = 40000    # MC cloud, used ONLY to SVD-orthonormalize the basis (conditioning)
# Integration is deterministic (polygon_quadrature): converged to ~1e-13, with no
# boundary margin -- the stiffness integrand u^{jk} vanishes on partial P on its
# own (log.md 2026-07-24).  Replaces the old eps-margin Monte-Carlo (0.46% wall).

# ============================ Step 1: validate on conifold (CP^1 x CP^1) =====
print(f"== validate Laplacian tool: conifold base CP^1 x CP^1 (psi=0), "
      f"deterministic quadrature ngl={NGL} ==")
ch_c = slice_chart(conifold(), B_CONIFOLD)
smp_c = sample_slice(jax.random.PRNGKey(5), ch_c, CLOUD, eps=1e-4)
quad_c = polygon_quadrature(ch_c.verts_s, ngl=NGL)
u_c = slice_potential(ch_c, lambda s: 0.0)
S_c = float(jnp.mean(jax.vmap(lambda s: abreu_S(ch_c, lambda s: 0.0, s))(
    sample_slice(jax.random.PRNGKey(9), ch_c, 200, eps=1e-2))))
for deg in (6,):
    basis, r = poly_basis(deg, smp_c)
    w = laplace_spectrum(u_c, basis, quad_c[0], weights=quad_c[1])
    print(f"  poly deg {deg} (dim {r}): lambda1={w[1]:.8f}  (closed form 6.000, "
          f"calibration error {abs(w[1] / 6 - 1) * 100:.5f}%)  S={S_c:.3f}")

# ============================ Step 2: grade dP3 ==============================
print("\n== dP3: train D6-invariant metric (deg 14) ==")
ch = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(ch.verts_s)
ss = sample_slice(jax.random.PRNGKey(1), ch, 2048, eps=2e-3)
powers, W, grp = whiten_sym_poly(14, ss, group)
nc = W.shape[1]


def L(v):
    return loss_fn(ch, sym_ortho_psi(v[:nc], powers, W, grp), v[nc], ss)


vg = jax.jit(jax.value_and_grad(L))
r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
v0 = np.zeros(nc + 1)
v0[nc] = -float(jnp.mean(r0))
res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
               v0, jac=True, method="L-BFGS-B",
               options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
v = jnp.asarray(res.x)
psi = sym_ortho_psi(v[:nc], powers, W, grp)
print(f"  trained loss {res.fun:.2e}  ({res.nit} iters)")

u = slice_potential(ch, psi)
smp = sample_slice(jax.random.PRNGKey(5), ch, CLOUD, eps=1e-4)  # basis conditioning
nodes, wts = polygon_quadrature(ch.verts_s, ngl=NGL)            # deterministic integ.
S_dp3 = float(jnp.mean(jax.vmap(lambda s: abreu_S(ch, psi, s))(
    sample_slice(jax.random.PRNGKey(9), ch, 200, eps=1e-2))))

print(f"\n== dP3 Laplacian spectrum (D6-invariant sector), S={S_dp3:.4f} ==")
print(f"  reconcile to DHHKW (Ric=g, S=4): lambda_DHHKW = lambda_ours * 4 / S")
for deg in (20,):  # basis-converged (matches deg 12/28 to >8 digits)
    basis, r = sym_basis(deg, smp, group, tol=1e-12)
    w = laplace_spectrum(u, basis, nodes, weights=wts)
    l1, l2 = w[1] * 4 / S_dp3, w[2] * 4 / S_dp3
    print(f"  basis deg {deg} (dim {r}): "
          f"lambda1 = {l1:.6f}  (DHHKW 6.322, {abs(l1 / 6.322 - 1) * 100:.3f}%);  "
          f"lambda2 = {l2:.5f}  (DHHKW 17.2, {abs(l2 / 17.2 - 1) * 100:.3f}%)")
