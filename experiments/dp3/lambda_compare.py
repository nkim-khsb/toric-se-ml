"""Grade the learned dP3 metric against DHHKW's Laplacian eigenvalues.

Step 1 (VALIDATE the tool): conifold transverse base = CP^1 x CP^1 (KE, psi=0).
  Closed form in our doubled-slice normalization (S=12 => each S^2 has K=3):
  lowest torus-invariant eigenvalue lambda1 = 6 exactly. The tool must reproduce
  it before we trust it on dP3.
Step 2 (GRADE dP3): load the persisted D6-invariant metrics (no retraining),
  compute lambda1, lambda2 in our normalization, reconcile via the dimensionless
  lambda/S, compare to DHHKW lambda1 = 6.322, lambda2 = 17.2  (their Ric=g =>
  S=4; ours S=12, so lambda_ours * 4/S_ours  should match DHHKW).
Step 3 (CONVERGENCE LADDER): the paper claims our lambda2 CORRECTS theirs, so
  our own lambda2 needs the convergence evidence that lambda1 already has. We
  vary all three knobs independently -- metric degree (14 / 18), Rayleigh-Ritz
  basis degree, and quadrature order ngl -- and print the full table.
  Rayleigh-Ritz converges FROM ABOVE, so along the basis-degree direction the
  eigenvalues must decrease monotonically; a violation is a quadrature (not a
  basis) failure, and is checked and reported as such.

Usage: PYTHONPATH=. python experiments/dp3/lambda_compare.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.artifacts import load_psi_dp3
from sugrasol.laplacian import (laplace_spectrum, polygon_quadrature,
                                slice_potential)
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, sample_slice, slice_chart,
)

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
NPZ = {"deg-14": ROOT / "experiments/dp3smooth/dp3_G_deg14.npz",
       "deg-18": ROOT / "experiments/dp3smooth/dp3_G_deg18.npz"}


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
# The persisted artifacts ARE the metrics quoted in the paper; no retraining.
ch = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(ch.verts_s)
smp = sample_slice(jax.random.PRNGKey(5), ch, CLOUD, eps=1e-4)  # basis conditioning
nodes, wts = polygon_quadrature(ch.verts_s, ngl=NGL)            # deterministic integ.
s_probe = sample_slice(jax.random.PRNGKey(9), ch, 200, eps=1e-2)

fits = {}
for name, path in NPZ.items():
    psi, _, dat = load_psi_dp3(path)
    S = float(jnp.mean(jax.vmap(lambda s: abreu_S(ch, psi, s))(s_probe)))
    fits[name] = (slice_potential(ch, psi), S, int(dat["W"].shape[1]))
    print(f"\n== dP3 {name}: {fits[name][2]} params, held-out MA loss "
          f"{float(dat['loss_test']):.2e}, Abreu S = {S:.6f} ==")

print("\n== dP3 Laplacian spectrum (D6-invariant sector) ==")
print("  reconcile to DHHKW (Ric=g, S=4): lambda_DHHKW = lambda_ours * 4 / S")
for name, (u, S, _) in fits.items():
    basis, r = sym_basis(20, smp, group, tol=1e-12)
    w = laplace_spectrum(u, basis, nodes, weights=wts)
    l1, l2 = w[1] * 4 / S, w[2] * 4 / S
    print(f"  {name}, basis deg 20 (dim {r}): "
          f"lambda1 = {l1:.6f}  (DHHKW 6.322, {abs(l1 / 6.322 - 1) * 100:.3f}%);  "
          f"lambda2 = {l2:.5f}  (DHHKW 17.2, {abs(l2 / 17.2 - 1) * 100:.3f}%)")

# ======================= Step 3: three-way convergence ladder ================
# Knobs: metric degree (14/18) x Rayleigh-Ritz basis degree x quadrature ngl.
# Rayleigh-Ritz bounds eigenvalues FROM ABOVE, so at fixed metric and ngl the
# basis-degree direction must be non-increasing; an increase means the
# quadrature (not the trial space) is the limiting error.
BASIS_DEGS = (10, 12, 14, 16, 20)
NGLS = (16, 24, 32)
print("\n== Step 3: convergence ladder for lambda1, lambda2 "
      "(converted to DHHKW's Ric=g) ==")
table = {}
for name, (u, S, npar) in fits.items():
    print(f"\n  --- metric {name} ({npar} params), S = {S:.6f} ---")
    print(f"  {'basis':>6} {'dim':>4} " +
          "".join(f"  ngl={n:<2d}: {'lambda1':>10} {'lambda2':>10}" for n in NGLS))
    for bdeg in BASIS_DEGS:
        basis, r = sym_basis(bdeg, smp, group, tol=1e-12)
        row = f"  {bdeg:6d} {r:4d} "
        for ngl in NGLS:
            nd, wt = polygon_quadrature(ch.verts_s, ngl=ngl)
            w = laplace_spectrum(u, basis, nd, weights=wt)
            l1, l2 = float(w[1] * 4 / S), float(w[2] * 4 / S)
            table[(name, bdeg, ngl)] = (l1, l2)
            row += f"          {l1:10.6f} {l2:10.5f}"
        print(row)

print("\n  monotonicity in basis degree (Rayleigh-Ritz decreases from above):")
for name in fits:
    for ngl in NGLS:
        seq1 = [table[(name, b, ngl)][0] for b in BASIS_DEGS]
        seq2 = [table[(name, b, ngl)][1] for b in BASIS_DEGS]
        ok = all(a >= b - 1e-12 for a, b in zip(seq1, seq1[1:])) and \
             all(a >= b - 1e-12 for a, b in zip(seq2, seq2[1:]))
        print(f"    {name} ngl={ngl:2d}: {'OK' if ok else 'VIOLATED'}  "
              f"lambda2 {seq2[0]:.5f} -> {seq2[-1]:.5f} "
              f"(drop {seq2[0] - seq2[-1]:.2e})")

print("\n  spread of the converged corner (basis >= 14, all ngl):")
for name in fits:
    v1 = [table[(name, b, n)][0] for b in (14, 16, 20) for n in NGLS]
    v2 = [table[(name, b, n)][1] for b in (14, 16, 20) for n in NGLS]
    print(f"    {name}: lambda1 = {np.mean(v1):.6f} +- {np.ptp(v1):.1e},  "
          f"lambda2 = {np.mean(v2):.5f} +- {np.ptp(v2):.1e}")
v1 = [table[(m, b, n)][0] for m in fits for b in (14, 16, 20) for n in NGLS]
v2 = [table[(m, b, n)][1] for m in fits for b in (14, 16, 20) for n in NGLS]
print(f"    across BOTH metrics: lambda1 = {np.mean(v1):.6f} +- {np.ptp(v1):.1e}, "
      f"lambda2 = {np.mean(v2):.5f} +- {np.ptp(v2):.1e}")
print("  success criterion (plan-referee-numbers.md §2): lambda2 stable to "
      "~1e-3 in all three directions -> keep the 'corrected lambda2' claim.")
