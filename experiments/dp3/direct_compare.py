"""dP3: (a) 3D surface of the learned potential psi; (b) direct pointwise
comparison against DHHKW via the lowest D6-invariant Laplacian eigenfunction.

DHHKW give (eq. 6.7/6.9) psi_1 = 1/10 + X1 U + X2 U^2 + X3 U^3 + Y1 V with
X1=-0.245, X2=0.006 in the D6-invariant coordinate U = x1^2+x1x2+x2^2, which on
their symmetric hexagon runs 0 (centre) -> 1 (vertices).  The eigenfunction has
no gauge freedom (fixed by psi_1(0)=0.1) and no metric-scale ambiguity, so it is
the cleanest DHHKW datum to overlay.  We compute our eigenfunction, express it
in our matched U, and compare X1, X2.

Usage: PYTHONPATH=. python experiments/dp3/direct_compare.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from sugrasol.cone import B_DP3, dp3
from sugrasol.artifacts import load_psi_dp3
from sugrasol.laplacian import slice_potential
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, sample_slice, slice_chart,
    t_of_s,
)

jax.config.update("jax_enable_x64", True)
BLUE, RED = "#4c72b0", "#c44e52"

# ---- the persisted D6-invariant metric (deg 14); no refit, so that this
# figure and the quoted numbers cannot drift from the artifact
NPZ = Path(__file__).resolve().parents[1] / "dp3smooth" / "dp3_G_deg14.npz"
psi, ch, dat = load_psi_dp3(NPZ)
group = dihedral_matrices(ch.verts_s)
verts = np.array(ch.verts_s)
print(f"deg-{int(dat['degree'])} metric: {dat['W'].shape[1]} params, "
      f"held-out loss {float(dat['loss_test']):.2e}")

# ---- our D6-invariant quadratic U(s), normalized U=0 centre, U=1 at vertices
G = np.array(group)
M = sum(g.T @ g for g in G)                      # Reynolds average -> D6-invariant form
uvert = float(verts[0] @ M @ verts[0])           # equal on all 6 vertices (one orbit)
U = lambda s: (s @ jnp.asarray(M) @ s) / uvert
print(f"U at the 6 vertices (should be equal): "
      f"{[round(float(verts[i] @ M @ verts[i] / uvert), 4) for i in range(6)]}")

# ============================ (a) 3D surface of psi ==========================
gx = np.linspace(verts[:, 0].min(), verts[:, 0].max(), 200)
gy = np.linspace(verts[:, 1].min(), verts[:, 1].max(), 200)
GX, GY = np.meshgrid(gx, gy)
pts = jnp.array(np.stack([GX.ravel(), GY.ravel()], 1))
inside = np.array(jax.vmap(lambda s: ch.cone.is_interior(t_of_s(ch, s), 1e-3))(pts)).reshape(GX.shape)
Z = np.array(jax.vmap(psi)(pts)).reshape(GX.shape); Z[~inside] = np.nan
fig = plt.figure(figsize=(6.2, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(GX, GY, Z, cmap="RdBu_r", linewidth=0, antialiased=True, alpha=0.95)
ax.set_xlabel(r"$s_1$"); ax.set_ylabel(r"$s_2$"); ax.set_zlabel(r"$\psi$")
ax.set_title(r"dP3 learned potential $\psi(s_1,s_2)$ (D6-invariant)")
fig.tight_layout(); fig.savefig("experiments/dp3/fig_dp3_psi3d.png", dpi=160)
print("wrote fig_dp3_psi3d.png")

# ============================ (b) eigenfunction vs DHHKW =====================
# D6-invariant basis (constant + symmetrized monomials), generalized eigenproblem
pw = _poly_powers(20)


def raw(s):
    gs = jnp.einsum("gij,j->gi", jnp.asarray(G), s)
    sym = jnp.mean(jax.vmap(lambda g: _monomials(g, pw))(gs), axis=0)
    return jnp.concatenate([jnp.ones(1), sym])


smp = sample_slice(jax.random.PRNGKey(5), ch, 40000, eps=1e-4)
A_raw = jax.vmap(raw)(smp)
_, Sv, Vt = jnp.linalg.svd(A_raw, full_matrices=False)
r = int(jnp.sum(Sv > 1e-12 * Sv[0])); Wb = Vt[:r].T / Sv[:r]
basis = lambda s: raw(s) @ Wb

u = slice_potential(ch, psi)
uinv = lambda s: jnp.linalg.inv(jax.hessian(u)(s))
jac = jax.jacfwd(basis)
Am = np.asarray(jnp.mean(jax.vmap(lambda s: jac(s) @ uinv(s) @ jac(s).T)(smp), 0))
Bm = np.asarray(jnp.mean(jax.vmap(lambda s: jnp.outer(basis(s), basis(s)))(smp), 0))
w, vec = eigh(0.5 * (Am + Am.T), 0.5 * (Bm + Bm.T))
c1 = jnp.asarray(vec[:, 1])                       # eigenvector for lambda1 (vec[:,0]=const)
eig = lambda s: basis(s) @ c1
eig0 = float(eig(jnp.zeros(2)))
psi1 = lambda s: 0.1 * eig(s) / eig0              # DHHKW normalization psi1(0)=0.1

# fit psi1 ~ a0 + X1 U + X2 U^2 on the inner region (U<0.5, U-only regime)
Us = jax.vmap(U)(smp); f1 = jax.vmap(psi1)(smp)
m = np.array(Us) < 0.5
Xd = np.stack([np.ones(m.sum()), np.array(Us)[m], np.array(Us)[m] ** 2], 1)
a0, X1, X2 = np.linalg.lstsq(Xd, np.array(f1)[m], rcond=None)[0]
print(f"\n(b) eigenfunction vs DHHKW (eq. 6.9):")
print(f"   ours:  psi1(0)={a0:.3f}  X1={X1:.3f}  X2={X2:.3f}")
print(f"   DHHKW: psi1(0)=0.100  X1=-0.245  X2=0.006")
print(f"   X1 agreement: {abs(X1 / -0.245 - 1) * 100:.1f}%")

# overlay plot: our psi1 vs U + DHHKW curve
fig, ax = plt.subplots(figsize=(6.2, 4.6))
oo = np.argsort(np.array(Us))
ax.scatter(np.array(Us)[oo][::40], np.array(f1)[oo][::40], s=8, alpha=0.4,
           color=RED, label="ours (learned eigenfunction)")
uu = np.linspace(0, 1, 100)
ax.plot(uu, 0.1 - 0.245 * uu + 0.006 * uu ** 2, color=BLUE, lw=2,
        label=r"DHHKW $0.1-0.245U+0.006U^2$")
ax.set_xlabel(r"$U$ (D6 invariant: 0 centre, 1 vertices)")
ax.set_ylabel(r"$\psi_1$ (lowest Laplacian eigenfunction)")
ax.set_title("dP3: eigenfunction overlay vs DHHKW (pointwise)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("experiments/dp3/fig_dp3_eigfn_compare.png", dpi=160)
print("wrote fig_dp3_eigfn_compare.png")
