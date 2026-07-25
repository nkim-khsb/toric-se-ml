"""Figures for the dP3 note: learned psi + Abreu S on the hexagon, and the
Laplacian lambda1 grading against DHHKW (with the conifold closed-form anchor).

Usage: PYTHONPATH=. python experiments/dp3/figures.py
"""
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.optimize import minimize

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.laplacian import (laplace_spectrum, polygon_quadrature,
                                slice_potential)
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, loss_fn, residual_on_slice,
    sample_slice, slice_chart, sym_ortho_psi, t_of_s, whiten_sym_poly,
)

jax.config.update("jax_enable_x64", True)
BLUE, RED = "#4c72b0", "#c44e52"

# ------------------------------------------------------------- train dP3
ch = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(ch.verts_s)
ss = sample_slice(jax.random.PRNGKey(1), ch, 2048, eps=2e-3)
powers, W, grp = whiten_sym_poly(14, ss, group)
nc = W.shape[1]


def Lf(v):
    return loss_fn(ch, sym_ortho_psi(v[:nc], powers, W, grp), v[nc], ss)


vg = jax.jit(jax.value_and_grad(Lf))
r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
v0 = np.zeros(nc + 1)
v0[nc] = -float(jnp.mean(r0))
res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
               v0, jac=True, method="L-BFGS-B",
               options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
v = jnp.asarray(res.x)
psi = sym_ortho_psi(v[:nc], powers, W, grp)
print(f"dP3 trained loss {res.fun:.2e}")


def abreu_S(s):
    u = slice_potential(ch, psi)
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)
    return -jnp.einsum("jkjk->", T)


# ------------------------------------------------------------- grid on hexagon
verts = np.array(ch.verts_s)
lo, hi = verts.min(0) - 0.02, verts.max(0) + 0.02
gx = np.linspace(lo[0], hi[0], 240)
gy = np.linspace(lo[1], hi[1], 240)
GX, GY = np.meshgrid(gx, gy)
pts = jnp.array(np.stack([GX.ravel(), GY.ravel()], axis=1))
inside = jax.vmap(lambda s: ch.cone.is_interior(t_of_s(ch, s), 1e-3))(pts)
inside = np.array(inside).reshape(GX.shape)
Zpsi = np.array(jax.vmap(psi)(pts)).reshape(GX.shape)
ZS = np.array(jax.vmap(abreu_S)(pts)).reshape(GX.shape)
Zpsi[~inside] = np.nan
ZS[~inside] = np.nan


def hexagon(ax):
    order = np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))
    ax.add_patch(Polygon(verts[order], fill=False, ec="k", lw=1.4, zorder=5))
    ax.scatter(verts[:, 0], verts[:, 1], c="k", s=22, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")


# ---- Figure 1: learned psi + Abreu S
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
c1 = a1.contourf(GX, GY, Zpsi, levels=18, cmap="RdBu_r")
a1.contour(GX, GY, Zpsi, levels=18, colors="k", linewidths=0.3, alpha=0.4)
hexagon(a1)
fig.colorbar(c1, ax=a1, fraction=0.046)
a1.set_title(r"learned correction $\psi$ (D6-invariant)")
c2 = a2.contourf(GX, GY, ZS, levels=np.linspace(11.99, 12.01, 21), cmap="RdBu_r",
                 extend="both")
hexagon(a2)
fig.colorbar(c2, ax=a2, fraction=0.046, ticks=[11.99, 12.0, 12.01])
a2.set_title(r"Abreu scalar $S$ (KE $\Rightarrow S\equiv 12$)")
fig.tight_layout()
fig.savefig("experiments/dp3/fig_dp3_learned.png", dpi=160)
print("wrote fig_dp3_learned.png")

# ------------------------------------------------------------- lambda grading
CLOUD, NGL = 40000, 24  # MC cloud for basis conditioning; GL nodes/tri for integ.


def poly_basis(degree, samples, tol=1e-9):
    pw = [(i, j) for tot in range(degree + 1) for i in range(tot + 1) for j in [tot - i]]
    raw = lambda s: jnp.stack([s[0] ** i * s[1] ** j for i, j in pw])
    A = jax.vmap(raw)(samples)
    _, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))
    Wm = Vt[:r].T / S[:r]
    return lambda s: raw(s) @ Wm


def sym_basis(degree, samples, tol=1e-12):
    pw = _poly_powers(degree)

    def raw(s):
        gs = jnp.einsum("gij,j->gi", group, s)
        sym = jnp.mean(jax.vmap(lambda g: _monomials(g, pw))(gs), axis=0)
        return jnp.concatenate([jnp.ones(1), sym])

    A = jax.vmap(raw)(samples)
    _, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))
    Wm = Vt[:r].T / S[:r]
    return lambda s: raw(s) @ Wm


# conifold anchor: convergence of |lambda1 - 6| for Monte-Carlo vs deterministic
# quadrature.  MC plateaus (variance + boundary-exclusion bias, error bars over
# seeds); deterministic polygon quadrature drops to machine precision.
chc = slice_chart(conifold(), B_CONIFOLD)
uc = slice_potential(chc, lambda s: 0.0)

mc_N = [2500, 5000, 10000, 20000, 40000, 80000]
mc_err = []  # RMS error over seeds = sqrt(bias^2 + variance): one positive number,
             # no log-axis underflow (a symmetric std whisker would dive to ~0)
for N in mc_N:
    vals = []
    for seed in range(8):
        smp = sample_slice(jax.random.PRNGKey(seed), chc, N, eps=1e-4)
        vals.append(float(laplace_spectrum(uc, poly_basis(6, smp), smp)[1]))
    vals = np.array(vals)
    mc_err.append(float(np.sqrt(np.mean((vals - 6.0) ** 2))))
print("MC RMS|lam1-6| vs N:", [f"{e:.1e}" for e in mc_err])

# deterministic: basis conditioned on one cloud, integrated on the GL nodes
basis_c = poly_basis(6, sample_slice(jax.random.PRNGKey(5), chc, CLOUD, eps=1e-4))
gl_ngl = [4, 8, 12, 16, 24, 32]
q_pts, q_err = [], []
for ngl in gl_ngl:
    nodes, wts = polygon_quadrature(chc.verts_s, ngl=ngl)
    lam = float(laplace_spectrum(uc, basis_c, nodes, weights=wts)[1])
    q_pts.append(nodes.shape[0])
    q_err.append(max(abs(lam - 6.0), 1e-16))   # floor for the log axis
print("quad |lam1-6| vs pts:", [f"{e:.1e}" for e in q_err])

# dP3 lambda1, lambda2 (deterministic integration, reconciled to DHHKW)
u = slice_potential(ch, psi)
smp = sample_slice(jax.random.PRNGKey(5), ch, CLOUD, eps=1e-4)  # basis conditioning
nodes, wts = polygon_quadrature(ch.verts_s, ngl=NGL)
w = laplace_spectrum(u, sym_basis(20, smp), nodes, weights=wts)
l1, l2 = float(w[1]) * 4 / 12, float(w[2]) * 4 / 12
print(f"dP3 lambda1={l1:.6f} (DHHKW 6.322), lambda2={l2:.5f} (DHHKW 17.2)")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
a1.plot(mc_N, mc_err, "o-", color=BLUE, label="Monte-Carlo (RMS over 8 seeds)")
a1.plot(q_pts, q_err, "s-", color=RED, label="deterministic quadrature")
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlabel(r"integration points")
a1.set_ylabel(r"error in $\lambda_1$  (conifold, exact $=6$)")
a1.set_title(r"anchor: conifold $\mathrm{CP}^1\times\mathrm{CP}^1$")
a1.legend(fontsize=9, loc="lower left")
x = np.arange(2)
a2.bar(x - 0.2, [l1, l2], 0.4, color=RED, label="ours (learned)")
a2.bar(x + 0.2, [6.322, 17.2], 0.4, color=BLUE, label="DHHKW (numerical KE)")
for i, (o, d) in enumerate([(l1, 6.322), (l2, 17.2)]):
    a2.text(i, max(o, d) + 0.5, f"{abs(o/d-1)*100:.2f}%", ha="center", fontsize=9)
a2.set_xticks(x)
a2.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$"])
a2.set_ylabel(r"eigenvalue (DHHKW normalization)")
a2.set_title(r"dP3 $D6$-invariant Laplacian spectrum")
a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig("experiments/dp3/fig_dp3_lambda.png", dpi=160)
print("wrote fig_dp3_lambda.png")
