"""Figures for the dP3 note: learned psi + Abreu S on the hexagon, and the
quadrature calibration on the conifold closed-form anchor.

The dP3 eigenvalues lambda1, lambda2 are still computed and printed -- they are
the numbers the paper quotes in sections 2.3 and 4.3 -- but they are no longer
plotted.  The bar chart that compared them with DHHKW was dropped from the
paper: at 0.01% and 0.62% the two pairs of bars were indistinguishable, so the
figure carried nothing that table 8 does not carry better.

Each figure is written both here and into the draft directory, so that the
script can regenerate the figures the paper actually embeds.

Usage: PYTHONPATH=. python experiments/dp3/figures.py
"""
# The draft tree is written to as well when it is present, so that one run
# regenerates the figures the paper embeds.  It is absent in the code release,
# and then only the local copy is written.
DRAFT = "writing/paper1-new/"
import jax
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


from pathlib import Path

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.artifacts import load_psi_dp3
from sugrasol.laplacian import (laplace_spectrum, polygon_quadrature,
                                slice_potential)
from sugrasol.ypq import (
    _monomials, _poly_powers, dihedral_matrices, sample_slice, slice_chart,
    t_of_s,
)

jax.config.update("jax_enable_x64", True)
BLUE, RED = "#4c72b0", "#c44e52"

# --------------------------------------------------- load the persisted dP3
# The artifact IS the metric quoted in the paper; refitting here would risk
# drifting from it (and, before 2026-07-26, silently refit at truncated rank).
NPZ = (Path(__file__).resolve().parents[1] / "dp3smooth" / "dp3_G_deg18.npz")
psi, ch, _dat = load_psi_dp3(NPZ)
group = dihedral_matrices(ch.verts_s)
print(f"dP3 deg-{int(_dat['degree'])}: {_dat['W'].shape[1]} params, "
      f"held-out loss {float(_dat['loss_test']):.2e}")


# ------------------------------------------------------------------- cache
# The two costly blocks -- the Abreu grid and the quadrature calibration --
# together take minutes, which makes re-plotting (a legend, a colour, a label)
# far more expensive than it should be.  They are cached in an npz keyed on
# every input they depend on: the persisted metric, the grid size, and the
# three sweeps.  Any change to those, or --refresh, recomputes.
GRID = 240
CLOUD, NGL = 40000, 24   # MC cloud for basis conditioning; GL nodes/tri
MC_N = [2500, 5000, 10000, 20000, 40000, 80000]
GL_NGL = [4, 8, 12, 16, 24, 32]
CACHE = Path(__file__).resolve().parent / "figures_cache.npz"
REFRESH = "--refresh" in sys.argv
_st = NPZ.stat()
STAMP = "|".join(map(str, (_st.st_mtime_ns, _st.st_size, GRID, CLOUD, NGL,
                           MC_N, GL_NGL, 6, 20)))
_c = None
if CACHE.exists() and not REFRESH:
    _z = np.load(CACHE, allow_pickle=False)
    if str(_z["stamp"]) == STAMP:
        _c = _z
        print(f"cache hit: {CACHE.name} (pass --refresh to recompute)")
    else:
        print("cache stamp differs from the current inputs; recomputing")


def _targets(name):
    """Where a figure is written: here always, the draft tree if it exists."""
    out = ["experiments/dp3/" + name]
    if Path(DRAFT).is_dir():
        out.append(DRAFT + name)
    return out


def abreu_S(s):
    u = slice_potential(ch, psi)
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)
    return -jnp.einsum("jkjk->", T)


# ------------------------------------------------------------- grid on hexagon
verts = np.array(ch.verts_s)
lo, hi = verts.min(0) - 0.02, verts.max(0) + 0.02
gx = np.linspace(lo[0], hi[0], GRID)
gy = np.linspace(lo[1], hi[1], GRID)
GX, GY = np.meshgrid(gx, gy)
if _c is not None:
    inside, Zpsi, ZS = _c["inside"], _c["Zpsi"], _c["ZS"]
else:
    pts = jnp.array(np.stack([GX.ravel(), GY.ravel()], axis=1))
    inside = jax.vmap(lambda s: ch.cone.is_interior(t_of_s(ch, s), 1e-3))(pts)
    inside = np.array(inside).reshape(GX.shape)
    Zpsi = np.array(jax.vmap(psi)(pts)).reshape(GX.shape)
    ZS = np.array(jax.vmap(abreu_S)(pts)).reshape(GX.shape)
    Zpsi[~inside] = np.nan
    ZS[~inside] = np.nan
_dev = np.abs(ZS[inside] / 12.0 - 1.0)
print(f"Abreu S on the plotted region: max |S/12-1| = {_dev.max() * 100:.4f}%, "
      f"mean {_dev.mean() * 100:.5f}%  ({inside.sum()} grid points)")


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
# Window must CONTAIN the worst deviation, or the extreme points saturate the
# colour scale and the caption's number is invisible.  DERIVED from the measured
# deviation, not hardcoded: a fixed window silently belongs to whichever rung it
# was tuned on (a hardcoded +-0.003 for deg-14 outlived the switch to deg-18 and
# left the caption quoting the wrong metric's accuracy).
_w = 1.15 * _dev.max() * 12.0
print(f"Abreu colour window: 12 +- {_w:.2e}  (caption must quote "
      f"[{12 - _w:.5f}, {12 + _w:.5f}])")
c2 = a2.contourf(GX, GY, ZS, levels=np.linspace(12 - _w, 12 + _w, 25), cmap="RdBu_r",
                 extend="both")
hexagon(a2)
fig.colorbar(c2, ax=a2, fraction=0.046, ticks=[12 - _w, 12.0, 12 + _w],
             format="%.5f")
a2.set_title(r"Abreu scalar $S$ (KE $\Rightarrow S\equiv 12$)")
fig.tight_layout()
for _p in _targets("fig_dp3_learned.png"):
    fig.savefig(_p, dpi=160)
    print("wrote", _p)

# ------------------------------------------------------------- lambda grading
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
# quadrature.  MC is the RMS over seeds, so it carries the boundary-exclusion
# bias as well as the variance; measured, it falls as N^{-0.69} across this
# range and is still ~0.3% at 8e4 points.  Deterministic polygon quadrature
# reaches machine precision with a few hundred nodes.
chc = slice_chart(conifold(), B_CONIFOLD)
uc = slice_potential(chc, lambda s: 0.0)

mc_N = MC_N
if _c is not None:
    mc_err, q_pts, q_err = list(_c["mc_err"]), list(_c["q_pts"]), list(_c["q_err"])
    l1, l2 = float(_c["l1"]), float(_c["l2"])
else:
    mc_err = []  # RMS error over seeds = sqrt(bias^2 + variance): one positive
                 # number, no log-axis underflow (a symmetric std whisker would
                 # dive to ~0)
    for N in mc_N:
        vals = []
        for seed in range(8):
            smp = sample_slice(jax.random.PRNGKey(seed), chc, N, eps=1e-4)
            vals.append(float(laplace_spectrum(uc, poly_basis(6, smp), smp)[1]))
        vals = np.array(vals)
        mc_err.append(float(np.sqrt(np.mean((vals - 6.0) ** 2))))

    # deterministic: basis conditioned on one cloud, integrated on the GL nodes
    basis_c = poly_basis(6, sample_slice(jax.random.PRNGKey(5), chc, CLOUD, eps=1e-4))
    q_pts, q_err = [], []
    for ngl in GL_NGL:
        nodes, wts = polygon_quadrature(chc.verts_s, ngl=ngl)
        lam = float(laplace_spectrum(uc, basis_c, nodes, weights=wts)[1])
        q_pts.append(int(nodes.shape[0]))
        q_err.append(max(abs(lam - 6.0), 1e-16))   # floor for the log axis

    # dP3 lambda1, lambda2 (deterministic integration, reconciled to DHHKW)
    u = slice_potential(ch, psi)
    smp = sample_slice(jax.random.PRNGKey(5), ch, CLOUD, eps=1e-4)
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=NGL)
    w = laplace_spectrum(u, sym_basis(20, smp), nodes, weights=wts)
    l1, l2 = float(w[1]) * 4 / 12, float(w[2]) * 4 / 12

    np.savez_compressed(CACHE, stamp=STAMP, inside=inside, Zpsi=Zpsi, ZS=ZS,
                        mc_err=np.array(mc_err), q_pts=np.array(q_pts),
                        q_err=np.array(q_err), l1=l1, l2=l2)
    print(f"wrote {CACHE}")
print("MC RMS|lam1-6| vs N:", [f"{e:.1e}" for e in mc_err])
print("quad |lam1-6| vs pts:", [f"{e:.1e}" for e in q_err])
print(f"dP3 lambda1={l1:.6f} (DHHKW 6.322), lambda2={l2:.5f} (DHHKW 17.2)")

fig, a1 = plt.subplots(figsize=(5.5, 4.4))
a1.plot(mc_N, mc_err, "o-", color=BLUE, label="Monte-Carlo (RMS over 8 seeds)")
a1.plot(q_pts, q_err, "s-", color=RED, label="deterministic quadrature")
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlabel(r"integration points")
a1.set_ylabel(r"error in $\lambda_1$  (conifold, exact $=6$)")
a1.set_title(r"anchor: conifold $\mathrm{CP}^1\times\mathrm{CP}^1$")
# The two series occupy the upper right (MC, y > 1e-2) and the lower left
# (quadrature, x < 5e3).  Both corners on the diagonal between them are
# therefore taken; the empty band is the middle left, around y ~ 1e-7.
a1.legend(fontsize=9, loc="center left")
fig.tight_layout()
for _p in _targets("fig_quadrature.png"):
    fig.savefig(_p, dpi=160)
    print("wrote", _p)
