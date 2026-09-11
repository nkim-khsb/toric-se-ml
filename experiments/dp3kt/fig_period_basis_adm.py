"""The period-fixed (hexagon) and parity-fixed (pentagon) bases of primitive
harmonic (1,1)-forms, recomputed in the ADMISSIBLE space of
r2b_stream_wachspress.py (review round 2, item 2).  Same rules as
fig_period_basis.py, whose forms came from the non-L^2 polynomial space and
are kept only as fig_period_basis_poly.png.

Rules (unchanged):
  hexagon  -- periods P_a = <m_a, a(V_a) - a(V_{a-1})>, a = rotated gradient of
              chi, fitted to the roots alpha_1, alpha_2, beta of K^perp = A_2 + A_1;
              the Gram P^T Q^+ P' of the representatives must be the root Gram.
  pentagon -- the +1 / -1 eigenvectors of the affine lattice involution on the
              two-dimensional solution space.
New here: the regularity diagnostic R_a = <n_a, a(V_a) - a(V_{a-1})> is now an
identity (affine traces), and the vertex values chi(V_a) themselves are printed:
in the admissible space they are the coefficients of the Wachspress directions.

Usage: PYTHONPATH=. python experiments/dp3kt/fig_period_basis_adm.py
"""
import itertools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments" / "dp3kt", ROOT / "experiments" / "dp2",
          ROOT / "experiments" / "dp3"):
    sys.path.insert(0, str(p))

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import dp2                                        # noqa: E402
import r2b_periods as per                                            # noqa: E402
from r2b_holomorphic import load_dp3                                 # noqa: E402
from r2b_stream_wachspress import wachspress_space, design           # noqa: E402
from r2b_stream_admissible import solve                              # noqa: E402
from harmonic_forms_dp2 import se_potential                          # noqa: E402

DRAFT = ROOT / "writing" / "paper1-new"
D = 16
NGL = 48


def kernel(ch, u, k, D=D, ngl=NGL):
    """The k lowest self-dual-energy stream potentials in the admissible space,
    as functions s -> chi(s) (jax), together with the energies."""
    B, K, rho = wachspress_space(ch.verts_s, D)
    X, Xs, smp, _ = design(ch, u, B, ngl)
    lam, coef, r = solve(X, Xs)
    chis = [(lambda s, c=jnp.asarray(coef[:, i]): jnp.dot(B(s), c)) for i in range(k)]
    return chis, lam, np.asarray(smp)


def hexagon_basis():
    ch, u, _, _ = load_dp3()
    chis, lam, _ = kernel(ch, u, 3)
    edges = per.edge_data()

    def alpha(chi, s):
        g = np.asarray(jax.grad(chi)(jnp.asarray(s, dtype=float)))
        return np.array([g[1], -g[0]])

    P = np.zeros((3, 6))
    Rreg = np.zeros((3, 6))
    for k, chi in enumerate(chis):
        for a, ed in enumerate(edges):
            da = alpha(chi, ed["p1"]) - alpha(chi, ed["p0"])
            P[k, a] = ed["e"] @ da
            n = np.array([-ed["w"][1], ed["w"][0]], float)      # normal to the edge tangent
            Rreg[k, a] = n @ da
    T, target, resid = per.root_representatives(P, per.intersection_matrix())
    Q = per.intersection_matrix()
    Gp = T @ P @ np.linalg.pinv(Q) @ P.T @ T.T
    print(f"  hexagon energies  {'  '.join(f'{v:.2e}' for v in lam[:4])}")
    print(f"  primitivity  max|sum_a P_a|/max|P| = {np.max(np.abs(P.sum(1)))/np.max(np.abs(P)):.1e}")
    print(f"  regularity R_a max/|P|max = {np.max(np.abs(Rreg))/np.max(np.abs(P)):.1e}   (identity in this space)")
    print(f"  root fit residual {resid:.2e};  Gram of the representatives:", flush=True)
    print("  " + np.array2string(Gp, precision=4, suppress_small=True).replace("\n", "\n  "), flush=True)
    mixed = [(lambda s, i=i: sum(T[i, j] * chis[j](s) for j in range(3))) for i in range(3)]
    # vertex values of the representatives (mod affine these are the Wachspress data)
    V = np.asarray(ch.verts_s)
    vv = np.array([[float(m(jnp.asarray(v))) for v in V] for m in mixed])
    print("  vertex values chi(V_a) of the three representatives:")
    print("  " + np.array2string(vv, precision=4, suppress_small=True).replace("\n", "\n  "), flush=True)
    return ch, mixed, Gp, resid, P


def pentagon_basis():
    ch, u, _ = se_potential()
    V = np.asarray(dp2().normals).astype(float)
    b, f, t0 = np.asarray(ch.b), np.asarray(ch.f), np.asarray(ch.t0)
    W = None
    for pm in itertools.permutations(range(5)):
        M, _, rank, _ = np.linalg.lstsq(V, V[list(pm)], rcond=None)
        if rank < 3 or np.max(np.abs(V @ M - V[list(pm)])) > 1e-9:
            continue
        Mi = np.rint(M)
        if np.max(np.abs(M - Mi)) > 1e-9 or abs(round(np.linalg.det(Mi))) != 1:
            continue
        if not np.array_equal(Mi, np.eye(3)):
            W = Mi
            break
    assert W is not None
    assert np.max(np.abs(b @ W - b)) < 1e-6
    R = f @ W @ f.T
    tau = f @ (W @ t0 - t0)
    Phi = lambda s: R @ s + tau
    verts = np.asarray(ch.verts_s)
    pm = [int(np.argmin(np.linalg.norm(verts - Phi(v), axis=1))) for v in verts]
    dev = max(np.linalg.norm(verts[pm[i]] - Phi(verts[i])) for i in range(5))
    print(f"\n  pentagon involution permutes the vertices as {pm}, to {dev:.1e}", flush=True)

    chis, lam, smp = kernel(ch, u, 2)
    HG = jax.jit(jax.hessian(u))
    sel = smp[::41]
    num = max(np.max(np.abs(R.T @ np.asarray(HG(jnp.asarray(Phi(p)))) @ R - np.asarray(HG(jnp.asarray(p))))) for p in sel)
    den = max(np.max(np.abs(np.asarray(HG(jnp.asarray(p))))) for p in sel)
    print(f"  Hess G_P invariance, relative {num/den:.1e}", flush=True)

    def hf(chi, pts):
        return np.asarray(jax.vmap(jax.hessian(chi))(jnp.asarray(pts)))

    Hs = [hf(c, smp) for c in chis]
    Hp = [np.einsum("ia,nab,bj->nij", R.T, hf(c, np.array([Phi(p) for p in smp])), R) for c in chis]
    col = lambda H: np.stack([H[:, 0, 0], H[:, 0, 1], H[:, 1, 1]], -1).ravel()
    Bm = np.stack([col(H) for H in Hs], 1)
    Bp = np.stack([col(H) for H in Hp], 1)
    S, *_ = np.linalg.lstsq(Bm, Bp, rcond=None)
    ev, EV = np.linalg.eig(S)
    print(f"  pentagon energies {'  '.join(f'{v:.2e}' for v in lam[:3])}")
    print(f"  pullback eigenvalues {np.round(ev.real, 7)}, |S^2-I| {np.max(np.abs(S@S-np.eye(2))):.1e}, "
          f"fit residual {np.linalg.norm(Bm@S-Bp)/np.linalg.norm(Bp):.1e}", flush=True)
    order = np.argsort(-ev.real)
    par = [(lambda s, i=i: sum(EV[j, order[i]].real * chis[j](s) for j in range(2))) for i in range(2)]
    return ch, par, ev


def field(ch, chi, n=220):
    verts = np.asarray(ch.verts_s)
    lo, hi = verts.min(0) - 0.02, verts.max(0) + 0.02
    gx = np.linspace(lo[0], hi[0], n)
    gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy)
    order = np.argsort(np.arctan2(verts[:, 1] - verts[:, 1].mean(), verts[:, 0] - verts[:, 0].mean()))
    Pg = verts[order]
    inside = np.ones(GX.shape, bool)
    for k in range(len(Pg)):
        a, b = Pg[k], Pg[(k + 1) % len(Pg)]
        inside &= ((b[0]-a[0])*(GY-a[1]) - (b[1]-a[1])*(GX-a[0])) >= -1e-12
    pts = jnp.asarray(np.stack([GX.ravel(), GY.ravel()], 1))
    Z = np.asarray(jax.vmap(chi)(pts)).reshape(GX.shape)
    Z = np.where(inside, Z, np.nan)
    return GX, GY, Z, Pg


def draw(ax, GX, GY, Z, Pg, title, vlim):
    lv = np.linspace(-vlim, vlim, 19)
    c = ax.contourf(GX, GY, Z, levels=lv, cmap="RdBu_r", extend="neither")
    ax.contour(GX, GY, Z, levels=lv, colors="k", linewidths=0.25, alpha=0.35)
    ax.add_patch(Polygon(Pg, fill=False, ec="k", lw=1.3, zorder=5))
    ax.scatter(Pg[:, 0], Pg[:, 1], c="k", s=14, zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)
    return c


if __name__ == "__main__":
    print("== admissible space, D = 16: period-fixed hexagon basis, parity-fixed pentagon basis ==", flush=True)
    ch6, hexb, Gp, resid, P = hexagon_basis()
    ch5, penb, ev = pentagon_basis()
    f6 = [field(ch6, c) for c in hexb]
    f5 = [field(ch5, c) for c in penb]
    v6 = max(np.nanmax(np.abs(F[2])) for F in f6)
    print(f"  hexagon shared colour scale +-{v6:.4f}; per-panel maxima "
          + "  ".join(f"{np.nanmax(np.abs(F[2])):.4f}" for F in f6), flush=True)
    p5 = [np.nanmax(np.abs(F[2])) for F in f5]
    print(f"  pentagon panels normalised individually, raw maxima " + "  ".join(f"{v:.4f}" for v in p5), flush=True)
    fig = plt.figure(figsize=(12.8, 7.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.10, hspace=0.16)
    lbl = (r"$\alpha_1=E_1-E_2$", r"$\alpha_2=E_2-E_3$", r"$\beta=H-E_1-E_2-E_3$")
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        c6 = draw(ax, *f6[i], "hexagon, periods of " + lbl[i], v6)
    fig.colorbar(c6, cax=fig.add_subplot(gs[0, 3]))
    for i, nm in enumerate((r"even, $+1$", r"odd, $-1$")):
        ax = fig.add_subplot(gs[1, i])
        GX, GY, Z, Pg = f5[i]
        c5 = draw(ax, GX, GY, Z / p5[i], Pg, "pentagon at $b^{\\ast}$, " + nm, 1.0)
    fig.colorbar(c5, cax=fig.add_subplot(gs[1, 3]))
    fig.subplots_adjust(left=0.015, right=0.945, top=0.955, bottom=0.02)
    for o in ([ROOT / "experiments" / "dp3kt" / "fig_period_basis.png"] +
              ([DRAFT / "fig_period_basis.png"] if DRAFT.is_dir() else [])):
        fig.savefig(o, dpi=160)
        print("wrote", o)
