"""dP3: apples-to-apples residual comparison with DHHKW, in THEIR error measure.

Why.  The paper used to claim our residual sits orders of magnitude below
DHHKW's.  The two residuals are different functionals: ours is the mean square
of the Monge-Ampere residual, theirs is the pointwise deviation of the metric
from the Einstein condition,

    D(x) = sqrt( (1/4) (R_mn - g_mn)(R^mn - g^mn) )        [DHHKW eq. (5.4)]

(hep-th/0703057; their flow (4.1) fixes the normalization Ric = g, so their
scalar curvature is S = 4).  Comparing orders of magnitude between different
functionals is exactly what a referee kills, so here we compute THEIR D on OUR
learned metrics and compare like with like.

Normalization (one place, rule 6).  Our doubled-slice convention has Ric = 3g
(S = 12 in 4d; laplacian.py docstring, and the conifold anchor below).  Since
Ric(c g) = Ric(g) for constant c, the metric g' = 3g satisfies Ric(g') = g'
exactly, and DHHKW's D may be applied to g' verbatim -- no correction factor.

Coordinates (reused, not re-derived).  DHHKW's D6-invariant U = x1^2+x1x2+x2^2
runs 0 at the centre to 1 at the vertices; direct_compare.py already matches our
slice coordinate s to it by the Reynolds-averaged quadratic form.  Their Fig. 12
profile D(x1, 0) is the centre-to-vertex ray (on x2 = 0, U = x1^2, and (1,0) is
a hexagon corner -- dhhkw.txt around eq. (6.4)).  Our matched ray is therefore
s(x1) = x1 * (vertex), for which U = x1^2 identically.

Their numbers (dhhkw.txt lines 1281-1300, Fig. 12): the 18th-order expansion has
max D on the origin-vertex lines, finite at the vertex, and about 1 part in 10^3
there.  The potential itself is good to ~1 part in 10^6; the error is localized
in the corners.

PRE-REGISTERED verdict (fixed before running, log.md 2026-07-21 lesson).  Let
Dmax(18) be our deg-18 max D over the region where the numerical noise floor
(same code on the conifold, where D = 0 exactly) is at least 10x below it:
  (A) Dmax(18) <= 1e-5  -> restore a quantitative claim: "two orders of
      magnitude below DHHKW's D at the vertex", definitions now matched.
  (B) 1e-5 < Dmax(18) <= 1e-4 -> claim only "an order of magnitude below".
  (C) Dmax(18) > 1e-4, i.e. same 1e-3-ish ballpark -> WITHDRAW the residual
      comparison from the paper and reframe as "different error functionals,
      both converged"; this is information, not failure (rule 5).
  (D) If the conifold noise floor is not 10x below our D at a given margin,
      that margin is undecidable and is reported as such, not as a value.

Usage: PYTHONPATH=. python experiments/dp3/dhhkw_D_compare.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3       # noqa: E402
from sugrasol.curvature import ricci, toric_metric               # noqa: E402
from sugrasol.artifacts import load_psi_dp3                      # noqa: E402
from sugrasol.laplacian import polygon_quadrature, slice_potential  # noqa: E402
from sugrasol.ypq import dihedral_matrices, slice_chart          # noqa: E402

BLUE, RED, GREY = "#4c72b0", "#c44e52", "#888888"
CHUNK = 2000        # fixed batch length: one XLA trace per metric, then stream
CLOUD = 400000      # uniform cloud for the max-D scan over the hexagon
NPZ14 = ROOT / "experiments" / "dp3smooth" / "dp3_G_deg14.npz"
NPZ18 = ROOT / "experiments" / "dp3smooth" / "dp3_G_deg18.npz"


# --------------------------------------------------------------- D functional
def D_field(u):
    """s -> DHHKW's D for the toric Kaehler metric with slice potential u.

    g = blockdiag(Hess u, (Hess u)^{-1}) in (s1,s2,phi1,phi2) is our Ric = 3g
    normalization; g' = 3g is DHHKW's Ric = g one (Ric is scale invariant)."""
    g_fn = toric_metric(u, 2)
    gp_fn = lambda x: 3.0 * g_fn(x)          # noqa: E731  (DHHKW normalization)

    def D(s):
        x = jnp.concatenate([s, jnp.zeros(2)])   # angles: metric is invariant
        gp = gp_fn(x)
        T = ricci(gp_fn, x) - gp
        gpi = jnp.linalg.inv(gp)
        return jnp.sqrt(0.25 * jnp.einsum("ab,ac,bd,cd->", T, gpi, gpi, T))

    # One compiled shape only: the deg-18 curvature graph costs ~60 s to trace,
    # and every new batch length re-triggers it.  Pad to CHUNK and stream.
    batched = jax.jit(jax.vmap(D))

    def run(pts):
        pts = jnp.asarray(pts)
        n = pts.shape[0]
        pad = (-n) % CHUNK
        pp = jnp.concatenate([pts, jnp.zeros((pad, 2))]) if pad else pts
        out = [np.asarray(batched(pp[i:i + CHUNK]))
               for i in range(0, pp.shape[0], CHUNK)]
        return np.concatenate(out)[:n]

    return run


def scalar_curvature(u, s):
    g_fn = toric_metric(u, 2)
    x = jnp.concatenate([s, jnp.zeros(2)])
    return float(jnp.einsum("ab,ab->", jnp.linalg.inv(g_fn(x)), ricci(g_fn, x)))


def rho_of_s(chart):
    """Radial fraction about the chart origin s = 0 (an interior point, and the
    centroid for dP3): rho = 0 there, rho = 1 on the polygon boundary.

    Facet a is l_a(t(s)) = l_a(t0) + (f v_a).s >= 0, hence
    rho(s) = max_a (-(f v_a).s) / l_a(t0); on a ray s = a * vertex, rho = a."""
    v = chart.cone.normals
    n = v @ chart.f.T                              # (d, 2)
    c = chart.cone.ells(chart.t0)                  # (d,)
    return jax.jit(jax.vmap(lambda s: jnp.max(-(n @ s) / c)))


# ------------------------------------------------------- anchor 1: conifold
print("== anchor: conifold base CP^1 x CP^1 (psi = 0), where D = 0 exactly ==")
ch_c = slice_chart(conifold(), B_CONIFOLD)
u_c = slice_potential(ch_c, lambda s: 0.0)
D_c, rho_c = D_field(u_c), rho_of_s(ch_c)
print(f"   S = {scalar_curvature(u_c, jnp.zeros(2)):.10f}   "
      f"(our doubled slice: 12)")

# noise floor along the origin->corner ray, at the same margins 1 - rho as the
# dP3 profile below (rho(a * vertex) = a, so the two are directly comparable)
ray_c = lambda a: a * ch_c.verts_s[0]                     # noqa: E731
margins = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
alphas = 1.0 - margins
noise = np.array(D_c(jnp.stack([ray_c(a) for a in alphas])))
for m, d in zip(margins, noise):
    print(f"   1-rho = {m:8.1e}   D_conifold = {d:.3e}   (exact value 0)")

# ------------------------------------------------------------------ dP3
ch = slice_chart(dp3(), B_DP3)
rho = rho_of_s(ch)
verts = np.array(ch.verts_s)
G = np.array(dihedral_matrices(ch.verts_s))
M = sum(g.T @ g for g in G)                       # Reynolds average (D6 form)
uvert = float(verts[0] @ M @ verts[0])
U_of_s = lambda s: (s @ M @ s) / uvert            # noqa: E731  (0 centre, 1 vertex)

metrics = {}
for name, npz in (("deg-14", NPZ14), ("deg-18", NPZ18)):
    psi, _, dat = load_psi_dp3(npz)
    u = slice_potential(ch, psi)
    metrics[name] = (u, D_field(u), int(dat["W"].shape[1]), float(dat["loss_test"]))
    print(f"\n== dP3 {name}: {metrics[name][2]} params, held-out MA loss "
          f"{metrics[name][3]:.2e}, S(centre) = "
          f"{scalar_curvature(u, jnp.zeros(2)):.6f} ==")

# ---- (c) profile D(x1, 0) along the centre->vertex ray (DHHKW Fig. 12) ------
x1_lin = np.linspace(0.0, 0.9, 19)
x1_log = 1.0 - np.logspace(-1, -5, 17)
x1 = np.unique(np.concatenate([x1_lin, x1_log]))
vtx = jnp.asarray(verts[3])                       # (1/3, 0); all 6 equivalent
ray = jnp.stack([a * vtx for a in x1])
prof = {k: np.array(v[1](ray)) for k, v in metrics.items()}
noise_ray = np.array(D_c(jnp.stack([ray_c(a) for a in x1])))
print("\n== (c) D along x2 = 0 (centre x1=0 -> vertex x1=1), DHHKW Fig. 12 ==")
print(f"{'x1':>9} {'1-x1':>9} {'U':>7} {'D deg-14':>11} {'D deg-18':>11} "
      f"{'noise':>10}")
for i, a in enumerate(x1):
    uu = float(U_of_s(np.array(a * vtx)))
    print(f"{a:9.5f} {1 - a:9.1e} {uu:7.4f} {prof['deg-14'][i]:11.3e} "
          f"{prof['deg-18'][i]:11.3e} {noise_ray[i]:10.2e}")

# check D is D6-invariant (all six rays agree) -- guards the coordinate matching
for k, (_, Dfn, _, _) in metrics.items():
    a = 0.99
    six = np.array(Dfn(jnp.stack([a * jnp.asarray(v) for v in verts])))
    print(f"   [{k}] D at rho={a} on the 6 vertex rays: spread "
          f"{six.max() / six.min() - 1:.2e} (D6 invariance)")

# ---- (a) max D over the hexagon, as a function of the boundary margin -------
key = jax.random.PRNGKey(0)
lo, hi = verts.min(0), verts.max(0)
box = lo + jax.random.uniform(key, (CLOUD, 2), dtype=jnp.float64) * (hi - lo)
rr_box = np.array(rho(box))
cloud = np.array(box)[rr_box <= 1.0 - 1e-5]      # inside, one margin for all
rr = rr_box[rr_box <= 1.0 - 1e-5]
Us = np.array([float(U_of_s(p)) for p in cloud])
Dcloud = {k: v[1](cloud) for k, v in metrics.items()}
# conditioning control: the SAME code on the conifold square, where D = 0
# exactly, sampled over its own polygon and bucketed by the same rho ladder
lo_c, hi_c = np.array(ch_c.verts_s).min(0), np.array(ch_c.verts_s).max(0)
box_c = lo_c + np.asarray(jax.random.uniform(
    jax.random.PRNGKey(1), (CLOUD // 4, 2), dtype=jnp.float64)) * (hi_c - lo_c)
rr_c = np.array(rho_c(box_c))
cloud_c, rr_c = box_c[rr_c <= 1.0 - 1e-5], rr_c[rr_c <= 1.0 - 1e-5]
noise_cloud = D_c(cloud_c)
print(f"\n== (a) max D over the hexagon truncated at rho <= rho_max "
      f"({len(cloud)} interior points) ==")
print(f"{'rho_max':>9} {'n pts':>8}   " +
      "   ".join(f"{k:>26}" for k in metrics) + "     noise")
for rmax in (0.9, 0.95, 0.99, 0.999, 0.9999):
    sel = rr <= rmax
    row = f"{rmax:9.4f} {int(sel.sum()):8d} "
    for k in metrics:
        d = Dcloud[k][sel]
        j = int(np.argmax(d))
        row += (f"  {d[j]:.3e} @U={Us[sel][j]:.3f},rho={rr[sel][j]:.4f}")
    print(row + f"   {noise_cloud[rr_c <= rmax].max():.1e}")

# ---- (b) global rms: DHHKW's beta, beta^2 = (1/V) int sqrt(g) D^2 ----------
# In symplectic coordinates det g = 1 (g = blockdiag(H, H^{-1})), so sqrt(g') is
# the constant 9 and beta^2 is simply the polygon average of D^2 -- computed
# with the deterministic polygon quadrature (laplacian.polygon_quadrature).
print("\n== (b) global rms beta (DHHKW eq. (5.5)); their crude fits: "
      "0.5 / 0.1 / 0.03 / 0.007 ==")
for ngl in (16, 24, 32):
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=ngl)
    w = np.array(wts)
    row = f"   ngl={ngl:3d} "
    for k, (_, Dfn, _, _) in metrics.items():
        d = Dfn(nodes)
        row += f"  beta[{k}] = {np.sqrt(np.sum(w * d ** 2) / w.sum()):.4e}"
    nc, wc = polygon_quadrature(ch_c.verts_s, ngl=ngl)
    dn, wc = D_c(nc), np.array(wc)
    row += f"   (noise floor {np.sqrt(np.sum(wc * dn ** 2) / wc.sum()):.1e})"
    print(row)
    if ngl == 32:
        beta = {k: float(np.sqrt(np.sum(w * Dfn(nodes) ** 2) / w.sum()))
                for k, (_, Dfn, _, _) in metrics.items()}

# ------------------------------------------- pre-registered verdict, applied
# Decidable region: the largest rho where the conifold control (D = 0 exactly)
# is at least 10x below our D.  The ray profile is decidable to 1-rho = 1e-5
# (floor ~1e-10), so we read Dmax off its plateau at the vertex.
Dmax = {k: float(prof[k][-1]) for k in metrics}
print("\n== verdict (criteria fixed in the docstring before running) ==")
for k in metrics:
    print(f"   {k}: D at the vertex = {Dmax[k]:.3e}   global rms beta = "
          f"{beta[k]:.3e}   (DHHKW 18th order: D_vertex ~ 1e-3)")
d18 = Dmax["deg-18"]
verdict = ("(A) two orders below DHHKW" if d18 <= 1e-5 else
           "(B) one order below DHHKW" if d18 <= 1e-4 else
           "(C) same ballpark -- withdraw the residual comparison")
print(f"   deg-18 Dmax = {d18:.3e}  ->  {verdict}")
print(f"   [boundary check] Dmax/1e-4 = {d18 / 1e-4:.3f}, "
      f"Dmax/1e-5 = {d18 / 1e-5:.1f}, ratio to DHHKW's 1e-3 = {1e-3 / d18:.1f}x")
if min(abs(np.log10(d18) + 4), abs(np.log10(d18) + 5)) < 0.1:
    print("   WARNING: the value sits ON a pre-registered threshold (within a "
          "factor 1.26). Report the raw numbers, not the verdict label.")

np.savez(ROOT / "experiments" / "dp3" / "dhhkw_D_profile.npz",
         x1=x1, D_deg14=prof["deg-14"], D_deg18=prof["deg-18"],
         noise=noise_ray, beta_deg14=beta["deg-14"], beta_deg18=beta["deg-18"],
         note="DHHKW D = sqrt(1/4 (Ric-g)^2) at their Ric=g normalization "
              "(g' = 3g), along the centre->vertex ray x1 = sqrt(U); "
              "noise = same code on the conifold where D = 0 exactly.")

# ------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(6.2, 4.4))
m = x1 > 0
ax.semilogy(np.log(1 - x1[m]), prof["deg-14"][m], "o-", ms=3, color=BLUE,
            label=f"ours, deg-14 ({metrics['deg-14'][2]} params)")
ax.semilogy(np.log(1 - x1[m]), prof["deg-18"][m], "s-", ms=3, color=RED,
            label=f"ours, deg-18 ({metrics['deg-18'][2]} params)")
ax.semilogy(np.log(1 - x1[m]), noise_ray[m], ":", color=GREY,
            label="numerical floor (conifold, D=0 exactly)")
ax.axhline(1e-3, color="k", ls="--", lw=1)
ax.text(0.02, 0.955, "DHHKW 18th order, at the vertex", fontsize=8,
        transform=ax.transAxes, va="top")
ax.set_ylim(1e-16, 2e-2)
ax.set_xlabel(r"$\ln(1-x_1)$   (centre $\to$ hexagon vertex)")
ax.set_ylabel(r"$D=\sqrt{\frac{1}{4}(R_{\mu\nu}-g_{\mu\nu})^2}$")
ax.set_title("dP3: DHHKW's Einstein-condition error on our learned metrics")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
out = ROOT / "experiments" / "dp3" / "fig_dp3_D_profile.png"
fig.savefig(out, dpi=160)
print(f"\nwrote {out.name}")
