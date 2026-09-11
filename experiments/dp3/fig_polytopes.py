"""Setup figure for paper1 section 2: the toric diagram and the Reeb slice
polygons P for the dP3 hexagon and the dP2 pentagon.

Three panels:
  (a) the fan rays w_a in Z^2 (DHHKW eq. (3.49) for dP3; dP2 is the same
      hexagon with the (0,-1) ray deleted, see sugrasol/cone.py:107-121);
  (b) the dP3 slice polygon P = {t in C : <b,t> = 1} at the regular Reeb
      b = (3,0,0);
  (c) the dP2 slice polygon at the regular b = (3,0,0) and at the
      volume-minimizing irregular b = (3, 0, 3(19-3*sqrt(33))/16).

The chart is reimplemented here in numpy (identically to slice_chart: SVD of
b, t0 = b/|b|^2, s = f (t - t0)) so the figure has no jax dependency.

Panels (b) and (c) are drawn in the DISPLAY CHART x = 3s of the paper, not in
s itself.  The factor is what puts the vertices of P on the integer lattice at
a regular Reeb vector; at the irregular b* of dP2 no rescaling does that, and
the dashed pentagon of panel (c) is drawn in the same chart for comparison.
Both facts are asserted below, since the paper's text asserts them too.

The slice-polygon vertices are PRINTED, not drawn: three panels across
\\textwidth leave each about 2 inches wide, and a label like "(-1/3, 1/3)" set
large enough to read in print would cover a quarter of the panel.  The printed
values are the ones quoted in the caption, and they are asserted exact below.

Run:  python experiments/dp3/fig_polytopes.py
"""
from fractions import Fraction

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "experiments/dp3/fig_polytopes.png"
# The draft embeds its own copy; keep the two in step or the script stops
# being able to regenerate the figure that is actually in the paper.
COPIES = ("writing/paper1-new/fig_polytopes.png",)
SCALE = 3.0   # display chart x = 3s

W_DP3 = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]
W_DP2 = W_DP3[:5]
B_REG = np.array([3.0, 0.0, 0.0])
B3_STAR = 3.0 * (19.0 - 3.0 * np.sqrt(33.0)) / 16.0
B_DP2_IRR = np.array([3.0, 0.0, B3_STAR])


def normals(w):
    """Calabi-Yau gauge v_a = (1, w_a); moment cone C = {<v_a,y> >= 0}."""
    return np.array([(1.0, float(a), float(b)) for a, b in w])


def slice_polygon(w, b):
    """Vertices of P = {t in C : <b,t> = 1} in the 2d chart s.

    Mirrors sugrasol/ypq.py::slice_chart exactly.  A vertex of P is where the
    slice meets an edge of C, i.e. the ray on which two adjacent facets l_a,
    l_{a+1} both vanish; that ray is v_a x v_{a+1}.
    """
    v = normals(w)
    d = len(v)
    _, _, vt = np.linalg.svd(b[None, :])
    f = vt[1:]                      # (2,3) orthonormal basis of ker(b)
    t0 = b / (b @ b)                # base point, <b,t0> = 1
    rays = np.stack([np.cross(v[a], v[(a + 1) % d]) for a in range(d)])
    rays = rays * np.sign(rays @ b)[:, None]
    verts = rays / (rays @ b)[:, None]
    # self-check: exactly two facets vanish at each vertex, none is negative
    ell = verts @ v.T
    assert ell.min() > -1e-12, ell.min()
    assert np.all(np.sort(np.abs(ell), axis=1)[:, 1] < 1e-12)
    assert abs(verts @ b - 1).max() < 1e-12
    return (verts - t0) @ f.T


def close(p):
    return np.vstack([p, p[:1]])


def lattice(ax_, xmax, ymax):
    """Faint integer lattice under a slice panel, drawn in the display chart."""
    xs = np.arange(-np.floor(xmax), np.floor(xmax) + 1)
    ys = np.arange(-np.floor(ymax), np.floor(ymax) + 1)
    g = np.array([(u, v) for u in xs for v in ys])
    ax_.plot(g[:, 0], g[:, 1], ".", color="0.78", ms=4, zorder=1)


def as_fraction(x, tol=1e-12):
    """'a/b' when x is a small rational, else four decimals."""
    f = Fraction(x).limit_denominator(64)
    if abs(float(f) - x) > tol:
        return f"{x:+.4f}"
    return f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def main():
    s_dp3 = slice_polygon(W_DP3, B_REG)
    s_dp2 = slice_polygon(W_DP2, B_REG)
    s_dp2_irr = slice_polygon(W_DP2, B_DP2_IRR)

    # At a rational Reeb the vertices are exact rationals; freeze them.
    t3 = 1.0 / 3.0
    assert np.allclose(np.sort(s_dp3, axis=0), np.sort(np.array(
        [(-t3, 0.0), (0.0, -t3), (t3, -t3), (t3, 0.0), (0.0, t3), (-t3, t3)]),
        axis=0))
    assert np.allclose(np.sort(s_dp2, axis=0), np.sort(np.array(
        [(-t3, 0.0), (0.0, -t3), (t3, -t3), (t3, 0.0), (-t3, 2 * t3)]), axis=0))

    # The display chart of the paper.  x = 3s is integral at a regular Reeb
    # and is not at the irregular one -- the claim the caption makes.
    x_dp3, x_dp2, x_dp2_irr = (SCALE * z for z in (s_dp3, s_dp2, s_dp2_irr))
    for nm, x in (("dP3", x_dp3), ("dP2", x_dp2)):
        assert np.abs(x - np.rint(x)).max() < 1e-12, (nm, x)
    # irrational b3* => no vertex lattice, at this or any other scale
    assert np.abs(x_dp2_irr - np.rint(x_dp2_irr)).max() > 0.05, x_dp2_irr

    for name, s, x in [("dP3, b=(3,0,0)", s_dp3, x_dp3),
                       ("dP2, b=(3,0,0)", s_dp2, x_dp2),
                       (f"dP2, b=(3,0,{B3_STAR:.10f})", s_dp2_irr, x_dp2_irr)]:
        pts = ", ".join(f"({as_fraction(u)},{as_fraction(v)})" for u, v in s)
        xs = ", ".join(f"({as_fraction(u)},{as_fraction(v)})" for u, v in x)
        print(f"\n{name}\n   s   {pts}\n   x=3s{xs}")

    plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))

    # (a) fan rays -------------------------------------------------------
    a = ax[0]
    hexa = np.array(W_DP3, dtype=float)
    penta = np.array(W_DP2, dtype=float)
    a.plot(*close(hexa).T, "-", color="0.25", lw=1.6, label=r"$\mathrm{dP}_3$")
    a.plot(*close(penta).T, "--", color="C3", lw=1.6, label=r"$\mathrm{dP}_2$")
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            a.plot(x, y, ".", color="0.78", ms=5, zorder=1)
    a.plot(hexa[:5, 0], hexa[:5, 1], "o", color="0.15", ms=6.5, zorder=3)
    a.plot([0], [-1], "o", mfc="white", mec="C3", mew=1.8, ms=7.5, zorder=3)
    # The Reeb lives in the FAN cone (b in int C^vee = int sigma), so b/b_1 is
    # an interior point of this diagram: the unique interior lattice point when
    # the Reeb is regular, a non-lattice point when it is irregular.
    a.plot([0], [0], "s", color="0.15", ms=6, zorder=4,
           label=r"$b/b_1$ regular")
    a.plot([0], [B3_STAR / 3.0], "s", color="C1", mfc="none", mew=1.7, ms=7,
           zorder=4, label=r"$b/b_1$ irregular ($\mathrm{dP}_2$)")
    offs = {(1, 0): (8, 3), (1, 1): (8, 3), (0, 1): (-4, 8), (-1, 0): (-9, 3),
            (-1, -1): (-10, -12), (0, -1): (9, -5)}
    for w in W_DP3:
        a.annotate(f"$({w[0]},{w[1]})$", w, textcoords="offset points",
                   xytext=offs[w], fontsize=9.5,
                   color="C3" if w == (0, -1) else "black")
    a.set_title(r"(a) fan rays $w_a$,  $v_a=(1,w_a)$", fontsize=11)
    a.legend(loc="upper left", fontsize=8, frameon=False,
             handlelength=1.4, borderpad=0.1, labelspacing=0.35,
             handletextpad=0.5)
    a.set_xlim(-2.0, 1.9)
    a.set_ylim(-1.75, 1.65)
    a.set_xlabel("$w_1$", fontsize=10)
    a.set_ylabel("$w_2$", fontsize=10)

    # (b) dP3 slice polygon ----------------------------------------------
    b_ = ax[1]
    lattice(b_, 1.4, 1.4)
    b_.fill(*close(x_dp3).T, color="C0", alpha=0.13)
    b_.plot(*close(x_dp3).T, "-o", color="C0", lw=1.6, ms=4.5)
    # vertex k is the ray v_k x v_{k+1}, so the edge from vertex k to k+1 lies
    # on facet l_{k+1} (0-based) = l_{(k+1) mod d + 1} labelled from one.
    mids = 0.5 * (x_dp3 + np.roll(x_dp3, -1, axis=0))
    for k, (x, y) in enumerate(mids):
        b_.annotate(rf"$\ell_{{{(k + 1) % len(x_dp3) + 1}}}$", (x, y),
                    fontsize=9.5, color="0.3", ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5))
    b_.plot([0], [0], "+", color="0.45", ms=8)
    b_.set_title(r"(b) $\mathrm{dP}_3$ slice $P$,  $b=(3,0,0)$", fontsize=11)
    b_.set_xlim(-1.38, 1.38)
    b_.set_ylim(-1.38, 1.38)

    # (c) dP2 slice polygon, two Reeb vectors ----------------------------
    c = ax[2]
    lattice(c, 1.6, 2.6)
    c.fill(*close(x_dp2).T, color="C2", alpha=0.13)
    c.plot(*close(x_dp2).T, "-o", color="C2", lw=1.6, ms=4.5,
           label=r"regular $b=(3,0,0)$")
    c.plot(*close(x_dp2_irr).T, "--s", color="C1", lw=1.6, ms=4, mfc="none",
           label=r"irregular $b=(3,0,b_3^{\ast})$")
    c.plot([0], [0], "+", color="0.45", ms=8)
    c.set_title(r"(c) $\mathrm{dP}_2$ slice $P$, two Reebs", fontsize=11)
    c.legend(loc="lower left", fontsize=8.5, frameon=False,
             handlelength=1.6, borderpad=0.1)
    c.set_xlim(-1.56, 1.56)
    c.set_ylim(-2.58, 2.58)

    for p in (ax[1], ax[2]):
        p.set_xlabel("$x_1$", fontsize=10)
        p.set_ylabel("$x_2$", fontsize=10)
    for p in ax:
        p.set_aspect("equal")
        p.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            p.spines[side].set_visible(False)

    fig.tight_layout(pad=0.6)
    for path in (OUT, *COPIES):
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
