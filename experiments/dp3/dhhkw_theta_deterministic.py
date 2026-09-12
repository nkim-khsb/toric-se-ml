"""Section 3.5's harmonic-form numbers on deterministic quadrature.

WHY.  dhhkw_theta_compare.py measures every constancy spread on a
20,000-point cloud drawn by sample_slice with a 1e-3 boundary margin.  The
margin drops a collar of the polygon, and that collar is where a polynomial
ansatz is worst, so the spreads come out too small -- the same failure that
r2b_stream_potential recorded for the harmonic forms (log.md 2026-08-19).
Section 3.5 already integrates the MEANS deterministically, so the paragraph
was carrying two estimators for the same quantities.  This script puts all of
them on polygon_quadrature.

WHAT IT COMPUTES.  Two domains, because the paragraph quotes both.

  [1] the whole hexagon: the spread of the contraction with DHHKW's published
      coefficients on our deg-18 metric and on the untrained Guillemin metric,
      their ratio (the discrimination), and the refit ladder at orders 6, 8, 10.

  [2] DHHKW's fitting domain |x_i| < 0.9, which is the hexagon clipped by
      |s_i| <= 0.3 (x = 3s).  Two of the square's corners fall outside the
      hexagon, so the region is a hexagon of its own with four edges from the
      square and two from the original facets; polygon_quadrature takes it as
      any other convex polygon.  On the whole polygon the mean is a boundary
      integral and comes out exactly 2/3; on a proper subdomain it need not,
      and [2] is what separates that domain effect from the estimator.

WEIGHTED LEAST SQUARES.  The refit minimizes the integral of (L[mu] - c)^2
rather than a sum over sample points, so the design rows carry sqrt(w).  The
spread reported is the rms about the weighted mean, in the contraction
normalization whose exact value is 2/3 -- the quantity of DHHKW (6.12), not
the Laplacian itself, which is three times larger.

CONVERGENCE.  Every number below is stable across the Gauss orders printed, to
all the figures shown.

Usage: PYTHONPATH=. python experiments/dp3/dhhkw_theta_deterministic.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dhhkw_theta_compare as T                       # noqa: E402
from sugrasol.laplacian import polygon_quadrature     # noqa: E402

HALF = 0.3            # their |x_i| < 0.9 in our s = x/3


def clip(poly, n, c):
    """Sutherland-Hodgman against the half-plane n . s + c >= 0."""
    out = []
    for i in range(len(poly)):
        A, B = poly[i], poly[(i + 1) % len(poly)]
        fa, fb = n @ A + c, n @ B + c
        if fa >= -1e-14:
            out.append(A)
        if (fa > 0) != (fb > 0):
            out.append(A + (B - A) * fa / (fa - fb))
    return np.array(out)


def fitting_domain(verts_s):
    poly = np.asarray(verts_s, dtype=float)
    for n, c in ((np.array([-1.0, 0.0]), HALF), (np.array([1.0, 0.0]), HALF),
                 (np.array([0.0, -1.0]), HALF), (np.array([0.0, 1.0]), HALF)):
        poly = clip(poly, n, c)
    d = poly - poly.mean(0)
    return poly[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]


def contraction(op, mu, nodes):
    """TO_THEM * L[mu]: the quantity whose exact value is 2/3."""
    return T.TO_THEM * np.asarray(jax.jit(jax.vmap(op(mu)))(nodes))


def spread(vals, W):
    m = float(W @ vals)
    return m, float(np.sqrt(W @ (vals - m) ** 2))


def wrefit(op, nodes, W, powers):
    A, b = T.build_design(op, nodes, powers, False)
    r = np.sqrt(W)[:, None]
    sol, *_ = np.linalg.lstsq(A * r, b * r[:, 0], rcond=None)
    resid = T.TO_THEM * (A @ sol - b)
    return T.TO_THEM * sol[-1], float(np.sqrt(W @ resid ** 2))


def main():
    psi, ch, dat = T.load_psi_dp3(T.NPZ["deg-18"])
    L = T.laplacian_op(ch, psi)
    L0 = T.laplacian_op(ch, lambda s: 0.0)
    print(f"deg-18 metric ({dat['W'].shape[1]} params); contraction "
          f"= {T.TO_THEM} * L[mu], exact constant 2/3")

    print("\n[1] whole hexagon")
    for ngl in (16, 28):
        nodes, w = polygon_quadrature(ch.verts_s, ngl)
        W = np.asarray(w) / float(np.sum(w))
        m_o, s_o = spread(contraction(L, T.mu_dhhkw, nodes), W)
        m_g, s_g = spread(contraction(L0, T.mu_dhhkw, nodes), W)
        print(f"  ngl {ngl:2d} ({len(W):5d} nodes)")
        print(f"    published coefficients, our metric : mean {m_o:.6f} "
              f"(off {abs(m_o - 2/3):.1e})   spread {s_o:.3e}")
        print(f"    published coefficients, Guillemin  : mean {m_g:.6f} "
              f"(off {abs(m_g - 2/3):.1e})   spread {s_g:.3e}")
        print(f"    discrimination = {s_g / s_o:.1f}")
        for order in (6, 8, 10):
            pw = T.sym_powers(order)
            const, sp = wrefit(L, nodes, W, pw)
            print(f"    refit order {order:2d} ({len(pw):3d} par)  const {const:.6f} "
                  f"(off {abs(const - 2/3):.1e})   spread {sp:.3e}")

    poly = fitting_domain(ch.verts_s)
    print(f"\n[2] their fitting domain |x_i| < 0.9: {len(poly)} vertices, in x = 3s")
    print(f"    {np.round(3 * poly, 4).tolist()}")
    for ngl in (16, 28, 40):
        nodes, w = polygon_quadrature(poly, ngl)
        W = np.asarray(w) / float(np.sum(w))
        m, s = spread(contraction(L, T.mu_dhhkw, nodes), W)
        print(f"  ngl {ngl:2d} ({len(W):5d} nodes)  published coefficients: "
              f"mean {m:.6f}   spread {s:.3e}")

    print("\nquoted in section 3.5: spreads 6.129e-03 and 3.568e-01, "
          "discrimination 58.2,\n  ladder 3.106e-03 / 6.763e-04 / 1.366e-04, "
          "fitting-domain mean 0.667037.")


if __name__ == "__main__":
    main()
