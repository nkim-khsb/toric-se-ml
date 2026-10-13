"""dP3: pointwise comparison of our deg-14/deg-18 potentials with the CERTIFIED
Kaehler-Einstein potential of Buttsworth-Hadden-Heyes-Platt-Wiseman (BHHPW),
arXiv:2609.32655, whose companion repository publishes the rationalised
Chebyshev coefficients (data/happrox/candidates/coeffs-rational-nonsymmetric.csv,
90x90, ~500-digit rationals, converted to float64 in experiments/dp3/bhhpw_coeffs_float64.npy;
the matrix decays from 1e-1 to 1e-49 so float64 loses nothing at our precision).

Why this matters.  Until now every dP3 number in paper I was graded by
held-out residual and psi-sensitive invariants (Abreu S, DHHKW's D, lambda_1):
there was no external truth below DHHKW's ~1e-3.  BHHPW prove the true KE
potential lies within ||phi||_{H^5} <= 1.7e-18 of theirs, so at our precision
their potential IS the truth, and the difference measured here is our
absolute error -- the first such measurement.

Conventions (rule 6, transcribed and gated below rather than assumed):
  BHHPW: unit hexagon P = {|x1|,|x2|,|x1+x2| <= 1}, Ric = g,
         u = u_P + h,  u_P = (1/2) sum_r l_r log l_r  (their eq. 2.3),
         h(x1,x2) = sum_{m,n<90} a_{mn} T_m(2 x1 - 1) T_n(1 + 2 x2)
         on the sector x1 in [0,1], x2 in [-1,0]  (Chebyshev/README.md:
         u(x,y) = sum a_{i+1,j+1} T_i(2x-1) T_j(1-2y), y = -x2), extended to
         the whole hexagon by D6.  Coefficients satisfy a_{mn} = (-1)^{m+n}
         a_{nm} exactly (checked on load).
  Ours:  doubled slice, l_a = 1/3 + v_a . s, Ric = 3 g  (dhhkw_D_compare.py).
         Hence x = phi(s) is affine with |scale| = 3 and
         u_them(phi(s)) = 3 u_ours(s) + affine  (dhhkw_theta_compare.py).
         phi is NOT assumed: it is fitted from the twelve vertex
         correspondences (both orientations) and gated.

PRE-REGISTERED gates (fixed before running):
  (G0) vertex correspondence: affine fit residual < 1e-12.
  (G1) convention check: their h, pulled back as psi_th = [u_them o phi]/3 -
       Guillemin_ours, plugged into OUR residual_on_slice must be constant:
       std over interior points < 1e-8  (a wrong x/y, sign or scale
       convention gives O(1)).  This is the gate that makes the rest meaningful.
  (G2) the pulled-back Guillemin part minus 3x ours is affine: Hessian of the
       difference < 1e-10 on the polygon.
  (G3) D6 folding is consistent: |psi_th(g s) - psi_th(s)| < 1e-10 for all
       twelve g in our dihedral group, on random interior points.
Measurements (no thresholds -- these are the numbers we did not have):
  (M1) relative metric error  e(s) = || L^{-1} (H_ours - H_th) L^{-T} ||_2,
       H = Hess_s u, L L^T = H_ours, on polygon_quadrature nodes (flat
       measure, whole hexagon) -- rms and max, with the location of the max.
  (M2) potential difference mod affine (3 gauge parameters fitted by least
       squares): rms and max, relative to the spread of psi.
  (M3) the deg-14 -> deg-18 ladder of (M1)/(M2): must decrease.

Usage: PYTHONPATH=. python experiments/dp3/bhhpw_pointwise_compare.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from sugrasol.artifacts import load_psi_dp3                              # noqa: E402
from sugrasol.cone import B_DP3, dp3                                     # noqa: E402
from sugrasol.laplacian import polygon_quadrature, slice_potential       # noqa: E402
from sugrasol.ypq import (dihedral_matrices, residual_on_slice,          # noqa: E402
                          sample_slice, slice_chart)

ROOT = Path(__file__).resolve().parents[2]
NPZ = {"deg-14": ROOT / "experiments/dp3smooth/dp3_G_deg14.npz",
       "deg-18": ROOT / "experiments/dp3smooth/dp3_G_deg18.npz"}
COEF = ROOT / "experiments/dp3/bhhpw_coeffs_float64.npy"

# ------------------------------------------------------------ their potential
A = np.load(COEF)                                   # (90, 90), a[m, n]
NDEG = A.shape[0]
assert A.shape == (NDEG, NDEG)
sym_dev = max(abs(A[m, n] - (-1) ** (m + n) * A[n, m])
              for m in range(NDEG) for n in range(NDEG))
A_J = jnp.asarray(A)

# their facet functions and D6 (their R1, R2 of sec. 3.1)
NU = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]], dtype=float)
R1 = np.array([[0., 1.], [1., 0.]])
R2 = np.array([[1., 1.], [-1., 0.]])
THEIR_GROUP = []
for k in range(6):
    Rk = np.linalg.matrix_power(R2, k)
    for M in (Rk, R1 @ Rk):
        if not any(np.allclose(M, G) for G in THEIR_GROUP):
            THEIR_GROUP.append(M)
assert len(THEIR_GROUP) == 12
THEIR_GROUP_J = jnp.asarray(np.stack(THEIR_GROUP))


def cheb_all(z, n=NDEG):
    """T_0..T_{n-1}(z) by the three-term recurrence (autodiff-safe at z = +-1)."""
    T = [jnp.ones_like(z), z]
    for _ in range(2, n):
        T.append(2.0 * z * T[-1] - T[-2])
    return jnp.stack(T[:n])


def h_sector(x):
    """h on the sector x1 in [0,1], x2 in [-1,0]."""
    Tx = cheb_all(2.0 * x[0] - 1.0)
    Ty = cheb_all(1.0 + 2.0 * x[1])
    return Tx @ A_J @ Ty


def h_hex(x):
    """h anywhere on the hexagon, folded into the sector by their D6."""
    imgs = THEIR_GROUP_J @ x                                  # (12, 2)
    tol = 1e-12
    ok = ((imgs[:, 0] >= -tol) & (imgs[:, 0] <= 1 + tol)
          & (imgs[:, 1] >= -1 - tol) & (imgs[:, 1] <= tol))
    k = jnp.argmax(ok)
    # No clip: the Chebyshev series is a polynomial, so a point a rounding
    # error outside the sector is harmless, whereas clip's tie rule at an
    # exact boundary value (the centre, the centre-vertex rays) halves the
    # derivative and corrupts the Hessian there.
    return h_sector(imgs[k])


def guillemin_them(x):
    l = 1.0 + NU @ x
    return 0.5 * jnp.sum(l * jnp.log(l))


def u_them(x):
    return guillemin_them(x) + h_hex(x)


# ------------------------------------------------------------ the bridge phi
chart = slice_chart(dp3(), B_DP3)
VERTS_S = np.asarray(chart.verts_s)                         # (6, 2) cyclic
VERTS_X = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]], float)


def fit_affine(src, dst):
    X = np.concatenate([src, np.ones((len(src), 1))], axis=1)   # (6, 3)
    coef, *_ = np.linalg.lstsq(X, dst, rcond=None)              # (3, 2)
    res = np.abs(X @ coef - dst).max()
    return coef[:2].T, coef[2], res                             # M (2,2), m (2,)


best = None
for orient in (1, -1):
    for shift in range(6):
        dst = VERTS_X[::orient]
        dst = np.roll(dst, shift, axis=0)
        M, m, res = fit_affine(VERTS_S, dst)
        if best is None or res < best[2]:
            best = (M, m, res, orient, shift)
PHI_M, PHI_m, G0_RES, _, _ = best
PHI_M_J, PHI_m_J = jnp.asarray(PHI_M), jnp.asarray(PHI_m)


def phi(s):
    return PHI_M_J @ s + PHI_m_J


# ---------------------------------------------------------- pulled-back psi
guil_ours = slice_potential(chart, lambda s: 0.0)


def psi_th(s):
    """Their full potential pulled back to our slice, minus OUR Guillemin:
    the object that plays the role of our psi."""
    return u_them(phi(s)) / 3.0 - guil_ours(s)


def guil_diff(s):
    """(G2): their Guillemin pulled back minus 3x ours -- must be affine."""
    return guillemin_them(phi(s)) - 3.0 * guil_ours(s)


# ------------------------------------------------------------------ main
def rel_metric_err(H_o, H_t):
    L = jnp.linalg.cholesky(H_o)
    Li = jnp.linalg.inv(L)
    E = Li @ (H_o - H_t) @ Li.T
    return jnp.linalg.norm(E, ord=2)


def main():
    print("== BHHPW arXiv:2609.32655 certified potential vs our dP3 artifacts ==")
    print(f"  their coefficients: {NDEG}x{NDEG}, max|a| = {abs(A).max():.3e}, "
          f"exact symmetry a_mn = (-1)^(m+n) a_nm: max dev {sym_dev:.1e}")
    print(f"  (G0) vertex affine fit: residual {G0_RES:.2e}  "
          f"{'PASS' if G0_RES < 1e-12 else 'FAIL'}")
    print(f"       phi(s) = M s + m,  M =\n{np.round(PHI_M, 12)}\n       m = {np.round(PHI_m, 12)}"
          f"\n       |det M| = {abs(np.linalg.det(PHI_M)):.12f}  (expect 9 = 3^2)")

    # (G1) convention check through OUR residual
    ss = sample_slice(jax.random.PRNGKey(3), chart, 1500, eps=2e-3)
    r_th = jax.jit(jax.vmap(lambda s: residual_on_slice(chart, psi_th, s)))(ss)
    print(f"  (G1) their h in our residual_on_slice: mean {float(jnp.mean(r_th)):+.12f}, "
          f"std {float(jnp.std(r_th)):.3e}, max|r-mean| {float(jnp.max(jnp.abs(r_th-jnp.mean(r_th)))):.3e}  "
          f"{'PASS' if float(jnp.std(r_th)) < 1e-8 else 'FAIL'}")

    # their psi in OUR gauge (centre value and gradient removed), for (M0)
    s0 = jnp.zeros(2)
    th0, dth0 = float(psi_th(s0)), jax.grad(psi_th)(s0)

    def psi_th_gf(s):
        return psi_th(s) - th0 - dth0 @ s

    r_gf = jax.jit(jax.vmap(lambda s: residual_on_slice(chart, psi_th_gf, s)))(ss)

    # (G2) Guillemin bridge is affine
    Hg = jax.jit(jax.vmap(jax.hessian(guil_diff)))(ss)
    g2 = float(jnp.max(jnp.abs(Hg)))
    print(f"  (G2) Hess[their Guillemin o phi - 3 ours] max {g2:.2e}  {'PASS' if g2 < 1e-10 else 'FAIL'}")

    # (G3) D6 folding consistency in OUR group
    G = dihedral_matrices(chart.verts_s)
    f_th = jax.jit(jax.vmap(psi_th))
    base = f_th(ss)
    g3 = max(float(jnp.max(jnp.abs(f_th(ss @ np.asarray(g).T) - base))) for g in G)
    print(f"  (G3) D6 invariance of pulled-back psi: max dev {g3:.2e}  {'PASS' if g3 < 1e-10 else 'FAIL'}")
    assert G0_RES < 1e-12 and float(jnp.std(r_th)) < 1e-8 and g2 < 1e-10 and g3 < 1e-10, \
        "a gate failed -- the comparison below would be meaningless"

    # measurements
    nodes, weights = polygon_quadrature(chart.verts_s, ngl=24)
    area = float(jnp.sum(weights))
    H_th = jax.jit(jax.vmap(jax.hessian(lambda s: u_them(phi(s)) / 3.0)))(nodes)
    psi_th_nodes = f_th(nodes)
    ones = np.concatenate([np.ones((len(nodes), 1)), np.asarray(nodes)], axis=1)
    print(f"\n  nodes: {len(nodes)} (polygon_quadrature ngl=24, flat measure)")
    print(f"  {'artifact':8s} {'params':>6s} {'held-out':>9s} | {'rel. metric err rms':>20s} {'max':>10s} "
          f"{'at r/r_vert':>11s} | {'psi diff mod affine rms':>23s} {'max':>10s} {'spread psi':>10s}")
    for tag, path in NPZ.items():
        psi, _, meta = load_psi_dp3(path)
        u_o = slice_potential(chart, psi)
        H_o = jax.jit(jax.vmap(jax.hessian(u_o)))(nodes)
        e = jax.vmap(rel_metric_err)(H_o, H_th)
        e_rms = float(jnp.sqrt(jnp.dot(weights, e ** 2) / area))
        k = int(jnp.argmax(e))
        r_rel = float(jnp.linalg.norm(nodes[k]) / jnp.linalg.norm(chart.verts_s[0]))
        d = np.asarray(jax.jit(jax.vmap(psi))(nodes)) - np.asarray(psi_th_nodes)
        coef, *_ = np.linalg.lstsq(ones * np.sqrt(np.asarray(weights))[:, None],
                                   d * np.sqrt(np.asarray(weights)), rcond=None)
        dres = d - ones @ coef
        d_rms = float(np.sqrt(np.dot(np.asarray(weights), dres ** 2) / area))
        spread = float(np.ptp(np.asarray(jax.jit(jax.vmap(psi))(nodes))))
        nparam = int(meta["coeffs"].shape[0])
        hkey = [k_ for k_ in meta.files if "held" in k_.lower() or "test" in k_.lower()]
        held = float(meta[hkey[0]]) if hkey else float("nan")
        print(f"  {tag:8s} {nparam:6d} {held:9.1e} | {e_rms:20.3e} {float(e[k]):10.3e} {r_rel:11.3f} | "
              f"{d_rms:23.3e} {float(np.abs(dres).max()):10.3e} {spread:10.3e}")
        # (M0) free consistency check: the MA constant.  The constant is gauge
        # dependent (an affine shift of psi moves 2 dG/dy_1 by a constant), so
        # their psi is first put in OUR gauge -- the invariant basis starts at
        # degree 2, i.e. psi(0) = 0, grad psi(0) = 0 at the hexagon centre --
        # after which our fitted c must agree with the constant their potential
        # produces in our residual, to our own accuracy.
        if "c" in meta.files:
            c_th = float(jnp.mean(r_gf))
            print(f"           (M0) c_ours = {float(meta['c']):+.12f}  vs  c_theirs(our gauge) = "
                  f"{-c_th:+.12f}   diff {float(meta['c']) + c_th:+.2e}")

    # also at the DHHKW/BHHPW corner region, direction-independent metric error
    print("\n  corner profile (deg-18), relative metric error along the centre->vertex ray s = t * vertex:")
    psi18, _, _ = load_psi_dp3(NPZ["deg-18"])
    u18 = slice_potential(chart, psi18)
    for t in (0.0, 0.5, 0.9, 0.99, 0.999):
        s = jnp.asarray(chart.verts_s[0]) * t
        e = float(rel_metric_err(jax.hessian(u18)(s), jax.hessian(lambda q: u_them(phi(q)) / 3.0)(s)))
        print(f"    t = {t:5.3f}: e = {e:.3e}")


if __name__ == "__main__":
    main()
