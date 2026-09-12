"""The self-dual energy of DHHKW's ansatz as its truncation order is escalated.

WHY.  Section 6.4 of the draft grades their published sixth-order form with the
energy (eq:sdenergy) and gets 2.0e-6, 2.0e-6, 1.2e-5 against our 1.17e-9,
1.17e-9, 5.56e-9 at D=16.  It then lists four things that enter that ratio --
degree, the four-decimal rounding of their printed coefficients, their fitting
domain, and the metric -- and measures exactly one of them, the degree, in a
DIFFERENT functional: the constancy residual of (3.46), which falls 2.9e-3 ->
6.2e-4 -> 1.2e-4 from order six to order ten (dhhkw_theta_compare.py, block E).
A caveat stated in one functional and measured in another is not a measurement.
This script puts the ladder in the same units as the comparison.

WHAT IS HELD FIXED.  The refit is the one section 3.5 quotes: weighted least
squares against the constancy condition on the deg-18 metric over the whole
hexagon, integrated by polygon_quadrature, so the constancy column here IS the
paper's ladder and must print 3.106e-03, 6.763e-04, 1.366e-04.  Fitting and
grading use different Gauss orders, 28 and 48, so the energy is not read back
on the nodes the coefficients were chosen to fit.  (An earlier version fitted
on a 20,000-point cloud with a 1e-3 margin, which is the estimator section 3.5
moved away from: the margin drops the collar where a polynomial is worst.)
The grading is the path of dhhkw_theta_sdenergy: the D_6 orbit of mu_2, wrapped
into the symplectic frame, flattened against the deterministic centroid-fan
quadrature, orthogonalized by pivoted QR, diagonalized.  Only the coefficient
vector changes down the table.  (The orbit is fused into one vmap over the
group here, which is a compile-time change and not a numerical one: at ngl 8
both spellings return 2.038e-06, 2.038e-06, 1.170e-05.)

WHAT THIS SEPARATES, AND WHAT IT DOES NOT.  Published-vs-refit-at-order-six
isolates rounding + fitting domain + their metric, since the order is the same.
Refit six -> eight -> ten isolates the truncation order, everything else fixed.
What remains unseparated is their metric against ours, which needs their metric
and we do not have it.

THE RANK, AND A PRE-REGISTERED GATE WITHDRAWN BEFORE THE RUN.  (P1) below was
registered as "rank 4 on the span", b_2(dP3) = 4 being the number of independent
classes among the six theta_a.  A smoke run at ngl 8 on the PUBLISHED row -- a
free anchor, not ladder data -- returned rank 6, and the two extra directions
came out at 0.4999.  That is not a failure of the geometry, it is the estimator:
the two relations among the six toric divisors are exact for exact theta_a and
only approximate for a fitted one, so the fit leaves two extra directions whose
divisor periods nearly vanish, and by the Stokes argument of section 6.2 a
zero-period direction sits at exactly 1/2.  The "largest drop" heuristic of
dhhkw_theta_sdenergy then has two candidate gaps to choose between -- the exact
collapse of the twelve group elements onto six facets at 1e-14, and the soft
6 -> 4 drop of the approximate relations -- and which one wins is not a property
of the fit.  So this script cuts at the unambiguous gap, keeps all six
directions, and reports the whole spectrum, which is more informative than the
rank: three energies below 1/2, two at 1/2, one at 1.  P1 is replaced by P1' in
those terms.  The three graded energies are unaffected either way.

PRE-REGISTERED, fixed before the first production run:
  (P1') the six-dimensional span separates as three energies below 1/2, two at
       1/2 to 1e-3, and one at 1.000: the last is the Kahler direction and the
       middle two are the zero-period directions left by an imperfect fit.
  (P2) the Kahler direction at 1.000 to 1e-3 at every order.  It is purely
       self-dual and saturates the bound; it cannot move, and if it does the
       grading is broken rather than the fit.
  (P3) the two lowest degenerate to 1e-3 relative at every order: D_6 acts
       irreducibly on the A_2 block and the refit preserves D_6 by construction.
  (P4) the published row reproduces 2.0e-6, 2.0e-6, 1.2e-5 to 5%.  Free anchor:
       it is the number already printed in the draft, and it also checks this
       machine, jax 0.11.1 on Python 3.14, against the one the draft was run on.
  (P5) the energies fall monotonically with order.  If they do not, read the
       design condition number printed alongside BEFORE saying anything about
       truncation: the monomial design conditions badly with order, and a
       conditioning floor would imitate a truncation floor.
  (P6) band for the descent: E goes as the square of a pointwise error, so the
       constancy ratio 2.9e-3/1.2e-4 = 24 predicts about 600 in E.  Registered
       band, wide on purpose: E(order 10)/E(refit order 6) between 1e-4 and
       1e-1.  Outside it, the two functionals are not tracking each other and
       the draft should say only what it measured.
  (P7) their order-ten refit still lies above our D=16 energies, 1.17e-9.  If it
       does not, section 6.4's comparison needs rewriting, not decorating.

Usage: PYTHONPATH=. python experiments/dp3/dhhkw_theta_sdenergy_ladder.py
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "dp3kt"))
sys.path.insert(0, str(ROOT / "experiments" / "dp3"))

jax.config.update("jax_enable_x64", True)

from sugrasol.forms import hodge                                     # noqa: E402
from r2b_ypq_calibration import base_metric, S1, P1, S2, P2          # noqa: E402
from r2b_holomorphic import load_dp3                                 # noqa: E402
from r2b_stream_potential import poly_quad, UT                       # noqa: E402
from dhhkw_theta_compare import (GROUP, CD, POWERS6, NPZ, TO_THEM,
                                 mu_log, mu_powers, sym_powers, build_design,
                                 laplacian_op, load_psi_dp3)         # noqa: E402
from dhhkw_theta_sdenergy import wrap                                # noqa: E402

OURS_D16 = (1.17e-9, 1.17e-9, 5.56e-9)     # section 6.4, deterministic quadrature
OURS_D20 = (2.45e-12, 2.45e-12, 1.24e-11)
PUBLISHED = (2.0e-6, 2.0e-6, 1.2e-5)       # the row the draft prints
GAP_CUT = 1e-10                            # the duplicate collapse sits at 1e-14


def wrefit(op, nodes, W, powers):
    """Least squares against the constancy condition, weighted by the quadrature.

    The fit minimizes the integral of (L[mu] - c)^2 over the polygon rather than
    a sum over sample points, so the design rows carry sqrt(w).  Linear in the
    coefficients either way: one lstsq, no optimizer.
    """
    A, b = build_design(op, nodes, powers, False)
    r = np.sqrt(W)[:, None]
    sol, _, _, sv = np.linalg.lstsq(A * r, b * r[:, 0], rcond=None)
    return (sol[:len(powers)], TO_THEM * sol[-1],
            float(sv[0] / sv[-1]))


def make_A(mu, u):
    """A = (1/2) U d_x(U^{-1} grad f) U^{-1} for f = mu(g . s), g a traced arg.

    Same formula as dhhkw_theta_sdenergy.A_of; g is an argument rather than a
    closure so the whole orbit compiles once instead of twelve times.
    """
    def A(s, g):
        f = lambda z: mu(g @ z)
        U = jax.hessian(u)(s)
        Ui = jnp.linalg.inv(U)
        flow = lambda z: jnp.linalg.inv(jax.hessian(u)(z)) @ jax.grad(f)(z)
        return 0.5 * U @ jax.jacobian(flow)(s) @ Ui
    return jax.jit(jax.vmap(jax.vmap(A, in_axes=(0, None)), in_axes=(None, 0)))


def grade(mu, smp, wt, u, gs, L_chol):
    """Self-dual energies of the span of the theta_a built from mu."""
    As = make_A(mu, u)(smp, jnp.asarray(np.asarray(GROUP)))    # (g, n, 2, 2)
    oms = jax.vmap(lambda AA: jax.vmap(wrap)(AA))(As)          # (g, n, 4, 4)
    oms = jnp.transpose(oms, (1, 0, 2, 3))                     # (n, g, 4, 4)
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(oms, gs)

    def flat(O):
        Ot = jnp.einsum("nai,nkab,nbj->nkij", L_chol, O, L_chol)
        v = jnp.stack([Ot[:, :, i, j] for (i, j) in UT], -1)
        M = np.asarray(jnp.transpose(v, (0, 2, 1)).reshape(-1, v.shape[1]))
        return M * np.repeat(np.sqrt(wt), 6)[:, None]

    X, Xs = flat(oms), flat(0.5 * (oms + st))
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    ratios = d / d[0]
    rank = int(np.sum(ratios > GAP_CUT))            # the unambiguous gap
    drop = int(np.argmax(ratios[:-1] / ratios[1:]) + 1)   # what the heuristic says

    T = np.zeros((X.shape[1], rank))
    T[piv[:rank], :] = np.linalg.inv(R[:rank, :rank])
    Z = Xs @ T
    lam = np.linalg.eigvalsh(0.5 * (Z.T @ Z + (Z.T @ Z).T))
    tail = float(ratios[rank]) if rank < len(ratios) else float("nan")
    return rank, drop, lam, tail


def main():
    t0 = time.time()
    ch, u, _, _ = load_dp3()
    ngl = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    smp, wt = poly_quad(ch, ngl, ngl)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    L_chol = jnp.linalg.cholesky(jnp.linalg.inv(gs))
    print(f"grading nodes: {len(smp)} (centroid fan, {ngl} per direction)")

    # the fit, identical to section 4.3: deg-18 metric, whole hexagon, seed 5
    psi, ch_c, dat = load_psi_dp3(NPZ["deg-18"])
    Lop = laplacian_op(ch_c, psi)
    ngl_fit = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    ss, wfit = poly_quad(ch_c, ngl_fit, ngl_fit)
    Wfit = np.asarray(wfit) / float(np.sum(wfit))
    print(f"fitting nodes: {len(ss)} (centroid fan, {ngl_fit} per direction), "
          f"deg-18 metric ({dat['W'].shape[1]} params)\n", flush=True)

    def constancy(mu, pts, W=Wfit):
        v = TO_THEM * np.asarray(jax.jit(jax.vmap(Lop(mu)))(pts))
        m = float(W @ v)
        return float(np.sqrt(W @ (v - m) ** 2)), m

    rows = []
    mu_pub = lambda s: mu_log(s) + jnp.dot(jnp.asarray(CD), mu_powers(s, POWERS6))
    rows.append(("published (6.13)", len(CD), mu_pub, float("nan")))

    for order in (6, 8, 10):
        pw = sym_powers(order)
        coef, const, cond = wrefit(Lop, ss, Wfit, pw)
        c = jnp.asarray(coef)
        rows.append((f"refit, order {order}", len(pw),
                     (lambda s, c=c, pw=pw: mu_log(s) + jnp.dot(c, mu_powers(s, pw))),
                     cond))
        print(f"  fitted order {order}: {len(pw)} coefficients, constant "
              f"{const:.6f} (exact 0.666667), design cond {cond:.1e}",
              flush=True)

    print(f"\n{'row':>18} {'n_par':>6} {'cond':>9} {'rms(3.46)':>10} "
          f"{'E1':>10} {'E2':>10} {'E3':>10} {'E4':>7} {'E5':>7} {'Kahler':>8} "
          f"{'rank':>5} {'drop':>5}")
    out = []
    for tag, npar, mu, cond in rows:
        rms, mean = constancy(mu, ss)
        rank, drop, lam, tail = grade(mu, smp, wt, u, gs, L_chol)
        out.append((tag, npar, cond, rms, mean, lam, rank, drop, tail))
        cs = "--" if not np.isfinite(cond) else f"{cond:.1e}"
        ext = list(lam) + [float("nan")] * 6
        print(f"{tag:>18} {npar:6d} {cs:>9} {rms:10.2e} "
              f"{ext[0]:10.3e} {ext[1]:10.3e} {ext[2]:10.3e} "
              f"{ext[3]:7.4f} {ext[4]:7.4f} {ext[5]:8.4f} {rank:5d} {drop:5d}",
              flush=True)

    print("\npre-registered gates")
    fails = []

    def gate(name, ok, msg):
        print(f"  ({name}) {'pass' if ok else 'FAIL'}  {msg}")
        if not ok:
            fails.append(name)

    struct = all(len(r[5]) == 6 and r[5][2] < 0.5
                 and abs(r[5][3] - 0.5) < 1e-3 and abs(r[5][4] - 0.5) < 1e-3
                 and abs(r[5][5] - 1.0) < 1e-3 for r in out)
    gate("P1'", struct, "three below 1/2, two at 1/2, one at 1 on every row")
    kah = [r[5][-1] for r in out]
    gate("P2", all(abs(k - 1.0) < 1e-3 for k in kah),
         "Kahler direction: " + ", ".join(f"{k:.5f}" for k in kah))
    deg = [abs(r[5][1] / r[5][0] - 1.0) for r in out]
    gate("P3", all(d < 1e-3 for d in deg),
         "degeneracy of the two lowest: " + ", ".join(f"{d:.1e}" for d in deg))
    pub = out[0][5][:3]
    gate("P4", all(abs(a / b - 1.0) < 0.05 for a, b in zip(pub, PUBLISHED)),
         "published row against the draft: " + ", ".join(f"{a:.3e}" for a in pub))
    lad = [r[5][:3] for r in out[1:]]
    mono = all(lad[i + 1][j] < lad[i][j] for i in range(len(lad) - 1)
               for j in range(3))
    gate("P5", mono, "monotone descent with order"
         + ("" if mono else " -- read the cond column before the energies"))
    drops = [lad[-1][j] / lad[0][j] for j in range(3)]
    gate("P6", all(1e-4 <= d <= 1e-1 for d in drops),
         "E(order 10)/E(refit 6) = " + ", ".join(f"{d:.1e}" for d in drops)
         + "  [band 1e-4 to 1e-1; the constancy ratio predicts ~1.7e-3]")
    gate("P7", all(a > b for a, b in zip(lad[-1], OURS_D16)),
         "their order-10 refit against our D=16: " +
         ", ".join(f"{a/b:.1e}" for a, b in zip(lad[-1], OURS_D16)) + " times ours")

    print("\nfor the draft")
    p, r6, r10 = out[0][5][:3], lad[0], lad[-1]
    print("  published -> refit at the SAME order six: "
          + ", ".join(f"{a:.2e} -> {b:.2e}" for a, b in zip(p, r6)))
    print("    that step is rounding + their fitting domain + their metric, factor "
          + ", ".join(f"{a/b:.1f}" for a, b in zip(p, r6)))
    print("  refit six -> ten, everything else held: "
          + ", ".join(f"{a:.2e} -> {b:.2e}" for a, b in zip(r6, r10))
          + ", factor " + ", ".join(f"{a/b:.1f}" for a, b in zip(r6, r10)))
    print("  distances sqrt(2E): published "
          + ", ".join(f"{np.sqrt(2*a)*100:.2f}%" for a in p)
          + " | refit order ten "
          + ", ".join(f"{np.sqrt(2*a)*100:.3f}%" for a in r10)
          + " | ours D=16 "
          + ", ".join(f"{np.sqrt(2*a)*100:.4f}%" for a in OURS_D16))
    print(f"\nFAILURES: {'none' if not fails else ', '.join(fails)}")
    print(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
