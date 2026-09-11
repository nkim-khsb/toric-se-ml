"""[R2b-adm-W] An EXACTLY admissible and COMPLETE trial space for the stream potential.

WHAT r2b_stream_admissible.py FOUND (v2, orthonormalized constraint basis).  On the
square, polynomials with exactly affine edge traces carry one non-affine vertex
datum (chi = x1 x2) and the kernel is exact.  On Y^{3,2} they carry NONE up to
degree 16, and on the pentagon and hexagon the "realizers" that appear at degree
16-20 have trace defects ~1e-9 with a shrinking singular-value gap: they are
near-admissible, not admissible.  The reason is elementary.  A polynomial affine on
every edge LINE takes one value at the intersection Q of two non-parallel edge
lines, so the affine traces of the two edges must agree at Q.  For d lines in
general position these conditions leave only the globally affine traces, and every
admissible polynomial is then  affine + prod_a ell_a * q , whose forms have all
divisor periods zero and, by Stokes (int theta^theta = P^T Q^+ P = 0), self-dual
energy exactly 1/2.  Exactly admissible POLYNOMIALS cannot approximate the harmonic
forms at all on a generic polygon (this script prints the rank of the
intersection-point conditions for each polygon).

THE FIX.  Add to prod_a ell_a * poly(D-d) the d Wachspress coordinates phi_a of the
convex polygon: rational, smooth on the closed polygon, phi_a(V_b) = delta_ab, EXACTLY
affine on every edge (linear interpolation of the endpoint values), and reproducing
affine functions (sum phi_a = 1, sum V_a phi_a = s).  Then
  * every trial chi has exactly affine edge traces  =>  every trial theta is in L^2
    and the type residual vanishes at the facets   =>  E is an honest L^2 ratio;
  * the space is COMPLETE for the problem: if chi_harm has vertex values v_a then
    chi_harm - sum v_a phi_a vanishes on every edge, hence equals prod ell_a * q
    with q smooth, and q is approximated by polynomials;
  * the vertex data enter through exactly d - 3 directions (the phi_a modulo the
    three affine combinations), which is the count of section 6 read off the ansatz.

PRE-REGISTERED (fixed before running):
  (W1) rank of the design after QR = d - 3 + (D-d+2)(D-d+1)/2 (the affine combinations
       of the phi_a drop out, nothing else does);
  (W2) exactly d-3 eigenvalues collapse on every polygon at every D >= d, gap growing
       with D; the rest sit at E = 1/2 up to the mixing with the realizer directions;
  (W3) numerator and denominator of the kernel vectors are flat in ngl to 1e-8 relative
       from ngl = 24 to 128 (bounded, piecewise-analytic integrands);
  (W4) Y^{3,2}: HEK ratio 11.898979 reproduced and converging with D;
  (W5) conifold: kernel at machine precision.

RESULT (2026-09-11, ngl = 48, see r2b_stream_wachspress.log).  Intersection-condition
rank = d-3 on Y^{3,2}, dP3, dP2 (0 on the square): exactly admissible polynomials carry
no vertex data anywhere but the square.  In the Wachspress space:
  conifold    kernel 2e-29 at every D, rest exactly 1/2.
  Y^{3,2}     D=4 (TWO functions): E = 6.3e-16, R = 11.898980 (1.4e-8), 1/sqrt(F) affine 1.8e-9;
              D=16: 4.2e-18, R 9.9e-10.  The HEK potential is rational and sits in this span.
  dP3         D=6: 1.4e-4 3.4e-4 3.4e-4 | 0.5 ; D=12: 4.2e-7 4.2e-7 1.7e-6 ;
              D=16 (69): 1.17e-9 1.17e-9 5.56e-9 ; D=20 (123): 2.45e-12 2.45e-12 1.24e-11 | 0.5, gap 4e10.
              The A_2 pair is exactly degenerate.  Polynomial space at D=16 was 6.2e-11 x2, 3.2e-10
              (20x lower, bought by leaving L^2: edge-trace defect 2-4e-5).
  dP2 at b*   D=8: 1.60e-5 6.32e-5 ; D=12: 7.21e-8 2.86e-7 ; D=16 (80): 1.39e-10 5.72e-10 | 0.5, gap 8.7e8.
  Quadrature scan of the kernel vectors, ngl 24 -> 128: dP3 num/den change 1.6e-13..5.7e-13 /
  <=2.4e-14, dP2 1.7e-12, 8.5e-13 / <=2.4e-14; edge-trace defects 1e-15.
GATES: W1, W2, W4, W5 pass.  (W3) as REGISTERED fails on the conifold (numerator 1e-30 -> 6e-28,
"relative change 5e2") and on Y^{3,2} (numerator 4e-18, relative change 2.5e-8 > 1e-8): the gate
was written as a RELATIVE change with no absolute floor, and both numerators are roundoff, so the
measure is undefined there rather than the integrals unstable (the denominators are flat to
1e-14).  The gate is not moved; the estimator was the wrong one for a zero.  Where the numerator
is a number (dP3, dP2) it passes by three decades.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_stream_wachspress.py [conifold y32 dp3 dp2 | all]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments" / "dp3kt", ROOT / "experiments" / "dp2"):
    sys.path.insert(0, str(p))

from sugrasol.forms import hodge                                    # noqa: E402
from r2b_ypq_calibration import load_case, base_metric, affine_probe  # noqa: E402
from r2b_holomorphic import load_dp3                                # noqa: E402
from r2b_stream_potential import poly_quad                          # noqa: E402
from r2b_stream_admissible import (EPSM, forms_from_hess, flat_weighted, solve,  # noqa: E402
                                   NGL_REF, NGL_SCAN)


def wachspress_space(verts, D):
    """Return B: s -> (K,) with the d Wachspress coordinates followed by
    prod_a A_a(s) * s1^p s2^q, p+q <= D-d, in the chart scaled by rho = max|V|."""
    V = np.asarray(verts)
    d = len(V)
    rho = float(np.max(np.linalg.norm(V, axis=1)))
    Vs = jnp.asarray(V / rho)

    def area(p, q, r):
        return 0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    C = jnp.array([area(Vs[a - 1], Vs[a], Vs[(a + 1) % d]) for a in range(d)])
    pw = [(i, t - i) for t in range(0, D - d + 1) for i in range(t + 1)] if D >= d else []
    # the d Wachspress coordinates reproduce affine functions (sum phi = 1, sum V phi = s),
    # whose Hessians vanish; remove those three combinations ANALYTICALLY, keeping the
    # d-3 directions c with sum c = 0, sum c V = 0 -- the vertex data modulo affine.
    Aff = np.column_stack([np.ones(d), V / rho])            # (d, 3)
    _, _, Vt = np.linalg.svd(Aff.T, full_matrices=True)
    Nc = jnp.asarray(Vt[3:].T)                              # (d, d-3), orthonormal columns

    def B(s):
        s = s / rho
        A = jnp.stack([area(s, Vs[a], Vs[(a + 1) % d]) for a in range(d)])
        prod = jnp.prod(A)
        # Wachspress: w_a = C_a / (A_{a-1} A_a), written with the polynomial numerator
        # C_a prod_{b != a-1, a} A_b so that it is smooth on the closed polygon
        w = jnp.stack([C[a] * jnp.prod(jnp.stack([A[b] for b in range(d) if b not in ((a - 1) % d, a)]))
                       for a in range(d)])
        phi = w / jnp.sum(w)
        mono = jnp.stack([s[0] ** p * s[1] ** q for (p, q) in pw]) if pw else jnp.zeros(0)
        return jnp.concatenate([phi @ Nc, prod * mono])

    return B, d - 3 + len(pw), rho


def intersection_rank(verts, tol=1e-9):
    """Rank of the conditions 'affine traces of two non-adjacent edge lines agree at
    their intersection' on the d vertex values.  Rank d-3 means only affine data
    survive, i.e. no exactly admissible polynomial carries vertex data."""
    V = np.asarray(verts)
    d = len(V)
    rows = []
    for a in range(d):
        for c in range(a + 1, d):
            if (c - a) % d in (1, d - 1):
                continue                                  # adjacent: share a vertex
            P0, t0 = V[a - 1], V[a] - V[a - 1]
            Q0, t1 = V[c - 1], V[c] - V[c - 1]
            M = np.column_stack([t0, -t1])
            if abs(np.linalg.det(M)) < 1e-12 * np.linalg.norm(t0) * np.linalg.norm(t1):
                continue                                  # parallel: meet at infinity
            lam, mu = np.linalg.solve(M, Q0 - P0)
            row = np.zeros(d)                             # aff_a(Q) - aff_c(Q) in vertex values
            row[a - 1] += 1 - lam
            row[a] += lam
            row[c - 1] -= 1 - mu
            row[c] -= mu
            rows.append(row)
    if not rows:
        return 0, 0
    sv = np.linalg.svd(np.array(rows), compute_uv=False)
    return int(np.sum(sv > tol * sv[0])), len(rows)


def hess_basis(B, nodes):
    return np.asarray(jax.vmap(jax.hessian(B))(jnp.asarray(nodes)))   # (n,K,2,2)


def design(ch, u, B, ngl, coef=None):
    smp, wt = poly_quad(ch, ngl)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    H = hess_basis(B, smp)
    if coef is not None:
        H = np.einsum("nkij,km->nmij", H, coef)
    A, W = forms_from_hess(H)
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(jnp.asarray(W), gs)
    X = flat_weighted(W, gs, wt)
    Xs = flat_weighted(0.5 * (jnp.asarray(W) + st), gs, wt)
    return X, Xs, smp, A


def edge_defect(B, verts, coef, interior):
    V = np.asarray(verts)
    d = len(V)
    Hin = np.einsum("nkij,km->nmij", hess_basis(B, interior), coef)
    scale = np.maximum(np.max(np.abs(Hin), axis=(0, 2, 3)), 1e-300)
    worst = np.zeros(coef.shape[1])
    tn = np.linspace(0.0, 1.0, 41)                         # the whole edge, vertices included
    for a in range(d):
        P0, tau = V[a - 1], V[a] - V[a - 1]
        pts = P0[None, :] + tn[:, None] * tau[None, :]
        H = np.einsum("nkij,km->nmij", hess_basis(B, pts), coef)
        e = np.abs(np.einsum("i,nmij,j->nm", tau, H, tau)) / (tau @ tau)
        worst = np.maximum(worst, e.max(0))
    return worst / scale


def run_case(tag, ch, u, d, y1=None, y2=None, Ds=(8, 12, 16), D_scan=16):
    nk = d - 3
    Rh = None if y1 is None else ((1 - y1) / (1 - y2)) ** 2
    verts = np.asarray(ch.verts_s)
    r_int, n_int = intersection_rank(verts)
    print(f"\n=== {tag}: d = {d}, kernel d-3 = {nk}" + (f", HEK R = {Rh:.6f}" if Rh else "") + " ===")
    print(f"  intersection-point conditions on vertex data: {n_int} conditions, rank {r_int}; "
          f"data surviving = {d - r_int} (affine = 3) => exactly admissible POLYNOMIALS carry "
          f"{d - r_int - 3} vertex directions", flush=True)
    print(f"  Wachspress + prod(ell)*poly space, ngl = {NGL_REF}:")
    print(f"  {'D':>3} {'K':>4} {'rank':>4} {'pred':>4} | lowest {nk + 2} SD eigenvalues"
          f"{'':>{max(0, 11 * (nk + 2) - 24)}} | gap" + ("   R (HEK)" if Rh else ""))
    fails, keep, gaps = [], {}, []
    for D in Ds:
        B, K, rho = wachspress_space(verts, D)
        X, Xs, smp, A = design(ch, u, B, NGL_REF)
        lam, coef, r = solve(X, Xs)
        pred = d - 3 + ((D - d + 2) * (D - d + 1) // 2 if D >= d else 0)
        gap = lam[nk] / lam[nk - 1] if lam[nk - 1] > 0 else np.inf
        line = (f"  {D:>3} {K:>4} {r:>4} {pred:>4} | " + "  ".join(f"{v:9.2e}" for v in lam[:nk + 2])
                + f" | {gap:8.1e}x")
        if Rh:
            A0 = np.einsum("nkij,k->nij", A, coef[:, 0])
            lamv = np.sqrt(np.maximum(-np.linalg.det(A0), 0.0))
            ok = lamv > 1e-8 * lamv.max()
            rr, _, Rv = affine_probe(np.asarray(smp)[ok], lamv[ok], verts)
            line += f"   {Rv:.6f} ({abs(Rv / Rh - 1):.1e}, affine {rr:.1e})"
        print(line, flush=True)
        if r != pred:
            fails.append(f"(W1) {tag} D={D}: rank {r} != {pred}")
        if lam[nk] / max(lam[nk - 1], 1e-300) < 1e3:
            fails.append(f"(W2) {tag} D={D}: fewer than {nk} collapsed")
        gaps.append(gap)
        keep[D] = (B, coef, lam, smp)
    if not all(g2 > g1 for g1, g2 in zip(gaps, gaps[1:])):
        fails.append(f"(W2) {tag}: gap not growing: {gaps}")

    B, coef, lam, smp = keep[D_scan]
    dfx = edge_defect(B, verts, coef[:, :nk + 1], smp)
    print(f"  edge-trace defect (kernel vectors, then first non-kernel): "
          + "  ".join(f"{v:.1e}" for v in dfx) + "   (exact zero up to roundoff)")
    print(f"\n  quadrature scan, D = {D_scan}, coefficient vectors fixed at ngl = {NGL_REF}:")
    print(f"  {'ngl':>4} | " + " | ".join(f"{'num_'+str(i):>10} {'den_'+str(i):>10}" for i in range(nk))
          + " |  first non-kernel E")
    table = []
    for ngl in NGL_SCAN:
        X, Xs, _, _ = design(ch, u, B, ngl, coef=coef[:, :nk + 1])
        num, den = np.sum(Xs ** 2, 0), np.sum(X ** 2, 0)
        table.append((ngl, num, den))
        print(f"  {ngl:>4} | " + " | ".join(f"{num[i]:10.4e} {den[i]:10.4e}" for i in range(nk))
              + f" | {num[nk]/den[nk]:10.4e}", flush=True)
    i24 = [t[0] for t in table].index(24)
    dnum = np.abs(table[-1][1][:nk] / table[i24][1][:nk] - 1)
    dden = np.abs(table[-1][2][:nk] / table[i24][2][:nk] - 1)
    print(f"     relative change ngl 24->{NGL_SCAN[-1]}: num " + " ".join(f"{v:.1e}" for v in dnum)
          + " ; den " + " ".join(f"{v:.1e}" for v in dden))
    if np.any(dnum > 1e-8) or np.any(dden > 1e-8):
        fails.append(f"(W3) {tag}: kernel num/den drift {dnum}, {dden}")
    return fails


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    t0 = time.time()
    FAILS = []
    if "all" in which or "conifold" in which:
        ch, u, y1, y2 = load_case("conifold")
        FAILS += run_case("conifold (square)", ch, u, 4, Ds=(4, 8, 12, 16))
    if "all" in which or "y32" in which:
        ch, u, y1, y2 = load_case((3, 2))
        FAILS += run_case("Y^{3,2}", ch, u, 4, y1, y2, Ds=(4, 8, 12, 16))
    if "all" in which or "dp3" in which:
        ch, u, _, _ = load_dp3()
        FAILS += run_case("dP3 (hexagon)", ch, u, 6, Ds=(6, 8, 12, 16, 20))
    if "all" in which or "dp2" in which:
        from harmonic_forms_dp2 import se_potential, PAPER_HELD
        print("\n--- refitting the dP2 SE potential at b* ---", flush=True)
        ch, u, held = se_potential()
        print(f"  gate (0): held-out {held:.2e} vs paper {PAPER_HELD:.2e}: "
              f"{'PASS' if abs(held / PAPER_HELD - 1) < 1 else 'FAIL'}")
        FAILS += run_case("dP2 (pentagon) at b*", ch, u, 5, Ds=(5, 8, 12, 16))
    print(f"\nFAILURES: {FAILS if FAILS else 'none'}   ({time.time() - t0:.0f} s)")
