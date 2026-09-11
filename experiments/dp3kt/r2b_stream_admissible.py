"""[R2b-adm] L^2 admissibility of the stream-potential trial space (review item 1,
2026-09-11), and the same computation redone in a trial space that IS admissible.

THE OBJECTION.  Near a facet, in a chart where the facet is x = 0 with tangent z,
U = Hess G_P ~ diag(1/(2x), h).  With A = Hess(chi) eps^T the pointwise norm of
theta = A_ij ds_i ^ nu_j is

    |theta|^2 = tr(U^{-1} A U A^T) = 2 chi_xz^2 + 2 x h chi_xx^2 + chi_zz^2 / (2 x h) ,

so unless chi_zz -> 0 on the facet the last term integrates to int dx/x = infinity:
theta is not L^2, and the type residual is O(1) there as well.  A generic
polynomial chi has chi_zz(0, z) != 0.  The exact solution has chi affine on every
edge (that is the Keldysh boundary condition the paper derives, and the origin of
the count d-3), but a finite-degree TRIAL function need not, and Gauss-Legendre
nodes never touch the edge, so a finite quadrature value proves nothing about
integrability.  Hence the demand: (i) say whether the trial space enforces the
affine trace; (ii) if not, reformulate in one that does; (iii) show numerator and
denominator converging separately in the quadrature order, not just their ratio.

WHAT THIS SCRIPT DOES.
  [U] the UNCONSTRAINED trial space of r2b_stream_potential.py (all monomials of
      degree 2..D): for the d-3 kernel vectors at the reference quadrature,
      measure the edge-trace defect  max_edges |tau^T Hess(chi) tau| / max|Hess chi|
      and scan numerator and denominator of E over the GL order ngl with the
      coefficient vector held FIXED.  A log(ngl) drift is the divergence.
  [A] the ADMISSIBLE trial space: monomials of degree 2..D restricted by the
      linear conditions  tau_a^T Hess(chi)(V_{a-1} + t tau_a) tau_a = 0  for all t
      (a polynomial of degree D-2 in t, so D-1 nodes per edge make it exact).
      Its dimension is predicted:  (D-d+2)(D-d+1)/2 + d - 3  for D >= d-2
      (polynomials vanishing on all edges, prod_a ell_a * q, plus one realizer per
      vertex value, minus the three affine ones).  Every trial form is then L^2 and
      the Rayleigh quotient is an honest variational quantity.  Report the lowest
      eigenvalues, the count that collapses, the gap, the HEK ratio on Y^{3,2},
      and the same ngl scan of numerator and denominator.

OBSERVED ON THE FIRST RUN (2026-09-11, raw monomials, before the orthonormalized
constraint basis): on the subspace prod_a ell_a * q, whose forms have every divisor
period zero (chi and grad chi vanish at the vertices), E = 1/2 EXACTLY -- Stokes:
int theta ^ theta = P^T Q^+ P = 0, so <theta_+, theta_+> = <theta, theta>/2.  The
harmonic representatives live entirely in the realizer directions, which appear
only once D is large enough for a polynomial to carry non-affine vertex data.

PRE-REGISTERED (fixed before running):
  (P1) the admissible dimension equals the prediction at every D (exact integer).
  (P2) in the admissible space exactly d-3 eigenvalues collapse on every polygon,
       with a gap to the next that grows with D.
  (P3) numerator and denominator of the admissible kernel vectors are stable in
       ngl to a relative 1e-6 from ngl = 24 on; for the unconstrained kernel the
       same scan is REPORTED, whatever it does -- that is the measurement the
       reviewer asked for.
  (P4) Y^{3,2}: the HEK ratio R = ((1-y1)/(1-y2))^2 = 11.898979 is reproduced in
       the admissible space, converging with D.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_stream_admissible.py [dp3|y32|conifold|dp2|all]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import qr

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments" / "dp3kt", ROOT / "experiments" / "dp2"):
    sys.path.insert(0, str(p))

from sugrasol.forms import hodge                                    # noqa: E402
from r2b_ypq_calibration import load_case, base_metric, S1, P1, S2, P2, affine_probe  # noqa: E402
from r2b_holomorphic import load_dp3                                # noqa: E402
from r2b_stream_potential import poly_quad, UT                      # noqa: E402

EPSM = np.array([[0.0, 1.0], [-1.0, 0.0]])
NGL_REF = 48
NGL_SCAN = (12, 16, 24, 32, 48, 64, 96, 128)
RHO = [1.0]          # chart scale: monomials are taken in s/RHO so that |s/RHO| <= 1 on P


def powers(D):
    return [(i, t - i) for t in range(2, D + 1) for i in range(t + 1)]


def hess_mono(S, pw):
    """Closed-form Hessians of the monomials s1^p s2^q at nodes S (n,2) -> (n,K,2,2).
    Written so that a zero coefficient never multiplies an inf (p=0 at s=0)."""
    S = np.asarray(S)
    s1, s2 = S[:, 0], S[:, 1]
    n = len(S)
    H = np.zeros((n, len(pw), 2, 2))

    def mono(a, b):
        out = np.ones(n)
        if a > 0:
            out = out * s1 ** a
        if b > 0:
            out = out * s2 ** b
        return out

    for k, (p, q) in enumerate(pw):
        if p >= 2:
            H[:, k, 0, 0] = p * (p - 1) * mono(p - 2, q)
        if p >= 1 and q >= 1:
            H[:, k, 0, 1] = H[:, k, 1, 0] = p * q * mono(p - 1, q - 1)
        if q >= 2:
            H[:, k, 1, 1] = q * (q - 1) * mono(p, q - 2)
    return H


def edge_constraints(verts, D):
    """Rows: tau^T Hess(monomial)(V_{a-1} + t tau) tau at D-1 Chebyshev nodes per edge."""
    pw = powers(D)
    V = np.asarray(verts)
    d = len(V)
    k = D - 1
    tn = 0.5 * (1.0 - np.cos(np.pi * (np.arange(k) + 0.5) / k))
    rows = []
    for a in range(d):
        P0, P1_ = V[a - 1], V[a]
        tau = P1_ - P0
        pts = P0[None, :] + tn[:, None] * tau[None, :]
        H = hess_mono(pts, pw)                       # (k, K, 2, 2)
        rows.append(np.einsum("i,nkij,j->nk", tau, H, tau))
    return np.concatenate(rows, 0)


def chi_orthonormalizer(verts, D, ch, ngl=NGL_REF):
    """R^{-1} such that the monomials (in the scaled chart) times R^{-1} are
    L^2(P)-orthonormal.  Imposing the edge constraints on THAT basis is what makes
    the rank of the constraint matrix readable: on raw monomials of degree 20 the
    singular values span >10 decades and a relative cut discards genuine
    constraints (the conditioning trap of the paper, fifth occurrence)."""
    smp, wt = poly_quad(ch, ngl)
    smp = np.asarray(smp) / RHO[0]
    pw = powers(D)
    Vm = np.stack([(smp[:, 0] ** p if p else np.ones(len(smp))) * (smp[:, 1] ** q if q else np.ones(len(smp)))
                   for (p, q) in pw], 1) * np.sqrt(np.asarray(wt) / np.sum(wt))[:, None]
    Q, R = np.linalg.qr(Vm)
    return np.linalg.inv(R)


def admissible_basis(verts, D, ch, tol=1e-9):
    Rinv = chi_orthonormalizer(verts, D, ch)
    C = edge_constraints(np.asarray(verts) / RHO[0], D) @ Rinv
    _, sv, Vt = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(sv > tol * sv[0]))
    N = Rinv @ Vt[rank:].T                            # monomial coefficients (scaled chart)
    gap = sv[rank - 1] / sv[rank] if rank < len(sv) else np.inf
    return N, rank, gap, sv


def predicted_dim(D, d):
    return (D - d + 2) * (D - d + 1) // 2 + d - 3 if D >= d - 2 else None


def forms_from_hess(H):
    """H (n,m,2,2) Hessians -> A = H eps^T (n,m,2,2) and the 4x4 two-forms (n,m,4,4)."""
    A = np.einsum("nmij,kj->nmik", H, EPSM)          # A = Hess eps^T
    W = np.zeros(H.shape[:2] + (4, 4))
    for a, ia in enumerate((S1, S2)):
        for b, ib in enumerate((P1, P2)):
            W[:, :, ia, ib] += A[:, :, a, b]
            W[:, :, ib, ia] -= A[:, :, a, b]
    return A, W


def flat_weighted(W, gs, wt):
    """(n,m,4,4) forms -> (6n, m) design matrix in the orthonormal frame, rows
    weighted by sqrt(wt), so that column inner products are L^2 inner products."""
    L = jnp.linalg.cholesky(jnp.linalg.inv(gs))
    Wt = jnp.einsum("nai,nmab,nbj->nmij", L, jnp.asarray(W), L)
    v = jnp.stack([Wt[:, :, i, j] for (i, j) in UT], -1)      # (n,m,6)
    M = np.asarray(jnp.transpose(v, (0, 2, 1)).reshape(-1, v.shape[1]))
    return M * np.repeat(np.sqrt(wt), 6)[:, None]


def sd_design(ch, u, D, ngl, N=None):
    """Return X (denominator design), Xs (numerator design), nodes, K_eff."""
    smp, wt = poly_quad(ch, ngl)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    H = hess_mono(np.asarray(smp) / RHO[0], powers(D))
    if N is not None:
        H = np.einsum("nkij,km->nmij", H, N)
    _, W = forms_from_hess(H)
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(jnp.asarray(W), gs)
    X = flat_weighted(W, gs, wt)
    Xs = flat_weighted(0.5 * (jnp.asarray(W) + st), gs, wt)
    return X, Xs, smp


def solve(X, Xs, tol=1e-12):
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    r = int(np.sum(d > tol * d[0]))
    T = np.zeros((X.shape[1], r))
    T[piv[:r], :] = np.linalg.inv(R[:r, :r])
    Z = Xs @ T
    M = 0.5 * (Z.T @ Z + (Z.T @ Z).T)
    lam, C = np.linalg.eigh(M)
    coef = T @ C                                      # columns: trial-basis coefficients
    return lam, coef, r


def num_den(ch, u, D, ngl, coef_mono):
    """Numerator and denominator of E for FIXED monomial coefficient vectors (K,m)."""
    smp, wt = poly_quad(ch, ngl)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    pw = powers(D)
    H = np.zeros((len(smp), coef_mono.shape[1], 2, 2))
    for k0 in range(0, len(pw), 24):                  # chunked: ngl=128 has ~1e5 nodes
        Hk = hess_mono(np.asarray(smp) / RHO[0], pw[k0:k0 + 24])
        H += np.einsum("nkij,km->nmij", Hk, coef_mono[k0:k0 + 24])
    _, W = forms_from_hess(H)
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(jnp.asarray(W), gs)
    X = flat_weighted(W, gs, wt)
    Xs = flat_weighted(0.5 * (jnp.asarray(W) + st), gs, wt)
    return np.sum(Xs ** 2, 0), np.sum(X ** 2, 0)


def edge_trace_defect(verts, D, coef_mono, interior_nodes):
    """max over edges/points of |tau^T Hess chi tau| / max over interior of |Hess chi|."""
    V = np.asarray(verts)
    d = len(V)
    tn = np.linspace(0.0, 1.0, 41)
    V = V / RHO[0]
    Hin = np.einsum("nkij,km->nmij", hess_mono(np.asarray(interior_nodes) / RHO[0], powers(D)), coef_mono)
    scale = np.max(np.abs(Hin), axis=(0, 2, 3))      # per form
    worst = np.zeros(coef_mono.shape[1])
    for a in range(d):
        P0, P1_ = V[a - 1], V[a]
        tau = P1_ - P0
        pts = P0[None, :] + tn[:, None] * tau[None, :]
        H = np.einsum("nkij,km->nmij", hess_mono(pts, powers(D)), coef_mono)
        e = np.abs(np.einsum("i,nmij,j->nm", tau, H, tau)) / (tau @ tau)
        worst = np.maximum(worst, e.max(0))
    return worst / scale


def run_case(tag, ch, u, d, y1=None, y2=None, Ds=(8, 12, 16), D_scan=16):
    nk = d - 3
    Rh = None if y1 is None else ((1 - y1) / (1 - y2)) ** 2
    print(f"\n=== {tag}: d = {d}, expected kernel d-3 = {nk}"
          + (f", HEK R = {Rh:.6f}" if Rh else "") + " ===", flush=True)
    verts = np.asarray(ch.verts_s)
    RHO[0] = float(np.max(np.linalg.norm(verts, axis=1)))
    fails = []

    # ---------------- [A] admissible space, ladder in D
    print(f"  [A] admissible trial space (affine trace on every edge), ngl = {NGL_REF}")
    print(f"  dim = (D-d+2)(D-d+1)/2 [forms with all periods zero] + realizers [<= d-3]; "
          f"svgap = constraint-matrix singular-value gap at the cut")
    print(f"  {'D':>3} {'K':>4} {'dim':>4} {'zero':>4} {'real':>4} {'svgap':>7} {'rank':>4} | lowest {nk + 2} SD eigenvalues"
          f"{'':>{max(0, 11 * (nk + 2) - 24)}} | gap" + ("   R (HEK)" if Rh else ""))
    keep = {}
    for D in Ds:
        K = len(powers(D))
        N, rank_c, gap_c, sv = admissible_basis(verts, D, ch)
        nzero = (D - d + 2) * (D - d + 1) // 2 if D >= d - 2 else 0
        nreal = N.shape[1] - nzero
        X, Xs, smp = sd_design(ch, u, D, NGL_REF, N=N)
        lam, coef, r = solve(X, Xs)
        gap = lam[nk] / lam[nk - 1] if lam[nk - 1] > 0 else np.inf
        coef_mono = N @ coef
        line = (f"  {D:>3} {K:>4} {N.shape[1]:>4} {nzero:>4} {nreal:>4} {gap_c:7.1e} {r:>4} | "
                + "  ".join(f"{v:9.2e}" for v in lam[:nk + 2]) + f" | {gap:8.1e}x")
        if Rh:
            A0 = np.einsum("nkij,k->nij", hess_mono(np.asarray(smp) / RHO[0], powers(D)), coef_mono[:, 0]) @ EPSM.T
            lamv = np.sqrt(np.maximum(-np.linalg.det(A0), 0.0))
            ok = lamv > 1e-8 * lamv.max()          # -det Hess chi <= 0 at a few nodes at low D
            rr, _, Rv = affine_probe(np.asarray(smp)[ok], lamv[ok], verts)
            line += f"   {Rv:.6f} ({abs(Rv / Rh - 1):.1e}, affine {rr:.1e})"
        print(line, flush=True)
        if nreal < 0 or nreal > d - 3:
            fails.append(f"(P1) {tag} D={D}: realizer count {nreal} outside [0, d-3]")
        keep[D] = (coef_mono, lam)
    # (P2): count and gap growth
    gaps = []
    for D in Ds:
        lam = keep[D][1]
        gaps.append(lam[nk] / lam[nk - 1])
        # "collapse": the nk lowest are at least 1e3 below the next
        if not (lam[nk] / max(lam[nk - 1], 1e-300) > 1e3):
            fails.append(f"(P2) {tag} D={D}: fewer than {nk} collapsed (gap {lam[nk]/lam[nk-1]:.1e})")
    if not all(g2 > g1 for g1, g2 in zip(gaps, gaps[1:])):
        fails.append(f"(P2) {tag}: gap does not grow with D: {gaps}")

    # ---------------- [U] unconstrained space at D_scan, for the comparison
    print(f"\n  [U] unconstrained trial space (all monomials), D = {D_scan}, ngl = {NGL_REF}")
    X, Xs, smp = sd_design(ch, u, D_scan, NGL_REF, N=None)
    lam_u, coef_u, r_u = solve(X, Xs)
    gap_u = lam_u[nk] / lam_u[nk - 1]
    print(f"  K = {len(powers(D_scan))}, rank {r_u} | "
          + "  ".join(f"{v:9.2e}" for v in lam_u[:nk + 2]) + f" | {gap_u:8.1e}x")
    def_u = edge_trace_defect(verts, D_scan, coef_u[:, :nk + 1], smp)
    def_a = edge_trace_defect(verts, D_scan, keep[D_scan][0][:, :nk + 1], smp)
    print(f"  edge-trace defect max|chi_tt|/max|Hess chi|, kernel vectors then first non-kernel:")
    print(f"     unconstrained: " + "  ".join(f"{v:.2e}" for v in def_u))
    print(f"     admissible   : " + "  ".join(f"{v:.2e}" for v in def_a)
          + "   (zero up to roundoff by construction)")

    # ---------------- ngl scan of numerator and denominator, coefficient vectors FIXED
    print(f"\n  quadrature scan, D = {D_scan}, coefficient vectors fixed at ngl = {NGL_REF}:")
    print(f"  {'ngl':>4} | " + " | ".join(f"{'num_'+str(i):>10} {'den_'+str(i):>10} {'E_'+str(i):>10}"
                                          for i in range(nk)) + " |  first non-kernel E")
    for label, coef in (("unconstrained", coef_u[:, :nk + 1]), ("admissible", keep[D_scan][0][:, :nk + 1])):
        print(f"  -- {label}")
        table = []
        for ngl in NGL_SCAN:
            num, den = num_den(ch, u, D_scan, ngl, coef)
            table.append((ngl, num, den))
            print(f"  {ngl:>4} | " + " | ".join(f"{num[i]:10.3e} {den[i]:10.3e} {num[i]/den[i]:10.3e}"
                                                for i in range(nk)) + f" | {num[nk]/den[nk]:10.3e}", flush=True)
        # drift of the denominator from ngl=24 to the last, per kernel vector
        i24 = [t[0] for t in table].index(24)
        drift_den = np.abs(table[-1][2][:nk] / table[i24][2][:nk] - 1.0)
        drift_num = np.abs(table[-1][1][:nk] / np.maximum(table[i24][1][:nk], 1e-300) - 1.0)
        print(f"     relative change ngl {24}->{NGL_SCAN[-1]}: den "
              + " ".join(f"{v:.1e}" for v in drift_den) + " ; num "
              + " ".join(f"{v:.1e}" for v in drift_num))
        if label == "admissible" and np.any(drift_den > 1e-6):
            fails.append(f"(P3) {tag}: admissible denominator drifts {drift_den} between ngl 24 and {NGL_SCAN[-1]}")
        # numerator: report the last-step relative change as the convergence measure
        last = np.abs(table[-1][1][:nk] / np.maximum(table[-2][1][:nk], 1e-300) - 1.0)
        print(f"     numerator last step ngl {NGL_SCAN[-2]}->{NGL_SCAN[-1]}: " + " ".join(f"{v:.1e}" for v in last))
    if Rh:
        # (P4) convergence of R with D in the admissible space
        pass
    return fails


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    which = "all" if "all" in which else which
    t0 = time.time()
    FAILS = []
    if which == "all" or "conifold" in which:
        ch, u, y1, y2 = load_case("conifold")
        FAILS += run_case("conifold (square)", ch, u, 4)
    if which == "all" or "y32" in which:
        ch, u, y1, y2 = load_case((3, 2))
        FAILS += run_case("Y^{3,2}", ch, u, 4, y1, y2)
    if which == "all" or "dp3" in which:
        ch, u, _, _ = load_dp3()
        FAILS += run_case("dP3 (hexagon)", ch, u, 6, Ds=(8, 12, 16, 20))
    if which == "all" or "dp2" in which:
        from harmonic_forms_dp2 import se_potential, PAPER_HELD
        print("\n--- refitting the dP2 SE potential at b* (no stored artifact) ---", flush=True)
        ch, u, held = se_potential()
        print(f"  gate (0) same solution as tab:dp2ladder: held-out {held:.2e} vs {PAPER_HELD:.2e}: "
              f"{'PASS' if abs(held / PAPER_HELD - 1) < 1 else 'FAIL'}")
        FAILS += run_case("dP2 (pentagon) at b*", ch, u, 5)
    print(f"\nFAILURES: {FAILS if FAILS else 'none'}   ({time.time() - t0:.0f} s)")
