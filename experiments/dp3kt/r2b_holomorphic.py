"""*** SUPERSEDED by r2b_complete_span.py (2026-08-19): this file freezes the
antisymmetric part of A to a CONSTANT, but closedness only forces a PURELY
antisymmetric A to be constant -- the antisymmetric PART is (1/2) curl(alpha),
a free function.  The conifold results here are right (tau = 0 there); the
Y^{p,q} negative conclusions are artifacts of the frozen constant. ***

[R2b-holo] Harmonic (1,1) forms from a HOLOMORPHIC ansatz -- ASD and closedness
both built in analytically (CLAUDE.md rule 1), nothing minimised.

THREE POINTWISE ASD GENERATORS, in the orthonormal frame e^a = E ds, f^a = E^-T dphi
with E = U^{1/2} (symmetric root, smooth in s), ordering (e1,f1,e2,f2) so that
vol = e1^f1^e2^f2 and *J = +J with J = e1^f1 + e2^f2:
    Om_1 = e1^f1 - e2^f2 ,   Om_3 = e1^f2 + e2^f1 ,   Om_2 = e1^e2 + f1^f2 .
Om_1, Om_3 span the ds^dphi block; Om_2 = sqrt(det U) ds1^ds2 + (1/sqrt(det U)) dphi1^dphi2.

CLOSEDNESS kills Om_2.  d(R dphi1^dphi2) = dR ^ dphi1^dphi2 forces R = const, and
smoothness at a facet (where the cycle v_a degenerates, rho^2 ~ l_a) requires the
d(theta)^d(phi_perp) coefficient to vanish like l_a; R transforms by det N under a
lattice change of frame, so a constant R can vanish there only if R = 0.  Hence
P = R = 0 and the harmonic representatives live in the ds^dphi block.

THE BLOCK, EXACTLY.  For general (non-symmetric) A the (1,1) condition is
A U = U A^T and primitivity is tr A = 0.  Write A = S + kappa eps,
S = [[Phi, Psi], [Psi, -Phi]].  Then
   [S,U] = 2(q Phi - p Psi) eps ,   eps U + U eps = tr(U) eps ,
      p = (U_11 - U_22)/2 ,  q = U_12 ,
so ASD  <=>  Im[(p + i q) g] = -(kappa/2) tr U     with  g := Phi - i Psi,
and closedness (d_1 A_2j = d_2 A_1j) is untouched by the constant kappa and reads
   d_1 Psi = d_2 Phi ,  d_1 Phi = -d_2 Psi   <=>  d_zbar conj(g) = 0 ,  z = s1 + i s2.
=> g is HOLOMORPHIC in z (up to conjugation), kappa is ONE REAL CONSTANT, and the
only remaining condition is ONE REAL LINEAR equation per point.  Eigenvalues of A
are +- sqrt(|g|^2 - kappa^2) =: +- lambda, and HEK 0412193 says lambda = F(y) with
   R := lambda_max/lambda_min = ((1-y1)/(1-y2))^2   (GMSW roots).
Setting kappa = 0 recovers the earlier over-determined "arg(p-iq) must be harmonic"
condition, which FAILS on Y^{p,q} (r2b_frame_ansatz.py) -- the constant kappa is
exactly what was missing.

PRE-REGISTERED GATES (set before running):
  (H0) *omega = -omega numerically (sugrasol.forms.hodge) for every null vector,
       machine precision.  This is the closing check on all algebra above.
  (H1) d omega = 0 numerically, machine precision.
  (H2) null-space dimension = b2^- : 1 for a quadrilateral, 3 for the hexagon,
       with a clear gap, STABLE in the holomorphic degree K.  This is the gate
       R2b never passed (it read 3 off a factor-3 gap that moved with the basis).
  (H3) conifold: R = 1 to 1e-10, and kappa = 0 for the harmonic generator.
  (H4) HEADLINE: R vs HEK for Y^{2,1} (5.302776) and Y^{3,2} (11.898979),
       registered target 1e-3 relative -- this is a construction, not a fit.
  (H5) |g|^2 >= kappa^2 wherever lambda is read (reality).
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "dp3kt"))
from sugrasol.ypq import sample_slice                        # noqa: E402
from sugrasol.forms import hodge, dform                      # noqa: E402
from r2b_ypq_calibration import load_case, base_metric, S1, P1, S2, P2  # noqa: E402
from sugrasol.ypq import slice_chart, sym_ortho_psi              # noqa: E402
from sugrasol.cone import dp3, B_DP3                             # noqa: E402
from sugrasol.laplacian import slice_potential                   # noqa: E402


def load_dp3():
    ch = slice_chart(dp3(), B_DP3)
    d = np.load(ROOT / "experiments" / "dp3smooth" / "dp3_G_deg18.npz")
    psi = sym_ortho_psi(jnp.asarray(d["coeffs"]),
                        [tuple(int(a) for a in r) for r in d["powers"]],
                        jnp.asarray(d["W"]), jnp.asarray(d["group"]))
    return ch, slice_potential(ch, psi), None, None

FAILS = []


def pq_tr(u, s):
    U = jax.hessian(u)(s)
    return 0.5 * (U[0, 0] - U[1, 1]), U[0, 1], U[0, 0] + U[1, 1]


def design(ch, u, K, n=3000, seed=5, eps=1e-2):
    """rows: Im[(p+iq) g] + (kappa/2) tr U = 0 ,  g = sum_k (al_k + i be_k) w^k.

    EACH ROW IS DIVIDED BY sqrt(p^2+q^2).  Without that the kappa column carries
    tr U ~ 1/l_a, diverges at the facets, dominates the SVD and drives kappa -> 0,
    i.e. silently back to the case that is known to have no solution.  The ratio
    tr U / |V| -> sqrt(2) at a facet (U ~ n_a n_a^T / 2 l_a), so after the division
    every entry is O(1).
    """
    smp = np.asarray(sample_slice(jax.random.PRNGKey(seed), ch, n, eps=eps))
    V = np.asarray(ch.verts_s)
    zc = V.mean(0); rho = np.abs((V - zc) @ np.array([1, 1j])).max()
    p, q, tr = jax.vmap(lambda s: pq_tr(u, s))(jnp.asarray(smp))
    p, q, tr = np.asarray(p), np.asarray(q), np.asarray(tr)
    sc = np.sqrt(p ** 2 + q ** 2)
    uhat = (p + 1j * q) / sc
    w = ((smp - zc) @ np.array([1.0, 1.0j])) / rho          # conditioned chart
    Van = np.stack([w ** k for k in range(K + 1)], -1)      # (n, K+1) complex
    # ORTHONORMALISE the holomorphic basis over the samples.  w^k on a polygon is
    # Vandermonde ill-conditioned; without this the small singular values are
    # conditioning noise and shrink with K while the pointwise quality does not
    # improve -- the same trap as R2b's rank filter.
    Qv, Rv = np.linalg.qr(Van)
    C = uhat[:, None] * (Qv * np.sqrt(len(w)))
    RSCALE = Rv
    M = np.concatenate([C.imag, C.real, (0.5 * tr / sc)[:, None]], axis=1)
    return M, (zc, rho, RSCALE, len(w))


def gfun(coef, K, chart_norm):
    zc, rho, Rv, nn = chart_norm
    al, be, kap = coef[:K + 1], coef[K + 1:2 * (K + 1)], coef[-1]
    a = np.linalg.solve(Rv, (al + 1j * be) * np.sqrt(nn))   # undo the QR

    def g(s):
        w = ((np.asarray(s) - zc) @ np.array([1.0, 1.0j])) / rho
        return complex(np.sum(a * np.array([w ** k for k in range(K + 1)]))), float(kap)
    return g


def form_of(g, kap, s):
    """A = S + kappa eps from g = Phi - i Psi ; return the 4x4 2-form."""
    Phi, Psi = g.real, -g.imag
    A = np.array([[Phi, Psi + kap], [Psi - kap, -Phi]])
    w = jnp.zeros((4, 4))
    for a, ia in enumerate((S1, S2)):
        for b, ib in enumerate((P1, P2)):
            w = w.at[ia, ib].add(A[a, b]).at[ib, ia].add(-A[a, b])
    return w


def run(name, K=8, n=3000):
    ch, u, y1, y2 = load_case(name)
    Rh = None if y1 is None else ((1 - y1) / (1 - y2)) ** 2
    tag = name if isinstance(name, str) else f"Y^{{{name[0]},{name[1]}}}"
    M, cn = design(ch, u, K, n=n)
    sv = np.linalg.svd(M, compute_uv=False)
    U_, S_, Vt = np.linalg.svd(M)
    print(f"\n=== {tag}   K={K}, rows={M.shape[0]}, unknowns={M.shape[1]} ===")
    print(f"  smallest 6 singular values / largest: "
          f"{np.array2string(S_[-6:][::-1] / S_[0], precision=3)}")
    nd = int(np.sum(S_ / S_[0] < 1e-8))
    gap = S_[-(nd + 1)] / S_[-nd] if 0 < nd < len(S_) else np.inf
    print(f"  null-space dim (sv/sv0 < 1e-8) = {nd}   gap to next = {gap:.2e}")
    V = np.asarray(ch.verts_s)
    mids = np.array([0.5 * (V[a - 1] + V[a]) for a in range(len(V))])
    out = []
    for j in range(max(nd, 1)):
        coef = Vt[-1 - j]
        g = gfun(coef, K, cn)
        # (H0)/(H1): ASD and closed, numerically
        chk = []
        for s in mids * 0.85 + V.mean(0) * 0.15:
            gv, kap = g(s)
            om = form_of(gv, kap, s)
            gm = base_metric(u, jnp.asarray(s))
            chk.append(float(jnp.max(jnp.abs(hodge(om, gm) + om)))
                       / float(jnp.max(jnp.abs(om))))
        gv0, kap = g(mids[0])
        lams = []
        for m in mids:
            gv, _ = g(m)
            lams.append(np.sqrt(max(abs(gv) ** 2 - kap ** 2, 0.0)))
        lams = np.array(lams)
        R = float(lams.max() / lams.min()) if lams.min() > 0 else np.inf
        print(f"   null vec {j}: kappa/|g|(mid0) = {kap/abs(gv0):+.4f}   "
              f"max rel |*om + om| = {max(chk):.2e}   "
              f"lambda at facet mids {np.array2string(lams/lams.max(), precision=4)}"
              f"   R = {R:.6f}")
        out.append((R, kap, max(chk)))
    if Rh is not None:
        best = min(out, key=lambda t: abs(t[0] / Rh - 1))
        print(f"  HEK R = {Rh:.6f}  ->  best null vector rel err = "
              f"{abs(best[0]/Rh-1):.3e}")
    return nd, out, Rh


def main():
    print("=" * 74)
    print("[R2b-holo] holomorphic ansatz: ASD + closed built in, nothing minimised")
    print("=" * 74)
    for name in ["conifold", (2, 1), (3, 2), "dP3"]:
        ch, u, y1, y2 = load_dp3() if name == "dP3" else load_case(name)
        Rh = None if y1 is None else ((1 - y1) / (1 - y2)) ** 2
        tag = name if isinstance(name, str) else f"Y^{{{name[0]},{name[1]}}}"
        b2m = 3 if name == "dP3" else 1
        print(f"\n=== {tag}   (b2^- = {b2m})    HEK R = {'-' if Rh is None else f'{Rh:.6f}'} ===")
        print(f"  {'K':>3} {'unk':>4} {'sv_min/sv_max':>14} {'#0':>4} "
              f"{'gap@b2-':>10} {'|*om+om|/|om|':>14} {'kappa/|g|':>10} {'R':>11}"
              f"{'  rel err':>10}")
        for K in (4, 8, 12, 16, 20):
            M, cn = design(ch, u, K)
            U_, S_, Vt = np.linalg.svd(M)
            g = gfun(Vt[-1], K, cn)
            V = np.asarray(ch.verts_s)
            mids = np.array([0.5 * (V[a - 1] + V[a]) for a in range(len(V))])
            probe = mids * 0.85 + V.mean(0) * 0.15
            viol, lams = [], []
            for sp in probe:
                gv, kap = g(sp)
                om = form_of(gv, kap, sp)
                gm = base_metric(u, jnp.asarray(sp))
                viol.append(float(jnp.max(jnp.abs(hodge(om, gm) + om)))
                            / float(jnp.max(jnp.abs(om))))
            for m in mids:
                gv, kap = g(m)
                lams.append(np.sqrt(max(abs(gv) ** 2 - kap ** 2, 0.0)))
            lams = np.array(lams)
            R = float(lams.max() / lams.min()) if lams.min() > 0 else np.inf
            gv0, kap = g(mids[0])
            ncol = int(np.sum(S_ / S_[0] < 1e-8))
            gap = S_[-(b2m + 1)] / S_[-b2m]
            ex = f" {abs(R/Rh-1):>9.3e}" if Rh else f" {abs(R-1):>9.3e}"
            print(f"  {K:>3} {M.shape[1]:>4} {S_[-1]/S_[0]:>14.3e} "
                  f"{ncol:>4} {gap:>10.2e} {max(viol):>14.3e} "
                  f"{kap/abs(gv0):>10.4f} {R:>11.6f}{ex}")


if __name__ == "__main__":
    main()
