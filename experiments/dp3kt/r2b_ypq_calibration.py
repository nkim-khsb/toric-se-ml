"""[R2b-cal] Calibrate the R2b polynomial basis against HEK's closed-form F(y).

QUESTION (raised 2026-08-19): R2b's harmonic (1,1) forms on dP3 stall at ~1%.
Is that because the true form is singular at the hexagon boundary, or because a
global polynomial basis cannot reach a function whose poles sit just OUTSIDE the
polytope?  Y^{p,q} settles it: there the answer is known in closed form,
    omega = F(y) (e^th^e^ph - e^y^e^be),   F(y) = 1/(1-y)^2      [HEK 0412193:590]
and the pole y=1 lies outside the moment interval [y1,y2], approaching the
boundary as q -> p (GMSW: y2 -> 1).

EXACT POINTWISE CHARACTERISATION (derived + machine-verified against the same
sugrasol.forms.hodge that R2b uses; see scratch derive.py):
  for omega = A_ij ds_i ^ dphi_j  with g = blockdiag(U, U^{-1}), U = Hess u,
      omega closed  <=>  A = Hess h
      omega ASD     <=>  tr A = 0  AND  [A, U] = 0      (primitive + type (1,1))
  => eigenvalues of A are +-lambda(s), and HEK says lambda \propto F(y).
  Two pointwise conditions on one function h: overdetermined, hence the finite-
  dimensional solution space (1 for a quadrilateral, 3 for the hexagon).

SCALE-FREE OBSERVABLE (no coordinate bridge to y needed, no normalisation):
      R := lambda_max / lambda_min  ==  ((1-y1)/(1-y2))^2
Both sides are pure numbers; the left from our polynomial solve, the right from
GMSW's closed-form roots.  Also predicted: lambda depends on y ONLY (one affine
direction), so 1/sqrt(lambda) is an AFFINE function of the moment coordinates.

PRE-REGISTERED GATES (fixed before looking at any output) AND OUTCOME:
  (Y0) free anchor, conifold: base = P1xP1, psi=0 exact; the exact answer
       (A = const, traceless, commuting with U) lies in the degree-2 span, so
       R = 1 must come out exactly.  PASS, 8.4e-15, at EVERY degree up to Dh=18.
       This validates machinery + conventions; failures below are not the setup.
  (Y1) kernel dim = 1 on a quadrilateral.  ***FAILS, AND THE GATE ITSELF IS
       WRONG.***  With conditioning fixed, Y^{3,2} shows THREE near-ASD forms
       (6.2e-5, 2.2e-3, 2.7e-3 at Dh=18, next 4.9e-2) where topology allows one,
       and the count grows with expressivity.  Reason: "exact cap ASD = 0" is a
       STOKES argument; an unconstrained interior fit has no Stokes, so the local
       near-ASD space is infinite-dimensional.  R2b's own printout already says
       "0 eigenvalues < 1e-3 (expect 3)" -- its 3 is read off a factor-3 gap, and
       at Dh=6 that happens to land on 3 by coincidence (at Dh=18 it is ambiguous).
       => THE GAP MEASURES THE BASIS, NOT THE TOPOLOGY.
  (Y2) HEADLINE: rel error of R vs HEK, registered band [1e-3, 3e-1].
       FAILS, and not by being too large in a stable way: R is non-monotone in
       degree (Y^{3,2}: 0.65, 0.054, 2.04, 9.33, 0.84, 1.81, 0.56) while the SD
       floor falls cleanly by 3 decades.  A falling residual with a wandering
       observable is the signature of (Y1): the minimiser is an arbitrary member
       of a degenerate near-kernel.  ==> the polynomial machinery does NOT
       reproduce F(y), and the obstruction is SELECTION, not approximation.
  (Y3) pole proximity ordering.  Not interpretable once (Y1) failed.
  (Y4) 1/sqrt(lambda) affine, registered > 1e-3.  Passes numerically (0.15-0.45)
       but for the wrong reason: it never IMPROVES with degree, so it too is
       reporting the selection failure, not approximation error.
  (Y5) RETRACTED AS AN ESTIMATOR: the level-direction tilt divides by |grad|,
       which is 0 exactly in the conifold case it was supposed to anchor on
       (reported 81 deg for an exactly constant function).  Removed.
Reported for context (ground-truth-free): pointwise violation of tr A = 0 and of
[A,U] = 0 -- the intrinsic error of the polynomial answer.

REFUTED HYPOTHESIS (recorded so it is not retried).  The motivating guess was
that the failure is polynomial approximation of a function whose poles sit just
outside the polytope, fixable with rational factors / a facet-adapted basis.  Two
measurements kill it: (a) the SD floor converges cleanly once conditioning is
fixed, so approximation is not the binding constraint; (b) the facet boundary
condition derived from Guillemin,
      A|_{l_a = 0}  \propto  T_a := n_a n_a^T - (1/2)|n_a|^2 I ,  n_a = grad_s l_a,
has rank 25 of 42 generators at Dh=6 -- a real constraint -- yet the SD minimiser
already satisfies it to 1e-29 at zero penalty weight, because bulk ASD near a
facet where U ~ 1/l_a implies it.  A facet-adapted basis is NOT the missing piece.
WHAT IS MISSING is a selection principle: fix the cohomology CLASS, e.g. by
prescribing the divisor periods, which [P1] (r2b_periods.py) computes exactly as
differences of vertex values of the potential, then minimise SD energy.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_ypq_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import eigh, qr

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sugrasol.cone import b_ypq, ypq, conifold, B_CONIFOLD      # noqa: E402
from sugrasol.gmsw import roots_and_a                           # noqa: E402
from sugrasol.laplacian import slice_potential                  # noqa: E402
from sugrasol.ypq import (slice_chart, sample_slice, whiten_poly,  # noqa: E402
                          ortho_psi, gauge_fixed)
from sugrasol.forms import hodge                                # noqa: E402

S1, P1, S2, P2 = 0, 1, 2, 3
CKPT = {(2, 1): 8, (3, 2): 12}          # persisted deg of the ortho psi solve
FAILS = []


def _form(pairs):
    w = jnp.zeros((4, 4))
    for (i, j, c) in pairs:
        w = w.at[i, j].add(c).at[j, i].add(-c)
    return w


def omega_hess(hfn, s):
    """A = Hess h in the ds^dphi block -- R2b's generator, verbatim."""
    H = jax.hessian(hfn)(s)
    return _form([(S1, P1, H[0, 0]), (S1, P2, H[0, 1]),
                  (S2, P1, H[1, 0]), (S2, P2, H[1, 1])])


def build_generators(Dh, Dp):
    """Identical span to r2b_harmonic_forms.build_generators."""
    gens = []
    for tot in range(2, Dh + 1):
        for a in range(tot + 1):
            gens.append(("h", lambda s, a=a, b=tot - a: omega_hess(
                lambda z: z[0] ** a * z[1] ** b, s)))
    for tot in range(0, Dp + 1):
        for a in range(tot + 1):
            gens.append(("P", lambda s, a=a, b=tot - a: (s[0] ** a * s[1] ** b)
                         * _form([(S1, S2, 1.0)])))
    gens.append(("antisym", lambda s: _form([(S1, P2, 1.0), (S2, P1, -1.0)])))
    gens.append(("R", lambda s: _form([(P1, P2, 1.0)])))
    return gens


def load_case(name):
    """-> (chart, u, y1, y2) ; u = transverse symplectic potential."""
    if name == "conifold":
        ch = slice_chart(conifold(), B_CONIFOLD)
        return ch, slice_potential(ch, lambda s: 0.0), None, None
    p, q = name
    deg = CKPT[(p, q)]
    v = np.load(ROOT / "experiments" / "ypq" / f"ckpt_y{p}{q}_deg{deg}_ortho.npy")
    ch = slice_chart(ypq(p, q), b_ypq(p, q))
    ss = sample_slice(jax.random.PRNGKey(1), ch, 4096, eps=2e-3)   # same as train
    powers, W = whiten_poly(deg, ss)
    psi = gauge_fixed(ortho_psi(jnp.asarray(v[:len(powers)]), powers, W),
                      ch.anchors)
    y1, y2, _ = roots_and_a(p, q)
    return ch, slice_potential(ch, psi), float(y1), float(y2)


def base_metric(u, s):
    U = jax.hessian(u)(s)
    Ui = jnp.linalg.inv(U)
    g = jnp.zeros((4, 4))
    for a, ia in enumerate((S1, S2)):
        for b, ib in enumerate((S1, S2)):
            g = g.at[ia, ib].set(U[a, b])
    for a, ia in enumerate((P1, P2)):
        for b, ib in enumerate((P1, P2)):
            g = g.at[ia, ib].set(Ui[a, b])
    return g


UT = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def solve_asd(ch, u, Dh, Dp, n=4000, seed=5, eps=2e-2, tol=1e-12):
    """R2b's variational problem, but with the rank filter FIXED.

    R2b used  keep = wB > 1e-10 * wB.max()  on the raw-monomial form Gram.  That
    silently truncates: at Dh=14 it retains 95 of 210 generators, so the SD floor
    looks like a wall.  Here the forms are L^2-orthonormalised (Cholesky of g^{-1}
    per sample -> plain Euclidean inner product) and the rank is taken by a
    rank-revealing PIVOTED QR -- the remedy CLAUDE.md prescribes and this is its
    first implementation.  Restores full rank and drops the floor by 3 decades.
    """
    gens = build_generators(Dh, Dp)
    smp = jnp.asarray(np.asarray(sample_slice(jax.random.PRNGKey(seed), ch,
                                              n, eps=eps)))
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    oms = jnp.stack([jax.vmap(g)(smp) for (_, g) in gens], axis=1)
    stars = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(oms, gs)
    L = jnp.linalg.cholesky(jnp.linalg.inv(gs))          # g^{-1} = L L^T

    def flat(O):
        """<X,X> = (1/n) sum_n sum_{i<j} Xt_ij^2 with Xt = L^T X L."""
        Ot = jnp.einsum("nai,nkab,nbj->nkij", L, O, L)
        v = jnp.stack([Ot[:, :, i, j] for (i, j) in UT], -1)
        return np.asarray(jnp.transpose(v, (0, 2, 1)).reshape(-1, v.shape[1])
                          ) / np.sqrt(n)

    X, Xsd = flat(oms), flat(0.5 * (oms + stars))
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    dg = np.abs(np.diag(R))
    r = int(np.sum(dg > tol * dg[0]))
    T = np.zeros((len(gens), r)); T[piv[:r], :] = np.linalg.inv(R[:r, :r])
    Z = Xsd @ T
    A = 0.5 * (Z.T @ Z + (Z.T @ Z).T)
    spec, C = eigh(A)
    om = jnp.einsum("nkij,k->nij", oms, jnp.asarray(T @ C[:, 0]))
    return smp, gs, om, spec, r


def observables(smp, gs, om, u):
    """lambda(s) = |eigenvalues| of the ds^dphi block A, plus exactness probes."""
    A = jnp.stack([jnp.stack([om[:, S1, P1], om[:, S1, P2]], -1),
                   jnp.stack([om[:, S2, P1], om[:, S2, P2]], -1)], -2)  # (n,2,2)
    U = jax.vmap(lambda s: jax.hessian(u)(s))(smp)
    ev = jnp.linalg.eigvalsh(0.5 * (A + jnp.swapaxes(A, -1, -2)))
    lam = 0.5 * (jnp.abs(ev[:, 0]) + jnp.abs(ev[:, 1]))
    nrm = jnp.linalg.norm(A, axis=(1, 2))
    tr_v = float(jnp.mean(jnp.abs(jnp.trace(A, axis1=1, axis2=2)) / nrm))
    com = A @ U - U @ A
    com_v = float(jnp.mean(jnp.linalg.norm(com, axis=(1, 2))
                           / (nrm * jnp.linalg.norm(U, axis=(1, 2)))))
    return np.asarray(lam), tr_v, com_v


def affine_probe(smp, lam, verts):
    """1/sqrt(lambda) should be AFFINE (\propto 1-y).  Returns (relres, spread_deg,
    R_from_vertices)."""
    s = np.asarray(smp); ell = 1.0 / np.sqrt(lam)
    M = np.column_stack([np.ones(len(s)), s[:, 0], s[:, 1]])
    c, *_ = np.linalg.lstsq(M, ell, rcond=None)
    res = ell - M @ c
    relres = float(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(ell ** 2)))
    # direction spread of grad(ell): finite-difference-free, use local lstsq-free
    # test -- exact affine => grad constant => use residual of a 1d reduction
    g = c[1:]; ghat = g / np.linalg.norm(g)
    perp = np.array([-ghat[1], ghat[0]])
    # spread: how much ell varies along the level direction, in degrees of tilt
    tp = (s - s.mean(0)) @ perp
    slope_perp, _ = np.polyfit(tp, ell, 1)
    spread = float(np.degrees(np.arctan2(abs(slope_perp), np.linalg.norm(g))))
    ev = np.column_stack([np.ones(len(verts)), verts[:, 0], verts[:, 1]]) @ c
    R = float((ev.max() / ev.min()) ** 2)
    return relres, spread, R


def run(name, Dh=6, Dp=4, n=4000, quiet=False):
    ch, u, y1, y2 = load_case(name)
    smp, gs, om, spec, ndof = solve_asd(ch, u, Dh, Dp, n=n)
    lam, tr_v, com_v = observables(smp, gs, om, u)
    verts = np.asarray(ch.verts_s)
    relres, spread, R_vert = affine_probe(smp, lam, verts)
    R_smp = float(lam.max() / lam.min())
    R_hek = None if y1 is None else ((1 - y1) / (1 - y2)) ** 2
    if not quiet:
        tag = "conifold" if name == "conifold" else f"Y^{{{name[0]},{name[1]}}}"
        print(f"\n--- {tag}   (Dh={Dh}, Dp={Dp}, n={n}) ---")
        print(f"  closed span dim {ndof};  SD-energy spectrum "
              f"{np.array2string(spec[:4], precision=3e0 and 4)}")
        print(f"  kernel dim (lam<1e-3) = {int(np.sum(spec < 1e-3))}   "
              f"gap lam1/lam0 = {spec[1]/max(spec[0],1e-300):.2e}")
        print(f"  pointwise exactness (ground-truth-free): "
              f"|tr A|/|A| = {tr_v:.3e}   ||[A,U]||/(|A||U|) = {com_v:.3e}")
        print(f"  1/sqrt(lam) affine: rel rms residual = {relres:.3e}"
              f"   [Y5 tilt estimator retracted: divides by |grad|]")
        print(f"  R = lam_max/lam_min : samples {R_smp:.6f}   "
              f"affine-extrapolated to facets {R_vert:.6f}")
        if R_hek is not None:
            print(f"  HEK closed form  ((1-y1)/(1-y2))^2 = {R_hek:.6f}   "
                  f"[y1={y1:+.6f}, y2={y2:+.6f}, pole gap 1-y2={1-y2:.6f}]")
            print(f"  => rel error: samples {abs(R_smp/R_hek-1):.3e}   "
                  f"extrapolated {abs(R_vert/R_hek-1):.3e}")
    return dict(spec=spec, ndof=ndof, R_smp=R_smp, R_vert=R_vert, R_hek=R_hek,
                relres=relres, spread=spread, tr=tr_v, com=com_v)


def main():
    print("=" * 74)
    print("[R2b-cal] polynomial basis vs HEK F(y)=1/(1-y)^2 on Y^{p,q}")
    print("=" * 74)

    # ---- (Y0) free anchor: the exact answer IS in the degree-2 span ----------
    c = run("conifold")
    err0 = abs(c["R_smp"] - 1.0)
    ok = err0 < 1e-8
    print(f"\n  (Y0) conifold R=1 exactly?  |R-1| = {err0:.2e}  (< 1e-8) "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append("Y0 conifold anchor")
    if c["ndof"] and int(np.sum(c["spec"] < 1e-3)) != 1:
        FAILS.append("Y0 conifold kernel dim != 1")

    # ---- Y^{p,q} -------------------------------------------------------------
    res = {}
    for key in [(2, 1), (3, 2)]:
        res[key] = run(key)
        r = res[key]
        if int(np.sum(r["spec"] < 1e-3)) != 1 or r["spec"][1] / max(r["spec"][0], 1e-300) < 10:
            FAILS.append(f"Y1 kernel/gap {key}")
        e = abs(r["R_vert"] / r["R_hek"] - 1)
        if not (1e-3 <= e <= 3e-1):
            FAILS.append(f"Y2 headline out of registered band {key}: {e:.2e}")
        if not r["relres"] > 1e-3:
            FAILS.append(f"Y4 affine residual too small {key}: {r['relres']:.2e}")
    
    e21 = abs(res[(2, 1)]["R_vert"] / res[(2, 1)]["R_hek"] - 1)
    e32 = abs(res[(3, 2)]["R_vert"] / res[(3, 2)]["R_hek"] - 1)
    print(f"\n  (Y3) pole proximity: err Y^{{3,2}} ({e32:.3e}) > "
          f"err Y^{{2,1}} ({e21:.3e}) ?  {'PASS' if e32 > e21 else 'FAIL'}")
    if not e32 > e21:
        FAILS.append("Y3 pole-proximity ordering")

    # ---- (Y6) degree ladder --------------------------------------------------
    print("\n  (Y6) degree ladder on Y^{3,2}  (Dp = Dh - 2):")
    print(f"    {'Dh':>4} {'span':>5} {'|trA|/|A|':>11} {'|[A,U]|':>11} "
          f"{'affine res':>11} {'R':>10} {'rel err':>10}")
    lad = []
    for Dh in (4, 6, 8, 10):
        r = run((3, 2), Dh=Dh, Dp=Dh - 2, quiet=True)
        e = abs(r["R_vert"] / r["R_hek"] - 1)
        lad.append(e)
        print(f"    {Dh:>4} {r['ndof']:>5} {r['tr']:>11.3e} {r['com']:>11.3e} "
              f"{r['relres']:>11.3e} {r['R_vert']:>10.4f} {e:>10.3e}")
    print(f"    HEK target R = {res[(3,2)]['R_hek']:.4f}     "
          f"ladder err ratio first/last = {lad[0]/lad[-1]:.2f}x")

    print("\n" + "=" * 74)
    print("FAILURES: " + ("none" if not FAILS else "\n  - " + "\n  - ".join(FAILS)))
    print("=" * 74)


if __name__ == "__main__":
    main()
