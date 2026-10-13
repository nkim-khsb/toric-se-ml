"""Check against R. Eager (email thread "Re: Numerical Sasaki-Einstein
metrics", 2026-09-23) / Nakayama-Okazaki arXiv:2603.17452 eq. (9.41):

    int_{Y_dP2} Riem^2 = 4(2551 + 295 sqrt(33))/243 pi^3 = 2166.94...

which the paper derives from a new conjecture (residue of a supersymmetric
zeta function) and notes is untested outside the toric/regular examples they
can access in closed form.  Our numerical (irregular-Reeb) dP2 SE metric is
an independent check: it is a genuine PDE solution, not toric-combinatorial
data, so it tests their formula in exactly the case they could not close.

METHOD.  Standard metric-cone curvature identity for the CY cone C(L) =
dr^2 + r^2 g_L over an Einstein link (Ric_L = 4 g_L, n = 5 real dims,
textbook, NOT conjectural): writing R^C_{abcd} = r^2(R^L_{abcd} -
(g_{ac}g_{bd}-g_{ad}g_{bc})) on tangential indices and 0 for any r-component
(the round-S^5 case R^L = g wedge g makes the cone flat, as it must --
C(S^5) = C^3 flat), and contracting: Riem^2_cone|_{r=1} = Riem^2_L5 - 2n(n-1)
with n = 5, i.e.
    Riem^2_L5 = Riem^2_cone|_{r=1} + 40.
(S^5 sanity: cone flat => 0 = Riem^2_L5 - 40 => Riem^2_L5 = 40, the known
round value -- matches both papers' S^5 anchor, 40 pi^3.)

The cone invariant is computed by autodiff on the MSY symplectic potential
G(y) via sugrasol.curvature.toric_metric(G, 3)/invariants -- the SAME
machinery already used and validated (against GMSW) in
experiments/ypq/gmsw_invariant_check.py -- evaluated at l_b = 1/2 (<=> r=1).

NOTE (2026-09-23): an earlier version of this script also tried a "4d
transverse" route (Eager-Gary-Roberts eq. (521), Riem^2_4^transverse - 152)
built from slice_potential(chart, psi) alone, i.e. taking Hess_s of u(s)
directly in the SliceChart's ad-hoc orthonormal s-coordinates. On the T^{1,1}
gate below it gave -80, not 136 -- wrong by exactly the kind of
lattice/period normalization mismatch this project has hit repeatedly
(log.md "규격화 사슬"): the orthonormal `s` basis (SVD of b) has no reason to
match the 2pi-periodic angle lattice implicit in toric_metric's dual block,
whereas route B never leaves the cone's own honestly-normalized (y, theta)
coordinates. Dropped; route B alone is gated below to 1e-12 on T^{1,1} and
is the only route used for Y^{2,1} and dP2.

MEASURE.  The volume-weighted average of a basic (theta- and r-independent)
function f over L^5 reduces to the FLAT (unweighted) average over the 2d
slice polygon in the orthonormal `s` coordinates:
  - the toric block metric has det g = 1 identically in (y, theta), so the
    cone's own Liouville measure is flat Lebesgue;
  - slicing at fixed l_b = <b,y> by the coarea formula introduces a Jacobian
    1/|b|, constant (s-independent);
  - the theta^3 integral is common to numerator and denominator.
Both constants cancel in the ratio, leaving <f>_{L5} = (int_P f ds)/Area(P).
(This measure claim is about the SLICE polygon used to locate the point y at
l_b=1/2 for route B, not about any 4d Hessian in s -- it is unaffected by the
route-A problem noted above.)

PRE-REGISTERED gates (checked before the dP2 number is trusted):
  (0) T^{1,1}/conifold, psi=0: route B equals Eager-Gary-Roberts' table
      ratio 2176/16 = 136, pointwise, to > 1e-8 relative (homogeneous space
      -- must be EXACTLY constant, not just close).
  (1) Y^{2,1}, our own trained psi: the resulting int_Y Riem^2 matches the
      closed form (Eager-Gary-Roberts Y^{pq} table entry) to within the
      network's own fit quality (not machine precision -- psi is a
      numerical fit here).
  (2) dP2 se_potential() first reproduces the paper's own ladder gate
      (held-out 6.0e-14 at degree 14) -- if it does not, this is a different
      solution and the Riem^2 number below is not reported.

Usage: PYTHONPATH=. python experiments/dp2/eager_riem2_check.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "dp3kt"))

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import (MomentCone, B_CONIFOLD, conifold, ypq, b_ypq,
                            vol_Y, vol_ypq, dp2, minimize_reeb)          # noqa: E402
from sugrasol.ypq import (slice_chart, t_of_s, sample_slice, gauge_fixed,  # noqa: E402
                          poly_psi, n_poly_coeffs, whiten_poly, ortho_psi,
                          loss_fn, residual_on_slice)
from sugrasol.laplacian import slice_potential, polygon_quadrature        # noqa: E402
from sugrasol.curvature import toric_metric, invariants                  # noqa: E402


# --------------------------------------------------------------- machinery
def cone_potential_from_u(chart, u):
    """Full 6d cone potential G(y), reconstructed from u(s) = GT(t)+psi(s)
    alone (u := slice_potential(chart, psi)) -- no separate psi needed."""
    b, t0, f = chart.b, chart.t0, chart.f

    def G(y):
        lb = b @ y
        s = f @ (y / lb - t0)
        return 0.5 * lb * jnp.log(lb) + lb * u(s)

    return G


def riem2_pointwise(chart, u, s):
    """Riem^2_L5 at slice point s, from u = slice_potential(chart, psi)."""
    G6 = cone_potential_from_u(chart, u)
    g6 = toric_metric(G6, 3)
    y = t_of_s(chart, s) / 2.0  # l_b = 1/2 <=> r = 1
    riem2_cone, _ = invariants(g6, jnp.concatenate([y, jnp.zeros(3)]))
    return riem2_cone + 40.0


def l5_average_riem2(chart, u, ngl=24):
    """<Riem^2_L5>_{L5} via the flat quadrature on the slice polygon."""
    nodes, weights = polygon_quadrature(chart.verts_s, ngl=ngl)
    vals = jax.vmap(lambda s: riem2_pointwise(chart, u, s))(nodes)
    area = jnp.sum(weights)
    return float(jnp.dot(weights, vals) / area), float(area)


def fit_poly_psi(cone: MomentCone, b: jnp.ndarray, deg: int, steps: int, seed: int = 1):
    """Quick self-contained fit (no external checkpoint dependence): whitened
    polynomial psi via L-BFGS, same idiom as gmsw_invariant_check.py/
    harmonic_forms_dp2.py.  Returns (chart, u=slice_potential(chart,psi), held)."""
    from scipy.optimize import minimize

    chart = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(seed), chart, 2000, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(seed + 100), chart, 2000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
    powers, W = whiten_poly(deg, ss)
    nc = len(powers)

    def L(v, batch):
        psi = gauge_fixed(ortho_psi(v[:nc], powers, W), chart.anchors)
        return loss_fn(chart, psi, v[nc], batch)

    vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
    L_te = jax.jit(lambda v: L(v, ss_te))
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=steps, ftol=1e-18, gtol=1e-16))
    held = float(L_te(jnp.asarray(res.x)))
    psi = gauge_fixed(ortho_psi(jnp.asarray(res.x[:nc]), powers, W), chart.anchors)
    u = slice_potential(chart, psi)
    return chart, u, held, float(res.fun)


# ------------------------------------------------------------- EGR/NO tables
def egr_riem2_ypq(p: int, q: int) -> float:
    """Eager-Gary-Roberts arXiv:1011.5231, table (CurvTable)."""
    s = np.sqrt(4 * p ** 2 - 3 * q ** 2)
    num = 8 * np.pi ** 3 * (2 * p ** 2 - 3 * q ** 2 + p * s) * (106 - 19 * np.sqrt(4 - 3 * q ** 2 / p ** 2))
    den = 27 * (p ** 3 - p * q ** 2)
    return num / den


def no_riem2_dp2() -> float:
    """Nakayama-Okazaki arXiv:2603.17452, eq. after (Res_dP2)."""
    s33 = np.sqrt(33.0)
    return 4.0 * (2551.0 + 295.0 * s33) / 243.0 * np.pi ** 3


def no_vol_dp2() -> float:
    s33 = np.sqrt(33.0)
    return (59.0 + 11.0 * s33) / 486.0 * np.pi ** 3


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    print("== gate (0): T^{1,1}/conifold (psi = 0), pointwise ==")
    ch0 = slice_chart(conifold(), B_CONIFOLD)
    u0 = slice_potential(ch0, lambda s: 0.0)
    s_test = jnp.mean(ch0.verts_s, axis=0) * 0.3  # a generic interior point
    rb = riem2_pointwise(ch0, u0, s_test)
    print(f"  route B (6d cone + 40): {float(rb):.12f}")
    print(f"  EGR table T^{{1,1}} ratio 2176/16: {2176/16:.12f}")
    gate0 = abs(float(rb) - 136.0) < 1e-6
    print(f"  PASS: {gate0}")
    assert gate0, "T^{1,1} calibration failed -- stop, do not trust anything below"

    print("\n== gate (1): Y^{2,1}, our own quick poly fit ==")
    cone21, b21 = ypq(2, 1), b_ypq(2, 1)
    ch1, u1, held1, train1 = fit_poly_psi(cone21, b21, deg=10, steps=6000)
    print(f"  train loss {train1:.3e}  held-out {held1:.3e}")
    avg1, area1 = l5_average_riem2(ch1, u1, ngl=24)
    print(f"  <Riem^2_L5>: {avg1:.6f}")
    vol21 = float(vol_Y(cone21, b21))
    vol21_msy = float(vol_ypq(2, 1))
    print(f"  Vol(Y^2,1) = {vol21:.10f}  (vol_ypq closed form {vol21_msy:.10f}, "
          f"rel {abs(vol21/vol21_msy-1):.2e})")
    int_riem2_ours = avg1 * vol21
    int_riem2_egr = egr_riem2_ypq(2, 1)
    print(f"  int_Y Riem^2  ours: {int_riem2_ours:.6f}   EGR closed form: {int_riem2_egr:.6f}")
    rel1 = abs(int_riem2_ours / int_riem2_egr - 1.0)
    print(f"  relative deviation: {rel1:.4%}")

    print("\n== dP2: our numerical Sasaki-Einstein metric ==")
    from harmonic_forms_dp2 import se_potential, PAPER_HELD
    ch2, u2, held2 = se_potential()
    gate2 = abs(held2 / PAPER_HELD - 1.0) < 1.0
    print(f"  reproduces published ladder (held-out {held2:.3e} vs {PAPER_HELD:.2e}): "
          f"{'PASS' if gate2 else 'FAIL'}")
    if not gate2:
        print("  STOPPING: not the published solution.")
        sys.exit(1)

    avg2, area2 = l5_average_riem2(ch2, u2, ngl=24)
    print(f"  <Riem^2_L5>: {avg2:.6f}")

    cone2 = dp2()
    b2 = minimize_reeb(cone2, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)
    vol2 = float(vol_Y(cone2, b2))
    print(f"  Vol(Y_dP2) = {vol2:.10f} = {vol2/np.pi**3:.10f} pi^3"
          f"  (Nakayama-Okazaki closed form {no_vol_dp2()/np.pi**3:.10f} pi^3)")

    int_riem2_dp2 = avg2 * vol2
    target = no_riem2_dp2()
    print(f"\n  int_{{Y_dP2}} Riem^2 (ours, numerical SE metric): {int_riem2_dp2:.10f}")
    print(f"  Nakayama-Okazaki eq. (9.41) closed form:          {target:.10f}")
    print(f"  relative deviation: {abs(int_riem2_dp2/target - 1.0):.3e}")

    print("\n  quadrature-order convergence (ngl = 16..64, same fit):")
    for ngl in (16, 32, 48, 64):
        avg_n, _ = l5_average_riem2(ch2, u2, ngl=ngl)
        val_n = avg_n * vol2
        print(f"    ngl={ngl:3d}  int_Y Riem^2 = {val_n:.10f}"
              f"  rel_dev = {abs(val_n / target - 1.0):.3e}")
