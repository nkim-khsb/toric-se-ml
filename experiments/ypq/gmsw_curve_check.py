"""What the blue curve of the Y^{p,q} invariant figure actually is.

`gmsw_invariant_check.py` draws the GMSW reference by evaluating (Phi1,Phi2)
along y in (y1,y2) with the remaining four coordinates pinned at arbitrary
values.  That is legitimate only if the invariants do not depend on those four
-- the cohomogeneity-one statement -- and the deviation it reports by
interpolating Phi2(Phi1) is well defined only if Phi1 is monotone in y.  Both
are assumptions of the figure; this script measures them.  Read-only.

Usage: PYTHONPATH=. python3 experiments/ypq/gmsw_curve_check.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from sugrasol.curvature import invariants
from sugrasol.gmsw import cone6, roots_and_a

jax.config.update("jax_enable_x64", True)


def curve_facts(p: int, q: int, n: int = 200, draws: int = 12, seed: int = 0):
    """Returns (independence, r_exponents, phi1_monotone, phi1_range)."""
    y1, y2, a = roots_and_a(p, q)
    g6 = cone6(float(a))
    f = jax.jit(lambda x: jnp.stack(invariants(g6, x)))

    ymid = 0.5 * (float(y1) + float(y2))
    rng = np.random.default_rng(seed)
    vals = np.array([f(jnp.array([1.0, ymid, *rng.uniform(0.2, 2.8, 4)]))
                     for _ in range(draws)])
    independence = np.ptp(vals, axis=0) / np.abs(vals.mean(axis=0))

    lo, hi = [np.asarray(f(jnp.array([r, ymid, 0.9, 0.3, 0.7, 0.2]))) for r in (1.0, 2.0)]
    r_exponents = np.log(hi / lo) / np.log(2.0)          # cone scaling, -4 and -6

    ys = np.linspace(float(y1) + 1e-4, float(y2) - 1e-4, n)
    cur = np.array([f(jnp.array([1.0, y, 0.9, 0.3, 0.7, 0.2])) for y in ys])
    d = np.diff(cur[:, 0])
    return independence, r_exponents, bool(np.all(d > 0) or np.all(d < 0)), \
        (cur[:, 0].min(), cur[:, 0].max())


def locus_thickness(inv, nbins: int = 20, nmin: int = 20):
    """How far the (Phi1,Phi2) image is from being a curve.

    Inside each Phi1 bin the curve's own slope is removed by a local linear fit
    in log-log; what is left is the thickness.  Without that detrending the
    estimator returns the slope instead, ~40% on this target, and reports a
    thick locus where there is none.
    """
    x, y = inv[:, 0], inv[:, 1]
    lx, ly = np.log10(x), np.log10(y)
    edges = np.quantile(lx, np.linspace(0, 1, nbins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lx >= lo) & (lx < hi)
        if m.sum() >= nmin:
            c = np.polyfit(lx[m], ly[m], 1)
            out.append(np.ptp(10 ** (ly[m] - np.polyval(c, lx[m])) - 1.0))
    return np.median(out), np.max(out)


if __name__ == "__main__":
    for p, q in [(2, 1), (3, 2)]:
        ind, pw, mono, rng_ = curve_facts(p, q)
        print(f"Y^{{{p},{q}}}: independence of (theta,phi,psi,alpha) "
              f"Phi1 {ind[0]:.1e} Phi2 {ind[1]:.1e} | r-exponents "
              f"{pw[0]:+.6f} {pw[1]:+.6f} | Phi1 monotone {mono} | "
              f"Phi1 in [{rng_[0]:.3f}, {rng_[1]:.3f}]")
