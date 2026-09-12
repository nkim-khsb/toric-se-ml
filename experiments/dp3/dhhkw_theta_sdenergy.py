"""DHHKW's published (1,1)-form, graded by OUR grader on OUR metric.

Section 6.3 of the paper declines the comparison: "the comparison to draw is not
between those numbers and ours, which grade different quantities".  It can be
drawn, and this draws it.  Their theta_a and our chi-forms are the same kind of
object, so putting theirs through the same self-dual energy (6.x) on the same
metric with the same quadrature makes the two numbers commensurable.

The dictionary.  For a torus-invariant f, in complex coordinates w = u + i vt
with u = grad G, one has i d dbar f = (1/2) f_{,u_i u_j} du_i ^ dvt_j, hence in
the symplectic frame theta = A_ij ds_i ^ dphi_j with

    A = (1/2) U d_x( U^{-1} grad f ) U^{-1} ,      U = Hess G_P .

CHECK, and it is what says the dictionary is right: tr A = (1/2) d_i(u^{ij} d_j f),
which is exactly DHHKW's own harmonicity operator (3.46).  The derivation
reproduces their condition rather than assuming it.

WHY NOT A PROJECTION, and why theta_2 alone is the wrong object.  Their
harmonicity is d_i(u^{ij} d_j mu_a) = const and primitivity is that constant
vanishing (3.48); for theta_2 the constant is 2/3, not 0 (their 6.15).  So
theta_2 is a harmonic (1,1)-form but NOT primitive, and projecting it onto our
primitive solution space would measure nothing.  What is primitive is a
combination.  Nothing has to be imposed to find it: on a Kahler surface the
self-dual part of a (1,1)-form IS its J-component, so the grader already
penalises non-primitivity, and diagonalising it on the span of the six theta_a
separates the primitive directions from J by itself.

PRE-REGISTERED (fixed before running):
  (R) the span of the six theta_a has rank 4, not 6: b_2(dP3) = 4 and the six
      toric divisors carry two relations.
  (C) of those 4, exactly 3 have small energy and 1 is O(1) -- the three
      primitive classes and the Kahler direction.
  (M) the three small ones are the number to compare with ours, which at degree
      16 are 6.2e-11, 8.1e-11, 3.2e-10.  Their fit is sixth order with fifteen
      coefficients and they report ~1e-3 pointwise on the constancy condition,
      so we expect their three to land far above ours; how far is the answer.

Usage: PYTHONPATH=. python experiments/dp3/dhhkw_theta_sdenergy.py
"""
import sys
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
from dhhkw_theta_compare import GROUP, mu_dhhkw                      # noqa: E402

# The admissible (Wachspress) space of the draft, degree 16, 69 generators,
# deterministic quadrature: r2b_stream_wachspress.log, stable from 16 nodes
# per direction to 128.  The earlier triple here, 6.2e-11 / 8.1e-11 / 3.2e-10,
# was the raw stream-potential space under Monte-Carlo integration with a
# boundary margin, which section 6.2 of the draft retracts: it understates the
# energy by excluding the region where a polynomial is worst.
OURS = (1.17e-9, 1.17e-9, 5.56e-9)


def A_of(f, s, u):
    """A = (1/2) U d_x(U^{-1} grad f) U^{-1}."""
    U = jax.hessian(u)(s)
    Ui = jnp.linalg.inv(U)
    flow = lambda z: jnp.linalg.inv(jax.hessian(u)(z)) @ jax.grad(f)(z)
    return 0.5 * U @ jax.jacobian(flow)(s) @ Ui


def wrap(A):
    w = jnp.zeros((4, 4))
    for a, ia in enumerate((S1, S2)):
        for b, ib in enumerate((P1, P2)):
            w = w.at[ia, ib].add(A[a, b]).at[ib, ia].add(-A[a, b])
    return w


def main():
    ch, u, _, _ = load_dp3()
    smp, wt = poly_quad(ch, 48, 48)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)

    # the six theta_a, reached as the D6 orbit of their published mu_2
    mus = [(lambda s, g=g: mu_dhhkw(g @ s)) for g in np.asarray(GROUP)]
    As = jnp.stack([jax.vmap(lambda s, f=f: A_of(f, s, u))(smp) for f in mus], 1)
    oms = jax.vmap(lambda AA: jax.vmap(wrap)(AA))(jnp.transpose(As, (1, 0, 2, 3)))
    oms = jnp.transpose(oms, (1, 0, 2, 3))
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(oms, gs)

    tr = float(jnp.max(jnp.abs(jnp.trace(As, axis1=2, axis2=3))))
    print(f"  max |tr A| over the orbit and the nodes: {tr:.3e}")
    print("  (nonzero by construction: their theta_a are harmonic but not "
          "primitive)\n", flush=True)

    L = jnp.linalg.cholesky(jnp.linalg.inv(gs))

    def flat(O):
        Ot = jnp.einsum("nai,nkab,nbj->nkij", L, O, L)
        v = jnp.stack([Ot[:, :, i, j] for (i, j) in UT], -1)
        M = np.asarray(jnp.transpose(v, (0, 2, 1)).reshape(-1, v.shape[1]))
        return M * np.repeat(np.sqrt(wt), 6)[:, None]

    X, Xs = flat(oms), flat(0.5 * (oms + st))
    Q, R, piv = qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    # Read the gap, not a tolerance.  The two relations among the six toric
    # divisors are exact for exact theta_a and only approximate for a fit, so a
    # 1e-12 cut counts twelve directions and measures the tolerance rather than
    # the geometry.  The drop below locates the rank on its own.
    ratios = d / d[0]
    gaps = ratios[:-1] / ratios[1:]
    rank = int(np.argmax(gaps) + 1)
    print(f"  (R) rank from the largest drop in the span: {rank}   "
          f"[expected 4 = b_2(dP3)]")
    print(f"      relative diagonal: " +
          "  ".join(f"{x:.1e}" for x in ratios[:6]))
    print(f"      the relations among the six theta_a therefore hold to "
          f"{ratios[rank]:.1e}, a measure of their fit", flush=True)

    T = np.zeros((X.shape[1], rank))
    T[piv[:rank], :] = np.linalg.inv(R[:rank, :rank])
    Z = Xs @ T
    lam = np.linalg.eigvalsh(0.5 * (Z.T @ Z + (Z.T @ Z).T))
    print(f"\n  self-dual energies on the span, ascending:")
    print("      " + "  ".join(f"{v:.3e}" for v in lam))
    small = lam[:3]
    print(f"\n  (C) three small and one O(1) on the rank-{rank} span? "
          f"{'yes' if len(lam) >= 4 and lam[3] > 100 * lam[2] else 'NO'}")
    print(f"  (M) theirs {'  '.join(f'{v:.2e}' for v in small)}")
    print(f"      ours   {'  '.join(f'{v:.2e}' for v in OURS)}")
    print(f"      ratio  {'  '.join(f'{a/b:.1e}' for a, b in zip(small, OURS))}")


if __name__ == "__main__":
    main()
