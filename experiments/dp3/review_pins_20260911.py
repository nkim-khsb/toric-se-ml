"""Two cheap pins for the 2026-09-11 review.

[9]  Volume Hessian at b* on dP3 vs the D6-invariant quadratic form.  D6 acts on the
     Reeb displacement (delta b_2, delta b_3) through its two-dimensional irrep, on
     which there is exactly one invariant quadratic form up to scale.  Both the
     volume excess and the exact residual floor are D6-invariant functions of the
     displacement, so at second order they MUST be proportional; the paper's
     "as quadratic forms they are not proportional" is wrong at leading order and
     the grid spread 1.1--3.8 is a finite-displacement effect.  Check that
     Hess vol(b*) is proportional to the invariant form.

[4]  The (1,1)-form dictionary.  In our normalization theta = i d dbar mu has
     A = (1/2) U d(U^{-1} grad mu) U^{-1}, tr A = (1/2) L[mu], L = d_i(u^{ij} d_j .),
     so the potential of omega_T = sum ds_i ^ nu_i (A = I, tr A = 2) obeys
     L[mu] = 4; it is mu_T = 2 (s . grad G_P - G_P), the Legendre dual.  DHHKW's
     omega-potential mu^D = (1/4) sum_g mu_2 o g satisfies -L[mu^D]/3 = 2, i.e.
     L[mu^D] = -6.  The two are the same form in two normalizations:
     mu^D = -(3/2) mu_T + affine.  The 3 is the chart x = 3 s, the -1/2 is their
     theta_{i jbar} = d_i d_jbar mu against our i d dbar mu.  Verify by fitting the
     difference to an affine function.

Usage: PYTHONPATH=. python experiments/dp3/review_pins_20260911.py
"""
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import B_DP3, dp3, vol_Y                          # noqa: E402
from sugrasol.artifacts import load_psi_dp3                          # noqa: E402
from sugrasol.laplacian import slice_potential                       # noqa: E402
from sugrasol.ypq import sample_slice                                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def pin9():
    cone = dp3()
    w = np.asarray(cone.normals)[:, 1:]                       # fan rays (six)
    rays = {tuple(r) for r in w.astype(int)}
    G = []
    for ents in itertools.product((-1, 0, 1), repeat=4):
        M = np.array(ents).reshape(2, 2)
        if abs(round(np.linalg.det(M))) != 1:
            continue
        if {tuple((M @ r).astype(int)) for r in w.astype(int)} == rays:
            G.append(M)
    print(f"[9] lattice automorphisms of the dP3 fan: {len(G)} (expect 12)")
    Q = sum(M.T @ M for M in G) / len(G)                        # invariant form on (b2,b3)
    H = np.asarray(jax.hessian(lambda bb: vol_Y(cone, jnp.array([3.0, bb[0], bb[1]])))(jnp.zeros(2)))
    V0 = float(vol_Y(cone, B_DP3))
    ratio = np.linalg.eigvals(H @ np.linalg.inv(Q))
    print(f"    Hess vol(b*)/vol = {np.round(H / V0, 6).tolist()}")
    print(f"    invariant form Q = {np.round(Q, 6).tolist()}")
    print(f"    eigenvalues of H Q^-1: {np.round(ratio.real, 8).tolist()}  "
          f"(proportional iff equal: rel spread {abs(ratio[0]-ratio[1])/abs(ratio).mean():.1e})")
    # invariance of the volume itself under the group, for good measure
    b = jnp.array([0.2, -0.13])
    vals = [float(vol_Y(cone, jnp.concatenate([jnp.array([3.0]), jnp.asarray(M @ np.asarray(b))]))) for M in G]
    print(f"    vol(g.b) over the group: spread {np.ptp(vals):.1e}  (exactly invariant)")


def pin4():
    import sys
    sys.path.insert(0, str(ROOT / "experiments" / "dp3"))
    from dhhkw_theta_compare import mu_omega, laplacian_op, NPZ, TO_THEM
    psi, ch, dat = load_psi_dp3(NPZ["deg-18"])
    u = slice_potential(ch, psi)
    mu_T = lambda s: 2.0 * (jnp.dot(s, jax.grad(u)(s)) - u(s))     # Legendre dual x2
    L = laplacian_op(ch, psi)
    ss = sample_slice(jax.random.PRNGKey(3), ch, 6000, eps=1e-3)
    LT = np.asarray(jax.vmap(L(mu_T))(ss))
    LD = np.asarray(jax.vmap(L(mu_omega))(ss))
    print(f"\n[4] L[mu_T] on samples: mean {LT.mean():.12f}, spread {LT.std():.1e}   (exact 4)")
    print(f"    L[mu^D] on samples: mean {LD.mean():.12f}, spread {LD.std():.1e}   (DHHKW anchor: -L/3 = {TO_THEM*LD.mean():.10f}, exact 2)")
    diff = np.asarray(jax.vmap(mu_omega)(ss)) + 1.5 * np.asarray(jax.vmap(mu_T)(ss))
    M = np.column_stack([np.ones(len(ss)), np.asarray(ss)])
    c, *_ = np.linalg.lstsq(M, diff, rcond=None)
    res = diff - M @ c
    spread = float(np.ptp(np.asarray(jax.vmap(mu_T)(ss))))
    print(f"    mu^D + (3/2) mu_T fitted to an affine function: residual rms {np.sqrt((res**2).mean()):.2e}, "
          f"max {np.abs(res).max():.2e}, against a spread of mu_T of {spread:.3f}")
    print(f"    => mu^D = -(3/2) mu_T + affine: the 3 is x = 3s, the -1/2 is theta_(i jbar) = d_i d_jbar mu "
          f"against i d dbar mu.")


if __name__ == "__main__":
    pin9()
    pin4()
