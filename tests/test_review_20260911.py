"""Regression tests for the numbers added in the 2026-09-11 review response.

(1) Full torus-invariant spectrum on dP3: the symplectic coordinates are exact
    eigenfunctions with eigenvalue 2 Lambda (DHHKW (3.40)), so the lowest nonzero
    level in a general polynomial basis is 2.000 (DHHKW units) and doubly
    degenerate; the value the paper compares with DHHKW, 6.3228, is the lowest
    D6-INVARIANT level and sits third.
(2) Section 6 trial space: on a polygon whose edge lines are in general position
    an exactly-admissible POLYNOMIAL carries only affine vertex data
    (intersection-point conditions have rank d-3), so the Wachspress-augmented
    space is needed; on the square the kernel is exact and every other direction
    sits at E = 1/2 (Stokes: int theta ^ theta = 0 for forms with zero periods).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "experiments" / "dp3kt", ROOT / "experiments" / "dp3"):
    sys.path.insert(0, str(p))


def test_dp3_moment_maps_are_eigenfunctions_at_2Lambda():
    from scipy.linalg import eigh
    from sugrasol.artifacts import load_psi_dp3
    from sugrasol.laplacian import polygon_quadrature, slice_potential
    from sugrasol.ypq import dihedral_matrices, sample_slice

    psi, ch, _ = load_psi_dp3(ROOT / "experiments/dp3smooth/dp3_G_deg18.npz")
    u = slice_potential(ch, psi)
    pw = [(i, t - i) for t in range(0, 13) for i in range(t + 1)]
    raw = lambda s: jnp.stack([s[0] ** i * s[1] ** j for i, j in pw])
    cloud = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)
    _, sv, Vt = jnp.linalg.svd(jax.vmap(raw)(cloud), full_matrices=False)
    r = int(jnp.sum(sv > 1e-12 * sv[0]))
    Wb = Vt[:r].T / sv[:r]
    basis = lambda s: raw(s) @ Wb
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=20)
    uinv = lambda s: jnp.linalg.inv(jax.hessian(u)(s))
    jac = jax.jacfwd(basis)
    A = np.asarray(jnp.tensordot(wts, jax.vmap(lambda s: jac(s) @ uinv(s) @ jac(s).T)(nodes), axes=(0, 0)))
    B = np.asarray(jnp.tensordot(wts, jax.vmap(lambda s: jnp.outer(basis(s), basis(s)))(nodes), axes=(0, 0)))
    w, vec = eigh(0.5 * (A + A.T), 0.5 * (B + B.T))
    lam = w * 4.0 / 12.0                                   # DHHKW units
    assert abs(lam[0]) < 1e-8
    assert abs(lam[1] - 2.0) < 1e-6 and abs(lam[2] - 2.0) < 1e-6   # the moment maps
    assert lam[3] > 2.5                                    # nothing else that low
    # the first D6-trivial level: average the eigenfunction over the group
    group = np.asarray(dihedral_matrices(ch.verts_s))
    fn = np.asarray(jax.vmap(basis)(nodes)) @ vec
    trivial = []
    for k in range(1, 12):
        proj = np.mean([np.asarray(jax.vmap(basis)(jnp.asarray(nodes @ g.T))) @ vec[:, k] for g in group], 0)
        frac = np.sum(wts * proj * fn[:, k]) / np.sum(wts * fn[:, k] ** 2)
        if frac > 0.5:
            trivial.append(lam[k])
    assert abs(trivial[0] - 6.32277) < 2e-3               # DHHKW's lambda_1 is the first A1 level


def test_exactly_admissible_polynomials_carry_no_vertex_data_on_generic_polygons():
    from r2b_stream_wachspress import intersection_rank
    from sugrasol.cone import dp2, dp3, B_DP3, ypq, b_ypq
    from sugrasol.ypq import slice_chart

    b3 = 3.0 * (19.0 - 3.0 * np.sqrt(33.0)) / 16.0
    for cone, b, d in ((dp2(), jnp.array([3.0, 0.0, b3]), 5), (ypq(3, 2), b_ypq(3, 2), 4)):
        ch = slice_chart(cone, b)
        rank, n = intersection_rank(ch.verts_s)
        assert rank == d - 3, (d, rank, n)                 # only affine traces survive
    ch = slice_chart(dp3(), B_DP3)
    rank, n = intersection_rank(ch.verts_s)
    assert rank == 3                                       # hexagon too: no exact polynomial realizer


def test_wachspress_space_on_the_conifold_is_exact_and_stokes_gives_one_half():
    from r2b_stream_wachspress import wachspress_space, design
    from r2b_stream_admissible import solve
    from r2b_ypq_calibration import load_case

    ch, u, _, _ = load_case("conifold")
    B, K, rho = wachspress_space(ch.verts_s, 8)
    X, Xs, _, _ = design(ch, u, B, 24)
    lam, coef, r = solve(X, Xs)
    assert r == 1 + 15                                     # d-3 realizers + (D-d+2)(D-d+1)/2
    assert lam[0] < 1e-18                                  # chi = x1 x2, exact
    assert np.allclose(lam[1:], 0.5, atol=1e-9)            # zero-period forms: E = 1/2 exactly
