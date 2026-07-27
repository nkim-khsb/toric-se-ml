"""Anchors for the autodiff curvature routine used as an independent grader.

paper1-methods.tex claims the routine (Christoffel -> Riemann -> invariants) is
anchored on the round S^2, on the closed-form conifold, and on the transcribed
GMSW metrics.  Those anchors lived only in prose (note-ypq.tex) and log.md; this
pins them, so every quoted anchor in the paper is backed by a test
(review-paper1-20260727.md A11, A15).

Measured here, and quoted in the paper as such:
    round S^2 (|Riem|^2 = 4 K^2)                exact to 1e-11
    GMSW g5, Ric = 4 g                          1e-15 relative
    conifold cone via OUR Guillemin potential   1e-13 relative

The last is coarser than the closed-form anchors and that is expected: it goes
through a numerical Hessian inverse of G, not a hand-written metric.  Note the
two distinct roles of the conifold: test_cone.py::test_conifold_cone_is_ricci_flat
checks the MONGE-AMPERE residual of the ansatz, whereas here the same geometry is
pushed through the CURVATURE routine, which shares no code with the residual, so
Ric = 0 is an independent statement.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import B_CONIFOLD, conifold
from sugrasol.curvature import invariants, ricci, toric_metric
from sugrasol.gmsw import cone6, g5, roots_and_a
from sugrasol.ypq import cone_potential, sample_slice, slice_chart, t_of_s


def test_round_s2_riemann_norm():
    """Round S^2 of radius rho: constant curvature K = 1/rho^2 gives
    R_{abcd} = K (g_ac g_bd - g_ad g_bc), hence |Riem|^2 = 4 K^2."""
    for rho in (1.0, 2.0, 0.5):
        g_fn = lambda x, r=rho: r**2 * jnp.diag(
            jnp.array([1.0, jnp.sin(x[0]) ** 2]))
        for theta in (0.4, 1.0, jnp.pi / 2, 2.3):
            R2, _ = invariants(g_fn, jnp.array([float(theta), 0.7]))
            exact = 4.0 / rho**4
            assert abs(float(R2) - exact) < 1e-11 * max(1.0, exact), \
                (rho, theta, float(R2), exact)


def test_gmsw_transcription_is_einstein():
    """GMSW Y^{p,q} (hep-th/0403002 (2.1)-(2.2)) satisfies Ric = 4g -- the check
    that validates the transcription of the referee metric itself -- and its cone
    is Ricci-flat."""
    worst5 = worst6 = 0.0
    for p, q in ((2, 1), (3, 2), (7, 3)):
        y1, y2, a = roots_and_a(p, q)
        g_fn, gc = g5(float(a)), cone6(float(a))
        for f in (0.3, 0.5, 0.7):
            yv = float(y1 + f * (y2 - y1))
            x5 = jnp.array([yv, 0.9, 0.3, 0.7, 0.2])
            g = g_fn(x5)
            worst5 = max(worst5, float(jnp.max(jnp.abs(ricci(g_fn, x5) - 4.0 * g))
                                      / jnp.max(jnp.abs(g))))
            x6 = jnp.concatenate([jnp.ones(1), x5])
            worst6 = max(worst6, float(jnp.max(jnp.abs(ricci(gc, x6)))
                                       / jnp.max(jnp.abs(gc(x6)))))
    print(f"GMSW g5: |Ric - 4g|/|g| <= {worst5:.2e};  "
          f"cone: |Ric|/|g| <= {worst6:.2e}")
    assert worst5 < 1e-14
    assert worst6 < 1e-14


def test_conifold_cone_ricci_flat_through_curvature_routine():
    """Closed-form conifold cone (psi = 0 IS the solution): Ric = 0.  The 6d
    invariant |Riem|^2 comes out 96 at the r = 1 slice, constant over the polygon
    -- a homogeneity check the routine passes for free (T^{1,1} is homogeneous)."""
    chart = slice_chart(conifold(), B_CONIFOLD)
    g6 = toric_metric(cone_potential(chart, lambda s: 0.0), 3)
    ss = sample_slice(jax.random.PRNGKey(4), chart, 12, eps=5e-2)

    def probe(s):
        y = t_of_s(chart, s) / 2.0                      # l_b = 1/2  <=>  r = 1
        x6 = jnp.concatenate([y, jnp.zeros(3)])
        g = g6(x6)
        return (jnp.max(jnp.abs(ricci(g6, x6))) / jnp.max(jnp.abs(g)),
                invariants(g6, x6)[0])

    rel, R2 = jax.vmap(probe)(ss)
    print(f"conifold cone: |Ric|/|g| <= {float(jnp.max(rel)):.2e}, "
          f"|Riem|^2 = 96 to {float(jnp.max(jnp.abs(R2 - 96.0))):.2e}")
    assert float(jnp.max(rel)) < 1e-12
    assert float(jnp.max(jnp.abs(R2 - 96.0))) < 1e-10
