"""Scalar Laplacian tool: closed-form anchor.

The conifold transverse base is CP^1 x CP^1 (Kaehler-Einstein, psi=0). In our
doubled-slice normalization (Abreu S=12 => each S^2 has Gauss curvature 3), the
lowest torus-invariant scalar Laplacian eigenvalue is 6 in closed form
(l=(1,0): 3*l(l+1) = 6). This fixes the tool's normalization with no free
parameter, so it is a genuine test (CLAUDE.md rule 7) before grading dP3
against DHHKW's lambda1 = 6.322.
"""
import jax
import jax.numpy as jnp

from sugrasol.cone import B_CONIFOLD, conifold
from sugrasol.laplacian import laplace_spectrum, slice_potential
from sugrasol.ypq import sample_slice, slice_chart

jax.config.update("jax_enable_x64", True)


def _poly_basis(degree, samples, tol=1e-9):
    powers = [(i, j) for tot in range(degree + 1)
              for i in range(tot + 1) for j in [tot - i]]
    raw = lambda s: jnp.stack([s[0] ** i * s[1] ** j for i, j in powers])
    A = jax.vmap(raw)(samples)
    _, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))
    W = Vt[:r].T / S[:r]
    return lambda s: raw(s) @ W


def test_conifold_lowest_eigenvalue_is_six():
    """lambda1(CP^1 x CP^1) = 6 (closed form). Small boundary margin removes the
    boundary-exclusion bias; constant mode must come out at ~0."""
    ch = slice_chart(conifold(), B_CONIFOLD)
    u = slice_potential(ch, lambda s: 0.0)
    smp = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)
    w = laplace_spectrum(u, _poly_basis(6, smp), smp)
    assert abs(float(w[0])) < 1e-6            # constant mode
    assert abs(float(w[1]) / 6.0 - 1.0) < 0.01  # lambda1 = 6 within 1%


def test_conifold_deterministic_quadrature_is_machine_precision():
    """Same anchor, deterministic quadrature (polygon_quadrature) instead of
    Monte-Carlo: lambda1 = 6 to ~1e-8, with NO boundary margin (the stiffness
    integrand u^{jk} vanishes on partial P on its own).  This is the 0.46%
    -> machine-precision fix of log.md 2026-07-24; the basis is still built on
    an MC cloud for conditioning, only the INTEGRATION is deterministic."""
    from sugrasol.laplacian import polygon_quadrature

    ch = slice_chart(conifold(), B_CONIFOLD)
    u = slice_potential(ch, lambda s: 0.0)
    cloud = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)
    basis = _poly_basis(6, cloud)
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=24)
    w = laplace_spectrum(u, basis, nodes, weights=wts)
    assert abs(float(w[0])) < 1e-9              # constant mode, no MC noise
    assert abs(float(w[1]) / 6.0 - 1.0) < 1e-7  # lambda1 = 6 to ~1e-8


# --------------------------------------------------------------------------
# Charged sector (link_charged_spectrum) — conventions pinned by closed forms
# --------------------------------------------------------------------------

def test_c3_charged_spectrum_is_round_s5():
    """Flat C^3 (CY gauge): the link at level (b,y)=1/2 is the unit round
    S^5, spectrum E = k(k+4).  Pins ALL conventions at once: slice level,
    charge normalization, chiral lattice.  Chiral count: 3 at lambda=1
    (z_i), 6 at lambda=2 (sym^2)."""
    from sugrasol.cone import flat_cn_cy_gauge
    from sugrasol.laplacian import (charged_basis, chiral_lattice_points,
                                    link_charged_spectrum)

    G3d, cone, b = flat_cn_cy_gauge(3)
    ch = slice_chart(cone, b)
    smp = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)

    pts = chiral_lattice_points(cone, b, 2.01)
    assert [lam for lam, _ in pts] == [1.0] * 3 + [2.0] * 6

    # chiral sector: E_min = 1*5 = 5; next in sector = 21 (k=3)
    bas = charged_basis(ch, (0, 1, 0), 6, smp)
    w = link_charged_spectrum(G3d, ch, (0, 1, 0), bas, smp, level=0.5)
    assert abs(float(w[0]) / 5.0 - 1.0) < 0.01
    assert abs(float(w[1]) / 21.0 - 1.0) < 0.03
    # m=0: first nonzero = 12 (k=2 invariants)
    bas0 = charged_basis(ch, (0, 0, 0), 6, smp)
    w0 = link_charged_spectrum(G3d, ch, (0, 0, 0), bas0, smp, level=0.5)
    assert abs(float(w0[0])) < 1e-6
    assert abs(float(w0[1]) / 12.0 - 1.0) < 0.02


def test_conifold_chiral_quartet():
    """T^{1,1}: exactly 4 chiral states at lambda = 3/2 (the A_i B_j
    quartet, Delta = 3/2) with E_min = (3/2)(3/2+4) = 8.25; 9 at lambda=3
    (3x3 of (AB)^2).  E_min emerges from the numerics at MC accuracy.
    Also the m=0 link eigenvalue is 12 = TWICE the doubled-slice base
    anchor of test_conifold_lowest_eigenvalue_is_six (the doubled slice
    halves curvatures — recorded so nobody 'fixes' one against the other)."""
    from sugrasol.cone import regular_cone_potential
    from sugrasol.laplacian import (charged_basis, chiral_lattice_points,
                                    link_charged_spectrum)

    cone, b = conifold(), B_CONIFOLD
    G3d = regular_cone_potential(cone, b)
    ch = slice_chart(cone, b)
    smp = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)

    pts = chiral_lattice_points(cone, b, 3.01)
    lams = [lam for lam, _ in pts]
    assert lams == [1.5] * 4 + [3.0] * 9

    for lam, m in pts[:2]:
        bas = charged_basis(ch, m, 6, smp)
        w = link_charged_spectrum(G3d, ch, m, bas, smp, level=0.5)
        assert abs(float(w[0]) / (lam * (lam + 4.0)) - 1.0) < 0.01

    bas0 = charged_basis(ch, (0, 0, 0), 6, smp)
    w0 = link_charged_spectrum(G3d, ch, (0, 0, 0), bas0, smp, level=0.5)
    assert abs(float(w0[1]) / 12.0 - 1.0) < 0.02


def test_conifold_chiral_deterministic_quadrature():
    """Charged sector, deterministic quadrature: the chiral anchor
    E_min = lambda(lambda+4) comes out EXACT (MC gives 0.1-0.7%, seed-dependent
    -- log.md 2026-07-24).  m=(1,0,0) has lambda = <b,m> = 3, so E_min = 21."""
    from sugrasol.cone import regular_cone_potential
    from sugrasol.laplacian import (charged_basis, link_charged_spectrum,
                                    polygon_quadrature)

    cone, b = conifold(), B_CONIFOLD
    G3d = regular_cone_potential(cone, b)
    ch = slice_chart(cone, b)
    cloud = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=32)

    m = (1, 0, 0)
    lam = float(jnp.asarray(b) @ jnp.asarray(m, dtype=jnp.float64))  # = 3
    bas = charged_basis(ch, m, 6, cloud)
    w = link_charged_spectrum(G3d, ch, m, bas, nodes, level=0.5, weights=wts)
    assert abs(float(w[0]) / (lam * (lam + 4.0)) - 1.0) < 1e-6


def test_dp3_chiral_tower_protected():
    """dP3 positive control, charged sector (log.md 2026-07-25). The chiral
    anchor E = lambda(lambda+4), lambda = <b,m>, is fixed by the Reeb vector
    b=(3,0,0) alone and is psi-BLIND (it uses holomorphicity + cone structure,
    not the Einstein condition). The lowest dP3 chiral multiplet is SEVENFOLD
    at lambda=3 -- the seven lattice points of the reflexive hexagon (six
    boundary, one interior) -- with E=21. Deterministic quadrature on the
    canonical (Guillemin, psi=0) toric metric reproduces it to machine
    precision, validating the charged-sector tool and the Reeb vector on dP3
    (the analogue of the conifold E=21 anchor, now on the hexagon)."""
    from sugrasol.cone import dp3, B_DP3
    from sugrasol.ypq import cone_potential
    from sugrasol.laplacian import (charged_basis, chiral_lattice_points,
                                    link_charged_spectrum, polygon_quadrature)

    ch = slice_chart(dp3(), B_DP3)
    G3d = cone_potential(ch, lambda s: 0.0)          # canonical (Guillemin) cone
    pts = chiral_lattice_points(dp3(), B_DP3, 3.01)
    assert [round(lam, 4) for lam, _ in pts] == [3.0] * 7   # sevenfold at lam=3

    cloud = sample_slice(jax.random.PRNGKey(5), ch, 20000, eps=1e-4)
    nodes, wts = polygon_quadrature(ch.verts_s, ngl=32)
    for lam, m in pts[:3]:                            # a few of the seven
        bas = charged_basis(ch, m, 6, cloud)
        w = link_charged_spectrum(G3d, ch, m, bas, nodes, level=0.5, weights=wts)
        assert abs(float(w[0]) / (lam * (lam + 4.0)) - 1.0) < 1e-6
