"""MSY cone layer: protected quantities + Ricci-flat MA (smoke test (c))."""
import jax
import jax.numpy as jnp
import pytest

from sugrasol.cone import (
    B_CONIFOLD,
    B_DP3,
    b_ypq,
    cone_ma_residual,
    conifold,
    dp3,
    flat_cn_cy_gauge,
    minimize_reeb,
    regular_cone_potential,
    sample_cone,
    vol_delta,
    vol_sigma,
    vol_Y,
    vol_ypq,
    ypq,
)

jax.config.update("jax_enable_x64", True)

VOL_T11 = 16.0 * jnp.pi**3 / 27.0


# ------------------------------------------------------- protected quantities
def test_conifold_volume_at_msy_reeb():
    """vol(Y) at b=(3,3/2,3/2) must be 16 pi^3/27 (MSY eq. (3.30))."""
    assert abs(float(vol_Y(conifold(), B_CONIFOLD)) / float(VOL_T11) - 1) < 1e-12


def test_conifold_polytope_volume_exact():
    """vol(Delta_b) = 1/81 (hand computation: two tetrahedra, det/6 each)."""
    assert abs(float(vol_delta(conifold(), B_CONIFOLD)) - 1.0 / 81.0) < 1e-14


def test_conifold_Z_function_shape():
    """vol(Y) as a function of b matches MSY eq. (3.28) up to the fixed
    2n(2pi)^n normalization: Z = (x-2)x / (8 y t (x-t)(x-y)), b=(x,y,t)."""
    key = jax.random.PRNGKey(3)
    for _ in range(5):
        key, k = jax.random.split(key)
        x, y, t = 3.0, *jax.random.uniform(k, (2,), minval=1.2, maxval=1.8)
        Z = (x - 2) * x / (8 * y * t * (x - t) * (x - y))
        ratio = float(vol_Y(conifold(), jnp.array([x, y, t]))) / float(Z)
        assert abs(ratio / (8 * jnp.pi**3) - 1) < 1e-10  # constant ratio (2pi)^3


def test_sigma_volumes():
    """3-cycle volumes, MSY (3.25): conifold gives four copies of 8 pi^2/9
    (their eq. (3.30)); identity vol(Y) = (pi/2b_1) sum_a vol(Sigma_a)."""
    from sugrasol.cone import vol_sigma

    vs = vol_sigma(conifold(), B_CONIFOLD)
    assert float(jnp.max(jnp.abs(vs - 8.0 * jnp.pi**2 / 9.0))) < 1e-12
    assert abs(float(jnp.pi / 6.0 * jnp.sum(vs)) / float(VOL_T11) - 1) < 1e-12
    # Y^{2,1}: identity between (3.25) and (3.26) at the closed-form Reeb
    vs2 = vol_sigma(ypq(2, 1), b_ypq(2, 1))
    assert abs(float(jnp.pi / 6.0 * jnp.sum(vs2)) / float(vol_ypq(2, 1)) - 1) < 1e-12


def test_reeb_by_volume_minimization():
    """Minimizing vol over b (b_1=3) must recover b=(3,3/2,3/2) (MSY 3.29)."""
    b = minimize_reeb(conifold(), jnp.array([1.4, 1.6]), steps=2000, lr=2e-3)
    assert float(jnp.max(jnp.abs(b - B_CONIFOLD))) < 1e-4


@pytest.mark.parametrize("p,q", [(2, 1), (3, 1), (3, 2), (5, 3)])
def test_ypq_volume_and_reeb(p, q):
    """Next-rung anchors: closed-form vol(Y^{p,q}) (3.35) at closed-form Reeb
    (3.33)-(3.34), and numeric minimization reproducing that Reeb."""
    cone = ypq(p, q)
    b = b_ypq(p, q)
    assert abs(float(vol_Y(cone, b)) / float(vol_ypq(p, q)) - 1) < 1e-12
    b_num = minimize_reeb(cone, jnp.array(b[1:]) * 1.05, steps=3000, lr=2e-3)
    assert float(jnp.max(jnp.abs(b_num - b))) < 1e-3


# ------------------------------------------------------- Ricci-flat cone MA
def test_flat_c3_cy_gauge():
    """Flat C^3 in CY gauge: R = ln det Hess G + 2 dG/dy_1 + c constant, and
    b_1 = n = 3 (MSY eq. (2.62))."""
    G, cone, b = flat_cn_cy_gauge(3)
    assert abs(float(b[0]) - 3.0) < 1e-14
    ys = sample_cone(jax.random.PRNGKey(5), cone, b, 200)
    r = jax.vmap(lambda y: cone_ma_residual(G, y, 0.0))(ys)
    assert float(jnp.max(r) - jnp.min(r)) < 1e-10


def test_flat_c3_decomposition_matches():
    """Decomposition ansatz reproduces exact flat G up to gauge: residual of
    the ansatz is also constant (same equation, possibly different c)."""
    _, cone, b = flat_cn_cy_gauge(3)
    G = regular_cone_potential(cone, b)
    ys = sample_cone(jax.random.PRNGKey(6), cone, b, 200)
    r = jax.vmap(lambda y: cone_ma_residual(G, y, 0.0))(ys)
    assert float(jnp.max(r) - jnp.min(r)) < 1e-10


def test_conifold_cone_is_ricci_flat():
    """T^{1,1} at cone level: the decomposition potential built from the
    Guillemin transverse solution solves MSY (2.54) — smoke test (c) core."""
    cone = conifold()
    G = regular_cone_potential(cone, B_CONIFOLD)
    ys = sample_cone(jax.random.PRNGKey(7), cone, B_CONIFOLD, 500)
    r = jax.vmap(lambda y: cone_ma_residual(G, y, 0.0))(ys)
    spread = float(jnp.max(r) - jnp.min(r))
    print(f"conifold residual spread: {spread:.3e}, mean: {float(jnp.mean(r)):.6f}")
    assert spread < 1e-9


def test_homogeneity():
    """Cone structure: Hess G homogeneous of degree -1."""
    cone = conifold()
    G = regular_cone_potential(cone, B_CONIFOLD)
    y = jnp.array([0.3, 0.21, 0.17])
    h1 = jax.hessian(G)(y)
    h2 = jax.hessian(G)(2.0 * y)
    assert float(jnp.max(jnp.abs(2.0 * h2 - h1))) < 1e-12


# ----------------------------------------------------------------- dP3 rung
def test_dp3_smoothness():
    """dP3 cone is smooth: det[v_{a-1}, v_a, v_{a+1}] = 1 for all facets
    (DHHKW hep-th/0703057 eq. (3.49), our normals)."""
    v = dp3().normals
    d = v.shape[0]
    assert d == 6
    for a in range(d):
        det = jnp.linalg.det(jnp.stack([v[(a - 1) % d], v[a], v[(a + 1) % d]]))
        assert abs(abs(float(det)) - 1.0) < 1e-12


def test_dp3_reeb_is_regular():
    """MSY volume minimization gives the regular Reeb b = (3, 0, 0) (hexagon
    D6 symmetry -> Reeb at the centroid)."""
    b = minimize_reeb(dp3(), jnp.array([0.05, 0.05]), steps=2000, lr=1e-2)
    assert float(jnp.max(jnp.abs(b - B_DP3))) < 1e-4


def test_dp3_protected_volumes():
    """vol(Y) = 2 pi^3/9; the six vol(Sigma_a) are all 2 pi^2/9 (hexagon
    symmetry); and the identity vol(Y) = (pi/2 b_1) sum_a vol(Sigma_a) holds."""
    cone = dp3()
    vY = vol_Y(cone, B_DP3)
    assert abs(float(vY) - 2.0 * jnp.pi**3 / 9.0) < 1e-10
    vS = vol_sigma(cone, B_DP3)
    assert float(jnp.max(jnp.abs(vS - 2.0 * jnp.pi**2 / 9.0))) < 1e-10
    ident = jnp.pi / (2 * B_DP3[0]) * jnp.sum(vS)
    assert abs(float(vY - ident)) < 1e-10
