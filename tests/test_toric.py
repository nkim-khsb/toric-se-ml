"""Normalization-safe checks for the toric layer (no MA constants involved)."""
import jax
import jax.numpy as jnp
import pytest

from sugrasol.sampling import sample_polytope
from sugrasol.toric import guillemin_potential, square, sym_hessian

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sq():
    return square(1.0)


def test_ells_positive_interior(sq):
    x = jnp.array([0.3, 0.7])
    assert bool(sq.is_interior(x))
    assert not bool(sq.is_interior(jnp.array([1.1, 0.5])))


def test_guillemin_hessian_positive_definite(sq):
    key = jax.random.PRNGKey(0)
    pts = sample_polytope(key, sq, jnp.zeros(2), jnp.ones(2), 200, eps=1e-3)
    G = lambda x: guillemin_potential(sq, x)
    eigs = jax.vmap(lambda x: jnp.linalg.eigvalsh(sym_hessian(G, x)))(pts)
    assert bool(jnp.all(eigs > 0)), "symplectic potential must be strictly convex"


def test_guillemin_square_symmetry(sq):
    """G_can inherits the D4 symmetry of the square about its center."""
    G = lambda x: guillemin_potential(sq, x)
    c = jnp.array([0.5, 0.5])
    x = jnp.array([0.31, 0.62])
    u = x - c
    images = [c + jnp.array(v) for v in
              [(u[0], u[1]), (u[1], u[0]), (-u[0], u[1]), (u[0], -u[1])]]
    vals = jnp.stack([G(y) for y in images])
    assert float(jnp.max(jnp.abs(vals - vals[0]))) < 1e-12


def test_hessian_boundary_blowup(sq):
    """Near a facet, the normal-normal component of Hess(G_can) ~ 1/(2 l)."""
    G = lambda x: guillemin_potential(sq, x)
    for l in [1e-2, 1e-3, 1e-4]:
        x = jnp.array([l, 0.5])
        h = sym_hessian(G, x)
        assert abs(float(h[0, 0]) * 2 * l - 1.0) < 5e-2


def test_legendre_consistency(sq):
    """Hess_y(K) = [Hess_x(G)]^{-1} — check y(x) = grad G is invertible here."""
    G = lambda x: guillemin_potential(sq, x)
    x = jnp.array([0.3, 0.7])
    Jy = jax.jacobian(jax.grad(G))(x)  # dy/dx = Hess(G)
    assert float(jnp.linalg.det(Jy)) > 0
