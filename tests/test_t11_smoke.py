"""T^{1,1} smoke tests (CLAUDE.md §2): the gate before any real training."""
import jax
import jax.numpy as jnp
import optax
import pytest

from sugrasol import t11
from sugrasol.losses import ke_loss, ke_residual, smoothness_indicator
from sugrasol.nets import init_mlp, symmetrized_psi
from sugrasol.sampling import sample_polytope
from sugrasol.toric import guillemin_potential, total_potential

jax.config.update("jax_enable_x64", True)

LO = -t11.S * jnp.ones(2)
HI = t11.S * jnp.ones(2)
CENTER = jnp.zeros(2)


@pytest.fixture(scope="module")
def setup():
    p = t11.transverse_polytope()
    prob = t11.transverse_problem()
    xs = sample_polytope(jax.random.PRNGKey(1), p, LO, HI, 1024, eps=1e-4)
    return p, prob, xs


def test_psi_zero_is_solution(setup):
    """(a) Guillemin potential of [-1/6,1/6]^2 solves MA with Lambda=6, c=2 ln s."""
    p, prob, xs = setup
    G = lambda x: guillemin_potential(p, x)
    r = jax.vmap(lambda x: ke_residual(prob, G, x))(xs)
    assert float(jnp.max(jnp.abs(r))) < 1e-10


def test_wrong_normalization_fails(setup):
    """Anti-test: wrong Lambda must NOT give zero residual (guards silent pass)."""
    p, prob, xs = setup
    bad = prob._replace(lam=4.0)
    G = lambda x: guillemin_potential(p, x)
    r = jax.vmap(lambda x: ke_residual(bad, G, x))(xs)
    assert float(jnp.max(jnp.abs(r))) > 1e-2


def test_linear_mode_is_flat_direction(setup):
    """psi linear => residual unchanged (automorphism gauge; motivates D4 layer)."""
    p, prob, xs = setup
    G0 = lambda x: guillemin_potential(p, x)
    G1 = lambda x: guillemin_potential(p, x) + 0.05 * x[0] - 0.02 * x[1]
    r0 = jax.vmap(lambda x: ke_residual(prob, G0, x))(xs)
    r1 = jax.vmap(lambda x: ke_residual(prob, G1, x))(xs)
    assert float(jnp.max(jnp.abs(r0 - r1))) < 1e-10


def test_smoothness_indicator_positive(setup):
    """DHHKW (3.14): det(Hess G) * prod l_a smooth positive up to dP."""
    p, prob, _ = setup
    G = lambda x: guillemin_potential(p, x)
    edge = jnp.stack([jnp.full(50, -t11.S + 1e-6), jnp.linspace(-0.9, 0.9, 50) * t11.S], axis=1)
    vals = jax.vmap(lambda x: smoothness_indicator(p, G, x))(edge)
    assert bool(jnp.all(vals > 0))


def test_perturbed_recovers_psi_zero(setup):
    """(b) Perturbed init -> training drives psi -> 0 (pipeline end-to-end)."""
    p, prob, xs = setup
    params = init_mlp(jax.random.PRNGKey(7), sizes=(2, 16, 16, 1), scale=0.5)

    def G_of(params):
        psi = lambda x: symmetrized_psi(params, x, CENTER)
        return total_potential(p, psi)

    loss_fn = jax.jit(lambda params: ke_loss(prob, G_of(params), xs))

    l0 = float(loss_fn(params))
    assert l0 > 1e-4, "perturbation too small to be a meaningful test"

    opt = optax.adam(1e-2)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        l, g = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(g, state)
        return optax.apply_updates(params, updates), state, l

    for _ in range(1500):
        params, state, l = step(params, state)

    # held-out points: no overfitting to collocation set
    xs_test = sample_polytope(jax.random.PRNGKey(2), p, LO, HI, 1024, eps=1e-4)
    l_test = float(ke_loss(prob, G_of(params), xs_test))
    psi_sup = float(
        jnp.max(jnp.abs(jax.vmap(lambda x: symmetrized_psi(params, x, CENTER))(xs_test)))
    )
    print(f"loss: {l0:.3e} -> train {float(l):.3e} / test {l_test:.3e}; sup|psi| = {psi_sup:.3e}")
    assert l_test < 1e-6
    assert psi_sup < 5e-3
