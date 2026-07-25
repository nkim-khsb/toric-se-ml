"""Y^{p,q} learning: the first rung where psi != 0 is genuinely learned.

Fast CI version (poly ansatz, fewer steps). The full run (MLP + deg-8 poly +
cross-validation) is experiments/ypq/train_y21.py; results in log.md.
"""
import jax
import jax.numpy as jnp
import optax
import pytest

from sugrasol.cone import B_CONIFOLD, b_ypq, conifold, ypq
from sugrasol.ypq import (
    gauge_fixed, loss_fn, n_poly_coeffs, poly_psi, sample_slice, slice_chart,
)

jax.config.update("jax_enable_x64", True)

DEG = 6


def _train_poly(chart, ss, steps=2000, lr=3e-3, c0=0.0):
    def L(pc):
        psi = gauge_fixed(poly_psi(pc[0], DEG), chart.anchors)
        return loss_fn(chart, psi, pc[1], ss)

    pc = (jnp.zeros(n_poly_coeffs(DEG)), jnp.array(c0))
    opt = optax.adam(lr)
    state = opt.init(pc)

    @jax.jit
    def step(pc, state):
        l, g = jax.value_and_grad(L)(pc)
        upd, state = opt.update(g, state)
        return optax.apply_updates(pc, upd), state, l

    for _ in range(steps):
        pc, state, l = step(pc, state)
    return pc, float(l)


def test_y21_learns_nontrivial_psi():
    """Irregular Y^{2,1}: loss drops orders of magnitude and psi != 0."""
    chart = slice_chart(ypq(2, 1), b_ypq(2, 1))
    ss = sample_slice(jax.random.PRNGKey(1), chart, 512, eps=2e-3)
    (coeffs, c), l = _train_poly(chart, ss, c0=-2.39)
    assert l < 5e-6
    psi = gauge_fixed(poly_psi(coeffs, DEG), chart.anchors)
    sup = float(jnp.max(jnp.abs(jax.vmap(psi)(ss))))
    assert sup > 1e-2, "psi must be genuinely nonzero for q != 0"
    # held-out generalization
    ss_t = sample_slice(jax.random.PRNGKey(9), chart, 512, eps=2e-3)
    l_t = float(loss_fn(chart, psi, c, ss_t))
    assert l_t < 1e-5
    # independent 4th-order check: Abreu transverse scalar curvature == 12
    # (Ric^T = 6 g^T, doubled-slice normalization; anchor = conifold exact).
    # The training loss is 2nd order — this operator never entered training.
    from sugrasol.ypq import t_of_s

    def u(s):
        t = t_of_s(chart, s)
        lt = chart.cone.ells(t)
        return 0.5 * jnp.sum(lt * jnp.log(lt)) + psi(s)

    def abreu_S(s):
        Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
        T = jax.jacfwd(jax.jacfwd(Hinv))(s)
        return -jnp.einsum("jkjk->", T)

    ss_S = sample_slice(jax.random.PRNGKey(11), chart, 128, eps=1e-2)
    S = jax.vmap(abreu_S)(ss_S)
    assert float(jnp.max(jnp.abs(S - 12.0))) / 12.0 < 0.05


def test_conifold_pipeline_regression():
    """Same pipeline on the conifold must keep psi ~ 0 (nothing to learn)."""
    chart = slice_chart(conifold(), B_CONIFOLD)
    ss = sample_slice(jax.random.PRNGKey(1), chart, 512, eps=2e-3)
    (coeffs, c), l = _train_poly(chart, ss, steps=500, c0=-0.92)
    psi = gauge_fixed(poly_psi(coeffs, DEG), chart.anchors)
    sup = float(jnp.max(jnp.abs(jax.vmap(psi)(ss))))
    assert l < 1e-12
    assert sup < 1e-5
