"""Y^{2,1} (irregular Reeb): learn psi via cone MA loss + polynomial cross-check."""
import jax
import jax.numpy as jnp
import optax

from sugrasol.cone import b_ypq, ypq
from sugrasol.nets import init_mlp
from sugrasol.ypq import (
    gauge_fixed, loss_fn, mlp_psi, n_poly_coeffs, poly_psi, sample_slice,
    slice_chart,
)

jax.config.update("jax_enable_x64", True)

P, Q = 2, 1
chart = slice_chart(ypq(P, Q), b_ypq(P, Q))
ss = sample_slice(jax.random.PRNGKey(1), chart, 1024, eps=2e-3)
ss_test = sample_slice(jax.random.PRNGKey(2), chart, 1024, eps=2e-3)


def make_loss(psi_of_params):
    def L(params, c, batch):
        psi = gauge_fixed(psi_of_params(params), chart.anchors)
        return loss_fn(chart, psi, c, batch)
    return jax.jit(L)


def train(psi_of_params, params, c0, steps, lr):
    L = make_loss(psi_of_params)
    opt = optax.adam(lr)
    state = opt.init((params, c0))

    @jax.jit
    def step(pc, state):
        l, g = jax.value_and_grad(lambda pc: L(pc[0], pc[1], ss))(pc)
        upd, state = opt.update(g, state)
        return optax.apply_updates(pc, upd), state, l

    pc = (params, c0)
    for k in range(steps):
        pc, state, l = step(pc, state)
        if k % max(1, steps // 6) == 0:
            print(f"  step {k:5d}  loss {float(l):.3e}")
    l_test = float(L(pc[0], pc[1], ss_test))
    return pc, l_test


# ---------------- MLP (pretrain-then-relax: init near psi=0 = previous rung)
print("== MLP ==")
params0 = init_mlp(jax.random.PRNGKey(0), sizes=(2, 24, 24, 1), scale=1e-2)
(params_nn, c_nn), l_nn = train(mlp_psi, params0, jnp.array(-2.3869), 4000, 3e-3)
print(f"MLP held-out loss: {l_nn:.3e}, c = {float(c_nn):.6f}")

# ---------------- independent polynomial ansatz (DHHKW Sec. 6 style)
print("== poly deg 8 ==")
DEG = 8
coeffs0 = jnp.zeros(n_poly_coeffs(DEG))
(coeffs, c_pl), l_pl = train(lambda cf: poly_psi(cf, DEG), coeffs0,
                             jnp.array(-2.3869), 4000, 3e-3)
print(f"poly held-out loss: {l_pl:.3e}, c = {float(c_pl):.6f}")

# ---------------- cross-validation: same gauge, pointwise comparison
psi_nn = gauge_fixed(mlp_psi(params_nn), chart.anchors)
psi_pl = gauge_fixed(poly_psi(coeffs, DEG), chart.anchors)
dv = jax.vmap(lambda s: psi_nn(s) - psi_pl(s))(ss_test)
v_nn = jax.vmap(psi_nn)(ss_test)
print(f"sup|psi_nn| = {float(jnp.max(jnp.abs(v_nn))):.4e}")
print(f"sup|psi_nn - psi_poly| = {float(jnp.max(jnp.abs(dv))):.4e}")
print(f"c_nn - c_poly = {float(c_nn - c_pl):.3e}")
