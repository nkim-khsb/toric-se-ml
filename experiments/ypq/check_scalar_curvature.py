"""Independent psi-sensitive check: Abreu scalar curvature of the transverse
metric must be pointwise CONSTANT (= 12 in doubled-slice coordinates).

Why 12: Ric^T = 6 g^T for any SE_5 => S^T = 24 (4 real dims). Our slice chart
uses doubled moment coordinates (t = 2 * true), which scales the metric so
S_chart = 24/2 = 12. Anchor: the conifold (exact solution) must give 12
identically; this fixes the convention with no free parameter, so 12 is a
genuine prediction for the LEARNED Y^{2,1} metric.

Abreu's formula (math/0004122): S = - sum_{j,k} d^2 u^{jk} / ds_j ds_k,
u^{jk} = (Hess u)^{-1}. Fourth order in u — never used in training (loss was
the 2nd-order MA residual): an independent differential operator.
"""
import jax
import jax.numpy as jnp
import optax

from sugrasol.cone import B_CONIFOLD, b_ypq, conifold, ypq
from sugrasol.ypq import (
    gauge_fixed, loss_fn, n_poly_coeffs, poly_psi, sample_slice, slice_chart,
    t_of_s,
)

jax.config.update("jax_enable_x64", True)


def slice_potential(chart, psi):
    """Transverse symplectic potential u(s) on the doubled slice."""

    def u(s):
        t = t_of_s(chart, s)
        lt = chart.cone.ells(t)
        return 0.5 * jnp.sum(lt * jnp.log(lt)) + psi(s)

    return u


def abreu_S(u, s):
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)  # T[j,k,a,b] = d_a d_b (Hinv[j,k])
    return -jnp.einsum("jkjk->", T)


def stats(chart, psi, ss, label):
    u = slice_potential(chart, psi)
    S = jax.vmap(lambda s: abreu_S(u, s))(ss)
    print(f"{label:28s} mean {float(jnp.mean(S)):10.6f}   "
          f"min {float(jnp.min(S)):10.6f}   max {float(jnp.max(S)):10.6f}")
    return S


# ---------------- anchor: conifold, exact solution
ch_c = slice_chart(conifold(), B_CONIFOLD)
ss_c = sample_slice(jax.random.PRNGKey(4), ch_c, 200, eps=1e-2)
stats(ch_c, lambda s: 0.0, ss_c, "conifold (exact, psi=0)")

# ---------------- Y^{2,1}
ch = slice_chart(ypq(2, 1), b_ypq(2, 1))
ss = sample_slice(jax.random.PRNGKey(1), ch, 1024, eps=2e-3)
ss_eval = sample_slice(jax.random.PRNGKey(11), ch, 200, eps=1e-2)

stats(ch, lambda s: 0.0, ss_eval, "Y^{2,1} BEFORE (psi=0)")

# train the deg-8 polynomial (fast, best loss; same as train_y21.py)
DEG = 8


def L(pc):
    psi = gauge_fixed(poly_psi(pc[0], DEG), ch.anchors)
    return loss_fn(ch, psi, pc[1], ss)


pc = (jnp.zeros(n_poly_coeffs(DEG)), jnp.array(-2.39))
opt = optax.adam(3e-3)
state = opt.init(pc)
step = jax.jit(lambda pc, st: (lambda l_g: (optax.apply_updates(pc, opt.update(l_g[1], st)[0]), opt.update(l_g[1], st)[1], l_g[0]))(jax.value_and_grad(L)(pc)))
for _ in range(4000):
    pc, state, l = step(pc, state)
print(f"trained poly loss: {float(l):.3e}")

psi_tr = gauge_fixed(poly_psi(pc[0], DEG), ch.anchors)
S = stats(ch, psi_tr, ss_eval, "Y^{2,1} AFTER (learned)")
print(f"prediction 12:  max relative deviation "
      f"{float(jnp.max(jnp.abs(S - 12.0)) / 12.0):.3e}")
