"""Network ansatz psi_theta(x) for G = G_can + psi.

Requirements (CLAUDE.md rules 1, 3):
  - psi smooth up to the boundary (it must NOT modify the l log l boundary
    behaviour of G_can — Guillemin BC lives entirely in G_can).
  - symmetry: T^{1,1} transverse square has D4 x (P^1 swap) symmetry;
    enforce by input symmetrization (invariant layer), not by penalty.
  - init at psi ~ 0 (pretrain seed = Guillemin), NOT random-large.

Small MLP is enough to start (De Luca: H=10, D=2 sufficed in 3d).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def init_mlp(key: jax.Array, sizes=(2, 16, 16, 1), scale: float = 1e-2):
    params = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        # NB: biases must NOT be zero: with tanh, zero biases make the MLP an
        # odd function, which the D4 symmetrization annihilates identically.
        params.append(
            (
                scale * jax.random.normal(k1, (n, m), dtype=jnp.float64),
                scale * jax.random.normal(k2, (n,), dtype=jnp.float64),
            )
        )
    return params


def mlp(params, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(W @ h + b)
    W, b = params[-1]
    return (W @ h + b)[0]


def symmetrized_psi(params, x: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
    """Average over the square's symmetry group acting about `center`.

    Group: coordinate swap x1<->x2 and reflections x_i -> 2c_i - x_i (order 8).
    """
    u = x - center
    orbit = [
        jnp.array([s1 * a, s2 * b])
        for (a, b) in [(u[0], u[1]), (u[1], u[0])]
        for s1 in (1.0, -1.0)
        for s2 in (1.0, -1.0)
    ]
    vals = jnp.stack([mlp(params, center + o) for o in orbit])
    return jnp.mean(vals)
