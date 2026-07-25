"""GMSW Y^{p,q} closed-form metric (hep-th/0403002, eqs. (2.1)-(2.2), c=1).

Coordinates (y, theta, phi, psi, alpha). Ric = 4 g for any a in (0,1).
Roots of the cubic a - 3y^2 + 2y^3 (their (2.3), (3.1)-(3.4)):
    lambda = 3q/(2p),  y1 = (1 - lam - sqrt(1 - lam^2/3))/2,  y2 = y1 + lam,
    a = 3 y1^2 - 2 y1^3.
Used ONLY as an external referee; never enters training.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def roots_and_a(p: int, q: int):
    lam = 3.0 * q / (2.0 * p)
    y1 = 0.5 * (1.0 - lam - jnp.sqrt(1.0 - lam**2 / 3.0))
    y2 = y1 + lam
    a = 3.0 * y1**2 - 2.0 * y1**3
    return y1, y2, a


def g5(a: float):
    """5d metric matrix in coords (y, theta, phi, psi, alpha)."""

    def g_fn(x):
        y, th = x[0], x[1]
        w = 2.0 * (a - y**2) / (1.0 - y)
        qq = (a - 3.0 * y**2 + 2.0 * y**3) / (a - y**2)
        f = (a - 2.0 * y + y**2) / (6.0 * (a - y**2))
        # sigma = dpsi - cos(th) dphi ; components in (y,th,phi,psi,al):
        sig = jnp.array([0.0, 0.0, -jnp.cos(th), 1.0, 0.0])
        dal = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
        dth = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
        dph = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
        dy = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        g = ((1.0 - y) / 6.0 * (jnp.outer(dth, dth)
                                + jnp.sin(th) ** 2 * jnp.outer(dph, dph))
             + jnp.outer(dy, dy) / (w * qq)
             + qq / 9.0 * jnp.outer(sig, sig)
             + w * jnp.outer(dal + f * sig, dal + f * sig))
        return g

    return g_fn


def cone6(a: float):
    """6d Ricci-flat cone dr^2 + r^2 g5, coords (r, y, theta, phi, psi, alpha)."""
    g5_fn = g5(a)

    def g_fn(x):
        r = x[0]
        g = jnp.zeros((6, 6))
        g = g.at[0, 0].set(1.0)
        g = g.at[1:, 1:].set(r**2 * g5_fn(x[1:]))
        return g

    return g_fn
