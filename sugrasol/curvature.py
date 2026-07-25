"""Generic curvature via autodiff: any metric g(x) given as a matrix function.

Pure functions, any dimension. Used for coordinate-free grading: scalar
invariants (|Riem|^2, Riem^3) compared invariant-vs-invariant between a
learned metric and a closed-form metric — no coordinate matching needed.

Conventions:
  Gamma^a_{bc} = (1/2) g^{ad} (d_b g_{dc} + d_c g_{db} - d_d g_{bc}),
  R^a_{bcd} = d_c Gamma^a_{db} - d_d Gamma^a_{cb}
              + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb},
  Ric_{bd} = R^a_{bad}.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def christoffel(g_fn: Callable, x: jnp.ndarray) -> jnp.ndarray:
    g = g_fn(x)
    ginv = jnp.linalg.inv(g)
    dg = jax.jacfwd(g_fn)(x)  # dg[a,b,c] = d_c g_{ab}
    # term[d,b,c] = d_b g_{dc} + d_c g_{db} - d_d g_{bc}
    term = (jnp.einsum("dcb->dbc", dg) + jnp.einsum("dbc->dbc", dg)
            - jnp.einsum("bcd->dbc", dg))
    return 0.5 * jnp.einsum("ad,dbc->abc", ginv, term)


def riemann(g_fn: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """R^a_{bcd}."""
    G = christoffel(g_fn, x)
    dG = jax.jacfwd(lambda xx: christoffel(g_fn, xx))(x)
    # dG[a,b,c,d] = d_d Gamma^a_{bc}
    return (jnp.einsum("adbc->abcd", dG)      # d_c Gamma^a_{db}
            - jnp.einsum("acbd->abcd", dG)    # d_d Gamma^a_{cb}
            + jnp.einsum("ace,edb->abcd", G, G)
            - jnp.einsum("ade,ecb->abcd", G, G))


def ricci(g_fn: Callable, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("abad->bd", riemann(g_fn, x))


def invariants(g_fn: Callable, x: jnp.ndarray):
    """(|Riem|^2, Riem^3), Riem^3 = R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}."""
    g = g_fn(x)
    ginv = jnp.linalg.inv(g)
    R_low = jnp.einsum("ae,ebcd->abcd", g, riemann(g_fn, x))  # R_{abcd}
    M = jnp.einsum("ae,bf,efcd->abcd", ginv, ginv, R_low)  # R^{ab}_{cd}
    R2 = jnp.einsum("abcd,cdab->", M, M)  # = R_{abcd} R^{abcd} (pair symmetry)
    R3 = jnp.einsum("abcd,cdef,efab->", M, M, M)
    return R2, R3


def toric_metric(potential: Callable, n: int) -> Callable:
    """Toric Kaehler metric in (moment coords x, angles): block diagonal
    [Hess u(x), (Hess u(x))^{-1}]; angle-independent."""

    def g_fn(x):
        h = jax.hessian(potential)(x[:n])
        z = jnp.zeros((n, n))
        return jnp.block([[h, z], [z, jnp.linalg.inv(h)]])

    return g_fn
