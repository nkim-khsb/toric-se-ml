"""Numeric exterior calculus on a coordinate chart (rung B; reused on rung D).

A p-form field is a function  x (n,) -> fully antisymmetric array (n,)*p.
Everything is pointwise and autodiff-based: exterior derivative via jacfwd,
Hodge star via the metric and the Levi-Civita symbol, wedge via explicit
antisymmetrization.  No global structure (no cohomology, no integration) —
those live in the experiment scripts.

Conventions (components, index placement):
  * A p-form A = (1/p!) A_{a1..ap} dx^{a1} ^ ... ^ dx^{ap}, A antisymmetric.
  * (dA)_{m a1..ap} = (p+1) d_[m A_{a1..ap]}.
  * (A ^ B)_{a..b..} = (p+q)!/(p! q!) Alt(A x B).
  * (star A)_{b1..bq} = (1/p!) sqrt(det g) eps_{a1..ap b1..bq}
        g^{a1 c1}...g^{ap cp} A_{c1..cp},   q = n - p,
    with eps the Levi-Civita *symbol* (+-1).  Riemannian signature: on a
    6-manifold star.star = (-1)^{p(n-p)} = -1 on 3-forms (tested).
  * <A, A> = (1/p!) A_{a..} A^{a..}  (the "F^2" of physics normalization
    F_3^2 = (1/3!) F_{mnl} F^{mnl}).

Complex forms are supported transparently (jnp complex dtype); the metric
must be real.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as _np

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Levi-Civita symbol and antisymmetrization
# --------------------------------------------------------------------------

def levi_civita(n: int) -> _np.ndarray:
    """Levi-Civita symbol as a dense (n,)*n NUMPY array of {-1, 0, +1}.
    Kept in numpy (not jnp) so the module-level cache never stores a traced
    value — it enters jit traces as a constant."""
    eps = _np.zeros((n,) * n)
    for perm in itertools.permutations(range(n)):
        p = list(perm)
        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if p[i] > p[j])
        eps[perm] = -1 if inv % 2 else 1
    return eps


_EPS_CACHE: dict[int, _np.ndarray] = {}


def _eps(n: int) -> _np.ndarray:
    if n not in _EPS_CACHE:
        _EPS_CACHE[n] = levi_civita(n)
    return _EPS_CACHE[n]


_ABC = "abcdefghijklmnopqrstuvwxyz"


def alt(T: jnp.ndarray) -> jnp.ndarray:
    """Antisymmetrize a rank-p tensor over all its indices (with 1/p!)."""
    p = T.ndim
    if p <= 1:
        return T
    out = jnp.zeros_like(T)
    for perm in itertools.permutations(range(p)):
        inv = sum(1 for i in range(p) for j in range(i + 1, p)
                  if perm[i] > perm[j])
        sign = -1.0 if inv % 2 else 1.0
        out = out + sign * jnp.transpose(T, perm)
    return out / math.factorial(p)


# --------------------------------------------------------------------------
# d, wedge, star, contractions
# --------------------------------------------------------------------------

def dform(A_fn: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Exterior derivative at x of the p-form field A_fn.
    (dA)_{m a1..ap} = (p+1) d_[m A_{a1..ap]}."""
    J = jax.jacfwd(A_fn)(x)             # (n,)*p + (deriv index,)
    D = jnp.moveaxis(J, -1, 0)          # derivative index first
    p1 = D.ndim
    return p1 * alt(D)


def wedge(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """(A ^ B)_{..} = (p+q)!/(p!q!) Alt(A x B)."""
    p, q = A.ndim, B.ndim
    T = jnp.tensordot(A, B, axes=0)
    return (math.factorial(p + q) // (math.factorial(p) * math.factorial(q))
            ) * alt(T)


def hodge(A: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Hodge dual of the p-form A w.r.t. metric g (both at one point).
    (star A)_{b1..bq} = (1/p!) sqrt|g| A^{a1..ap} eps_{a1..ap b1..bq}."""
    n = g.shape[0]
    p = A.ndim
    ginv = jnp.linalg.inv(g)
    Aup = A
    for _ in range(p):
        # raise the first index, cycle it to the back; after p steps all
        # indices are raised and the original order is restored
        Aup = jnp.moveaxis(jnp.tensordot(ginv, Aup, axes=(1, 0)), 0, p - 1)
    sq = jnp.sqrt(jnp.linalg.det(g))
    eps = jnp.asarray(_eps(n), dtype=A.dtype)
    src = _ABC[:p] + _ABC[p:n]
    out = _ABC[p:n]
    return sq / math.factorial(p) * jnp.einsum(
        f"{_ABC[:p]},{src}->{out}", Aup, eps)


def form_norm2(A: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """<A, A> = (1/p!) A_{a..} A^{a..}  (complex forms: A bar-A)."""
    p = A.ndim
    ginv = jnp.linalg.inv(g)
    Aup = A
    for _ in range(p):
        Aup = jnp.moveaxis(jnp.tensordot(ginv, Aup, axes=(1, 0)), 0, p - 1)
    axes = list(range(p))
    return jnp.tensordot(jnp.conj(A), Aup, axes=(axes, axes)) \
        / math.factorial(p)
