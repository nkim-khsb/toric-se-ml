"""Toric geometry in symplectic (action-angle) coordinates.

Convention (Abreu, math/0004122; Guillemin):
  Delzant polytope  P = { x in R^n : l_a(x) = <x, v_a> + c_a >= 0,  a = 1..d },
  v_a = primitive inward-pointing facet normals (integer vectors).

  Guillemin symplectic potential:
      G_can(x) = (1/2) * sum_a l_a(x) * log l_a(x)

  Toric Kaehler metric on P x T^n (Abreu eq. (2.2)):
      g = G_ij dx^i dx^j + G^ij dphi_i dphi_j,   G_ij = Hess(G), G^ij = (G_ij)^{-1}

General ansatz: G = G_can + psi, with psi smooth on P (boundary behaviour of
G_can is what encodes the manifold; psi must not spoil it).

NOTE (CLAUDE.md rule 6): any *normalization-sensitive* quantity (Ricci
potential F, Einstein constant, Reeb slice) lives in losses.py and must be
checked against the source paper before use. This module is normalization-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Polytope:
    """l_a(x) = <x, normals[a]> + offsets[a] >= 0."""

    normals: jnp.ndarray  # (d, n) inward primitive normals
    offsets: jnp.ndarray  # (d,)

    @property
    def dim(self) -> int:
        return self.normals.shape[1]

    def ells(self, x: jnp.ndarray) -> jnp.ndarray:
        """All facet functions l_a(x), shape (d,)."""
        return self.normals @ x + self.offsets

    def is_interior(self, x: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
        return jnp.all(self.ells(x) > eps)


def square(side: float = 1.0) -> Polytope:
    """[0, side]^2 (generic Delzant square, origin corner)."""
    normals = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    offsets = jnp.array([0.0, side, 0.0, side])
    return Polytope(normals, offsets)


def centered_square(s: float) -> Polytope:
    """[-s, s]^2 centered at 0 — D4-symmetric frame (gamma = 0 in DHHKW 3.25).

    T^{1,1} transverse base P^1 x P^1 with Ric = 6 g^T: s = 1/Lambda = 1/6
    (per-factor closed-form check: Guillemin on [-s,s] is KE iff Lambda*s = 1).
    """
    normals = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    offsets = jnp.array([s, s, s, s])
    return Polytope(normals, offsets)


def guillemin_potential(p: Polytope, x: jnp.ndarray) -> jnp.ndarray:
    """G_can(x) = (1/2) sum_a l_a log l_a  (defined for x in the interior)."""
    l = p.ells(x)
    return 0.5 * jnp.sum(l * jnp.log(l))


def sym_hessian(potential, x: jnp.ndarray) -> jnp.ndarray:
    """G_ij = Hessian of the symplectic potential at x."""
    return jax.hessian(potential)(x)


@partial(jax.jit, static_argnums=(0,))
def metric_blocks(potential, x: jnp.ndarray):
    """Return (G_ij, G^ij): the xx-block and phiphi-block of the toric metric."""
    h = sym_hessian(potential, x)
    return h, jnp.linalg.inv(h)


def total_potential(p: Polytope, psi):
    """G = G_can + psi, psi: R^n -> R (e.g. a neural network)."""

    def G(x):
        return guillemin_potential(p, x) + psi(x)

    return G
