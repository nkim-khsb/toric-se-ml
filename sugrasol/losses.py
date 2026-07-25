"""Toric Kaehler-Einstein Monge-Ampere residual, symplectic side.

Source (verified 2026-07-11 against the paper text — CLAUDE.md rule 6):
DHHKW, hep-th/0703057:

  complex side, eq. (3.24):   ln det F_ij = -2 L f + gamma . u - c
  symplectic side, eq. (3.25):
      ln det G_ij = -2 L g + (dg/dx) . (2 L x - gamma) + c
  h-form + rho_can: eqs. (3.27), (3.28).  Smoothness cond: (3.14).

(3.24) <-> (3.25) via Legendre duality (f = <x,y> - g, y = dg/dx,
Hess_y f = [Hess_x g]^{-1}); checked symbolically. We implement (3.25):
it lives directly in polytope coordinates x — no Legendre transform, and the
Guillemin BC sits in g_can (ansatz g = g_can + psi, cf. their g_can + h).

Route decision: symplectic 2nd-order MA (3.25), NOT Abreu's 4th-order scalar
curvature equation (higher derivatives, weaker condition).

Normalization chain for T^{1,1} transverse base P^1 x P^1 (fixed by two
independent checks — 1d closed form and area/Ric matching):
  Ric = 2(n+1) g^T = 6 g^T (n = 2)  =>  Lambda = 6,
  each P^1 factor: interval [-s, s] with  Lambda * s = 1  =>  s = 1/6,
  polytope = [-1/6, 1/6]^2,  gamma = 0 (D4 symmetry),  c = 2 ln s.

Gauge structure (important):
  - constant mode of psi: NOT flat — fixed by the -2*Lambda*psi term.
  - linear mode of psi: EXACT flat direction (toric automorphism / translation
    in y). Excluded by the D4-symmetrized ansatz (nets.symmetrized_psi).
    => invariant layer is a uniqueness device, not an optimization.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class KEProblem(NamedTuple):
    """Data of eq. (3.25): residual = ln det Hess g + 2Lg - dg.(2Lx - gamma) - c."""

    lam: float  # Lambda (Einstein constant, Ric = Lambda * g... their 2L conv.)
    gamma: jnp.ndarray  # affine datum, 0 for symmetric polytopes
    c: float  # additive constant (gauge-linked to psi(x0); fixed analytically here)


def ke_residual(problem: KEProblem, G: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Pointwise MA residual of DHHKW eq. (3.25) for symplectic potential G.

    Zero iff g is Kaehler-Einstein with Ric = 2*Lambda*... (their convention:
    (3.24) is Ric = Lambda * omega with the 2L f factor; Lambda enters as 2L).
    """
    val = G(x)
    grad = jax.grad(G)(x)
    hess = jax.hessian(G)(x)
    sign, logdet = jnp.linalg.slogdet(hess)
    lam, gamma, c = problem.lam, problem.gamma, problem.c
    return logdet + 2.0 * lam * val - grad @ (2.0 * lam * x - gamma) - c


def ke_loss(problem: KEProblem, G: Callable, xs: jnp.ndarray) -> jnp.ndarray:
    """Mean-square residual over Lebesgue (= symplectic) samples."""
    r = jax.vmap(lambda x: ke_residual(problem, G, x))(xs)
    return jnp.mean(r**2)


def smoothness_indicator(polytope, G: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """det(Hess G) * prod_a l_a — must extend smooth & positive to dP
    (DHHKW eq. (3.14)); diagnostic for boundary-condition damage by psi."""
    hess = jax.hessian(G)(x)
    return jnp.linalg.det(hess) * jnp.prod(polytope.ells(x))
