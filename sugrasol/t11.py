"""T^{1,1} = Y^{1,0} transverse KE problem — fully pinned constants.

Chain (all cross-checked 2026-07-11, see experiments/t11/log.md):
  SE_5: Ric = 4 g  =>  transverse KE: Ric^T = 6 g^T (lambda = 2n+2, n=2)
  base = P^1 x P^1, each factor area 2pi/3  =>  moment interval length 1/3
  polytope = [-1/6, 1/6]^2,  Lambda = 6,  gamma = 0,  c = 2 ln(1/6)
  volume chain: Area(P) = 1/9 => Vol(base) = (2pi)^2/9 = 4pi^2/9;
                eta-fiber length 4pi/3 => Vol(T^{1,1}) = 16 pi^3/27  [protected]

Cone-lift data (for the Y^{p,q} rung, NOT used by the transverse problem):
  moment cone facets v_a = (1, w_a), w_a in {(0,0),(1,0),(1,1),(0,1)},
  Reeb b = (3, 3/2, 3/2)  — TODO(verify vs MSY hep-th/0503183 before use).
"""
from __future__ import annotations

import jax.numpy as jnp

from .losses import KEProblem
from .toric import Polytope, centered_square

LAMBDA: float = 6.0
S: float = 1.0 / 6.0
VOL_T11: float = 16.0 * jnp.pi**3 / 27.0


def transverse_polytope() -> Polytope:
    return centered_square(S)


def transverse_problem() -> KEProblem:
    return KEProblem(
        lam=LAMBDA,
        gamma=jnp.zeros(2),
        c=2.0 * jnp.log(S),
    )
