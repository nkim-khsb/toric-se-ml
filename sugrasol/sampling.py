"""Sampling on the moment polytope (toric measure).

Toric sampling (notes §2.3): Lebesgue on the polytope x uniform T^n — no
embedding, no Shiffman-Zelditch machinery needed. Rejection sampling from the
bounding box is exact for Lebesgue.

An eps-margin keeps samples away from facets where log l_a in G_can is
singular; the boundary condition is carried by the ansatz (Guillemin),
not by boundary collocation points. Facet neighbourhoods are measure-small;
if facet accuracy becomes an issue, switch to importance sampling, not to
penalty terms (CLAUDE.md rule 1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .toric import Polytope

jax.config.update("jax_enable_x64", True)


def bounding_box(p: Polytope, lo: jnp.ndarray, hi: jnp.ndarray):
    return lo, hi


def sample_polytope(
    key: jax.Array,
    p: Polytope,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    n_samples: int,
    eps: float = 1e-3,
    oversample: int = 4,
) -> jnp.ndarray:
    """Lebesgue-uniform points in {x : l_a(x) > eps}. Returns (n_samples, n).

    Rejection from the box [lo, hi]; oversample controls the batch factor.
    Raises if not enough accepted points (increase oversample).
    """
    n = p.dim
    m = oversample * n_samples
    u = jax.random.uniform(key, (m, n), dtype=jnp.float64)
    pts = lo + u * (hi - lo)
    ok = jax.vmap(lambda x: p.is_interior(x, eps))(pts)
    idx = jnp.nonzero(ok, size=m, fill_value=-1)[0]
    accepted = int(jnp.sum(ok))
    if accepted < n_samples:
        raise RuntimeError(
            f"rejection sampling: {accepted}/{n_samples}; raise oversample"
        )
    return pts[idx[:n_samples]]
