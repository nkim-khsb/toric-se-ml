"""DHHKW's Einstein-condition error D on our learned dP3 metric.

D = sqrt(1/4 (R_mn - g_mn)(R^mn - g^mn))  is DHHKW's eq. (5.4)
(hep-th/0703057), defined at THEIR normalization Ric = g.  Ours is the
doubled-slice Ric = 3g, and Ric(c g) = Ric(g), so D applies verbatim to 3g.
The paper quotes D to compare like with like against their eighteenth-order
expansion (~1e-3 at the hexagon vertex); these tests pin the convention with a
closed form and lock the quoted numbers (CLAUDE.md rules 2, 7).

Production numbers and the pre-registered verdict: experiments/dp3/dhhkw_D_compare.py.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.curvature import ricci, toric_metric
from sugrasol.artifacts import load_psi_dp3
from sugrasol.laplacian import slice_potential
from sugrasol.ypq import slice_chart

jax.config.update("jax_enable_x64", True)
NPZ14 = Path(__file__).resolve().parents[1] / "experiments/dp3smooth/dp3_G_deg14.npz"


def _D(u, s):
    """DHHKW's D at slice point s, for the toric metric with potential u."""
    gp_fn = lambda x: 3.0 * toric_metric(u, 2)(x)          # noqa: E731
    x = jnp.concatenate([jnp.asarray(s), jnp.zeros(2)])
    gp = gp_fn(x)
    T = ricci(gp_fn, x) - gp
    gpi = jnp.linalg.inv(gp)
    return float(jnp.sqrt(0.25 * jnp.einsum("ab,ac,bd,cd->", T, gpi, gpi, T)))


def test_conifold_fixes_the_normalization_and_vanishes():
    """CP^1 x CP^1 (psi = 0) is Kaehler-Einstein in closed form, so it pins
    BOTH halves of the convention: Ric = 3g at machine precision in our
    doubled slice (which is what makes 3g DHHKW's Ric = g), and D = 0 for the
    rescaled metric.  D = 0 alone would not pin the factor 3 -- it holds for
    any constant rescaling -- so the Ric = 3g check is the load-bearing one."""
    ch = slice_chart(conifold(), B_CONIFOLD)
    u = slice_potential(ch, lambda s: 0.0)
    g_fn = toric_metric(u, 2)
    for s in ([0.0, 0.0], [0.05, -0.02], [-0.3, -0.4]):
        x = jnp.concatenate([jnp.asarray(s) + jnp.asarray(ch.verts_s).mean(0),
                             jnp.zeros(2)])
        assert float(jnp.max(jnp.abs(ricci(g_fn, x) - 3.0 * g_fn(x)))) < 1e-12
        assert _D(u, x[:2]) < 1e-12


def test_dp3_D_localizes_in_the_corner_and_matches_the_quoted_value():
    """On the learned deg-14 metric (the paper's fourteen-parameter headline),
    D grows from the hexagon centre towards the vertex and plateaus at
    3.94e-4 -- a factor 2.5 below DHHKW's eighteenth-order expansion at the
    vertex, which is exactly the claim the paper now makes.  The ray is
    s(x1) = x1 * vertex, matching their x2 = 0 line (on which their invariant
    U = x1^2); the six vertices are one D6 orbit, so D must agree on all six."""
    psi, ch, _ = load_psi_dp3(NPZ14)
    u = slice_potential(ch, psi)
    verts = np.array(ch.verts_s)

    centre, vertex = _D(u, [0.0, 0.0]), _D(u, 0.99999 * verts[3])
    assert centre < 1e-4                       # smallest at the origin
    assert vertex > 10 * centre                # localized in the corner
    assert abs(vertex / 3.936e-4 - 1.0) < 0.01  # the number quoted in the paper

    six = [_D(u, 0.99 * v) for v in verts]     # one D6 orbit -> one value
    assert max(six) / min(six) - 1.0 < 1e-6
