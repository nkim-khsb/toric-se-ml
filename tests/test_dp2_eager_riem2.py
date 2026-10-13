"""Gates for experiments/dp2/eager_riem2_check.py (R. Eager, email thread
"Re: Numerical Sasaki-Einstein metrics", 2026-09-23): the standard cone-
curvature identity Riem^2_L5 = Riem^2_cone|_{r=1} + 40 (n=5 Einstein link),
checked against closed forms before it is trusted on the numerical dP2
metric. The dP2 production number itself (vs Nakayama-Okazaki arXiv:2603.
17452 eq. 9.41, 2166.94...) is left to the script -- it retrains the SE fit
and is not repeated here."""
import jax.numpy as jnp

from sugrasol.cone import B_CONIFOLD, b_ypq, conifold, vol_Y, vol_ypq, ypq
from sugrasol.laplacian import slice_potential
from sugrasol.ypq import slice_chart

from experiments.dp2.eager_riem2_check import (egr_riem2_ypq, fit_poly_psi,
                                                l5_average_riem2,
                                                riem2_pointwise)


def test_t11_conifold_reproduces_egr_table():
    """psi = 0 is exact for the conifold; T^{1,1} is homogeneous, so this
    must be EXACTLY constant, matching Eager-Gary-Roberts' table ratio
    2176/16 = 136 (arXiv:1011.5231, table CurvTable)."""
    chart = slice_chart(conifold(), B_CONIFOLD)
    u = slice_potential(chart, lambda s: 0.0)
    s = jnp.mean(chart.verts_s, axis=0) * 0.3
    assert abs(float(riem2_pointwise(chart, u, s)) - 136.0) < 1e-8


def test_y21_matches_egr_closed_form():
    """Non-constant (cohomogeneity-one) case: our own quick trained psi
    against Eager-Gary-Roberts' Y^{p,q} closed form."""
    cone, b = ypq(2, 1), b_ypq(2, 1)
    chart, u, held, train = fit_poly_psi(cone, b, deg=10, steps=6000)
    assert train < 1e-10 and held < 1e-10
    avg, _ = l5_average_riem2(chart, u, ngl=24)
    vol = float(vol_Y(cone, b))
    assert abs(vol / float(vol_ypq(2, 1)) - 1.0) < 1e-10
    rel = abs(avg * vol / egr_riem2_ypq(2, 1) - 1.0)
    assert rel < 1e-4
