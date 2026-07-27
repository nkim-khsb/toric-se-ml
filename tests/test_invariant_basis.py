"""The D6-invariant basis must span the whole invariant space, not part of it.

The Reynolds-averaged monomials of `whiten_sym_poly` are severely collinear
(singular values spanning ten orders on the dP3 hexagon), so a relative SVD
tolerance silently decides how many genuine invariant directions survive.  With
the old default it kept 11 of 14 at degree 14 and 17 of 21 at degree 18, which
cost a factor 25 in held-out residual and 3.5 in DHHKW's Einstein-condition
error (log.md 2026-07-26).  That is the third appearance of this conditioning
trap, so the count is now checked against a closed form rather than trusted.

Ground truth is exact and independent of the SVD: for a rank-2 reflection group
of order 2m the invariant ring is free on generators of degree 2 and m, and for
the hexagon (m = 6) those are DHHKW's U and V (their eq. (5.1)).  Molien's
series reproduces that count for any finite matrix group, and is what
`invariant_dimension` computes.
"""
import warnings

import jax
import pytest

from sugrasol.cone import B_CONIFOLD, B_DP3, conifold, dp3
from sugrasol.ypq import (dihedral_matrices, invariant_dimension, sample_slice,
                          slice_chart, whiten_sym_poly)

jax.config.update("jax_enable_x64", True)


def _uv_count(degree, m=6):
    """#{U^i V^j : 2 <= 2i + m j <= degree}, deg U = 2, deg V = m."""
    return sum(1 for i in range(degree // 2 + 1) for j in range(degree // m + 1)
               if 2 <= 2 * i + m * j <= degree)


def test_molien_matches_the_UV_count_on_the_hexagon():
    """Molien vs the free-generator count -- two independent routes."""
    ch = slice_chart(dp3(), B_DP3)
    group = dihedral_matrices(ch.verts_s)
    assert len(group) == 12
    for degree in (6, 10, 14, 18, 22):
        assert invariant_dimension(group, degree) == _uv_count(degree)
    assert invariant_dimension(group, 14) == 14      # the numbers that bit us
    assert invariant_dimension(group, 18) == 21


def test_default_tolerance_recovers_the_full_rank():
    """Regression on the artifacts: dp3_G_deg14/18.npz are fits in the FULL
    invariant space, 14- and 21-dimensional.  Anything less is the truncation
    bug returning."""
    ch = slice_chart(dp3(), B_DP3)
    group = dihedral_matrices(ch.verts_s)
    ss = sample_slice(jax.random.PRNGKey(1), ch, 8192, eps=2e-3)
    for degree, expect in ((14, 14), (18, 21)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")            # the guard must not fire
            _, W, _ = whiten_sym_poly(degree, ss, group)
        assert W.shape[1] == expect


def test_the_guard_fires_on_the_old_tolerance():
    """The point of the guard is that a truncation cannot pass unnoticed."""
    ch = slice_chart(dp3(), B_DP3)
    group = dihedral_matrices(ch.verts_s)
    ss = sample_slice(jax.random.PRNGKey(1), ch, 2048, eps=2e-3)
    with pytest.warns(RuntimeWarning, match="11 of the 14"):
        _, W, _ = whiten_sym_poly(14, ss, group, tol=1e-9)
    assert W.shape[1] == 11


def test_no_check_rather_than_a_wrong_check():
    """dihedral_matrices assumes the centroid is the origin; the conifold square
    is off-centre, so its 'symmetries' are affine and the linear maps built from
    them are not a group.  invariant_dimension must detect that (non-integral
    Molien dimensions) and decline, rather than emit a bogus count."""
    ch = slice_chart(conifold(), B_CONIFOLD)
    assert abs(ch.verts_s.mean(0)).max() > 1e-3       # genuinely off-centre
    assert invariant_dimension(dihedral_matrices(ch.verts_s), 14) is None
