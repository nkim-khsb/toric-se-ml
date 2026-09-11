"""The pentagon sits on the displaced-Reeb curve measured on dP3."""
import numpy as np

from experiments.dp2.floor_vs_volume_excess import dp2_excess, dp3_ray


def test_dp2_relative_excess():
    """Quoted in the negative-control section as 3.12e-2."""
    e, _ = dp2_excess()
    assert abs(e - 3.1179e-2) < 1e-5


def test_floor_over_excess_lies_in_the_dp3_grid_range():
    """1.35, inside the 1.1-3.8 the dP3 displacement gives over directions."""
    e, _ = dp2_excess()
    r = 4.2e-2 / e
    assert 1.1 < r < 3.8
    assert abs(r - 1.35) < 0.02


def test_dp3_ray_reproduces_the_quoted_band():
    """The paper quotes 2.26-2.69 along the ray."""
    floors = (1.20e-3, 1.06e-2, 4.00e-2)
    ratios = [f / e for e, f in zip(dp3_ray(), floors)]
    assert all(2.26 <= r <= 2.69 for r in ratios), ratios
