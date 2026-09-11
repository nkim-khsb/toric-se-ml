"""The blue curve of the Y^{p,q} invariant figure rests on two facts."""
import numpy as np
import pytest

from experiments.ypq.gmsw_curve_check import curve_facts


@pytest.mark.parametrize("p,q", [(2, 1), (3, 2)])
def test_invariants_depend_on_y_alone(p, q):
    """Cohomogeneity one: this is why the image of a 2d polygon is a curve."""
    ind, _, _, _ = curve_facts(p, q)
    assert np.max(ind) < 1e-11


@pytest.mark.parametrize("p,q", [(2, 1), (3, 2)])
def test_cone_scaling_and_monotonicity(p, q):
    """r^-4 and r^-6 fix the evaluation; monotonicity makes Phi2(Phi1) single
    valued, which is what the interpolated deviation in the figure needs."""
    _, pw, mono, _ = curve_facts(p, q)
    assert np.allclose(pw, [-4.0, -6.0], atol=1e-9)
    assert mono


def test_thickness_estimator_needs_detrending():
    """Guard the estimator, not the physics: on a locus that is exactly a curve
    the detrended thickness must vanish, while the raw bin spread returns the
    slope.  Getting this backwards once reported a 40% thick locus that was not
    there."""
    import numpy as np
    from experiments.ypq.gmsw_curve_check import locus_thickness
    x = np.logspace(0.5, 3.0, 4000)
    inv = np.stack([x, 2.7 * x ** 1.5], axis=1)      # an exact curve
    med, mx = locus_thickness(inv)
    assert mx < 1e-6
    raw = np.ptp(inv[:200, 1]) / np.median(inv[:200, 1])
    assert raw > 0.2
