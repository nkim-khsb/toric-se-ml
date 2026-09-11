"""DHHKW's published harmonic (1,1)-form theta_2, tested on our dP3 metric.

Pins experiments/dp3/dhhkw_theta_compare.py (see note-dhhkw-theta.md).  The
refit and order-ladder blocks of that script are too slow for CI; what is pinned
here is the part that needs no fitting, which is also the part carrying the
ground truth:

  * the coordinate bridge x = 3 s between their polytope and ours, exactly;
  * their eq. (6.13) coefficients, pushed through OUR metric, satisfy their
    harmonicity condition (3.46) with the exactly-known constant 2/3;
  * the same mu on the canonical Guillemin metric does NOT -- the CONSTANCY is
    the psi-sensitive content, and without this anti-test the check would pass
    on a metric that is not Einstein;
  * the protected omega anchor (6.14), reached through the D_6 orbit.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "dp3"))
import dhhkw_theta_compare as T  # noqa: E402

from sugrasol.artifacts import load_psi_dp3  # noqa: E402
from sugrasol.ypq import sample_slice  # noqa: E402

N, EPS = 3000, 1e-3


def _samples():
    return sample_slice(jax.random.PRNGKey(5), T.CHART, N, eps=EPS)


def _stats(L, mu, pts):
    v = T.TO_THEM * np.asarray(jax.jit(jax.vmap(L(mu)))(pts))
    return float(v.mean()), float(np.sqrt(((v - v.mean()) ** 2).mean()))


def test_coordinate_bridge_is_exact():
    """Their polytope is l_a = 1 + v_a.x, ours l_a = 1/3 + v_a.s, so x = 3 s and
    their vertices land on the integer points they quote (|x_i| <= 1)."""
    x = T.SCALE * np.asarray(T.CHART.verts_s)
    assert np.allclose(x, np.round(x), atol=1e-12), x
    assert np.abs(x).max() <= 1.0 + 1e-12
    # their c_nm = c_mn is the stabilizer of their facet a = 2
    assert np.allclose(T.V2[T.A_FACET], T.V2[T.A_FACET][::-1])


def test_published_theta_is_harmonic_on_our_metric():
    """DHHKW (6.13) on our metric: (3.46) constant, exactly 2/3 by (6.15).
    Their own fit reported 0.6672."""
    pts = _samples()
    for tag in ("deg14", "deg18"):
        psi, ch, _ = load_psi_dp3(
            Path(__file__).resolve().parents[1]
            / f"experiments/dp3smooth/dp3_G_{tag}.npz")
        mean, rms = _stats(T.laplacian_op(ch, psi), T.mu_dhhkw, pts)
        print(f"{tag}: constant {mean:.6f} (exact 2/3), spread rms {rms:.2e}")
        assert abs(mean - 2.0 / 3.0) < 2e-3
        assert rms < 2e-2


def test_constancy_is_psi_sensitive():
    """The anti-test that gives the grade teeth.  On the canonical Guillemin
    metric (psi = 0, not Einstein) the spread blows up by more than an order of
    magnitude, while the MEAN barely moves -- the constant is a boundary
    (protected) quantity, the constancy is not."""
    pts = _samples()
    psi, ch, _ = load_psi_dp3(Path(__file__).resolve().parents[1]
                              / "experiments/dp3smooth/dp3_G_deg18.npz")
    m_ke, r_ke = _stats(T.laplacian_op(ch, psi), T.mu_dhhkw, pts)
    m_0, r_0 = _stats(T.laplacian_op(ch, lambda s: 0.0), T.mu_dhhkw, pts)
    print(f"KE: mean {m_ke:.6f} rms {r_ke:.2e};  "
          f"Guillemin: mean {m_0:.6f} rms {r_0:.2e};  ratio {r_0 / r_ke:.0f}x")
    assert r_0 / r_ke > 20.0                     # measured ~60x
    assert abs(m_0 - 2.0 / 3.0) < 1e-2           # the mean is nearly protected


def test_protected_omega_anchor():
    """omega = (1/2) sum_a theta_a (6.14): the D_6 orbit sum of mu_2 has
    (3.46)-constant exactly 2.  Unlike the theta_a constant, this one does move
    on a wrong metric, by two orders."""
    pts = _samples()
    psi, ch, _ = load_psi_dp3(Path(__file__).resolve().parents[1]
                              / "experiments/dp3smooth/dp3_G_deg18.npz")
    m_ke, _ = _stats(T.laplacian_op(ch, psi), T.mu_omega, pts)
    m_0, _ = _stats(T.laplacian_op(ch, lambda s: 0.0), T.mu_omega, pts)
    print(f"omega anchor: KE {m_ke:.6f}, Guillemin {m_0:.6f} (exact 2)")
    assert abs(m_ke - 2.0) < 2e-3
    assert abs(m_0 - 2.0) > 5e-3
