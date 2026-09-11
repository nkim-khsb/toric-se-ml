"""Place the dP2 negative control on the displaced-Reeb curve.

MSY show the Futaki invariant is the first variation of the volume functional,
which is strictly convex with a unique critical point, so "no Kaehler-Einstein
metric at the regular Reeb vector" and "the minimiser is not the regular Reeb
vector" are the same statement.  The dP2 negative control is therefore an
instance of solving at a displaced Reeb vector, and its floor should sit on the
curve measured on the unobstructed dP3 cone.  This script computes the volume
excess on both and forms the ratio the paper quotes.  Read-only, no training.

Usage: PYTHONPATH=. python3 experiments/dp2/floor_vs_volume_excess.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from sugrasol.cone import B_DP2_REG, dp2, dp3, vol_Y

jax.config.update("jax_enable_x64", True)

# closed-form minimiser, = MSY eq. (3.48) in our lattice basis
B3_DP2_STAR = 3.0 * (19.0 - 3.0 * np.sqrt(33.0)) / 16.0


def relative_excess(cone, b, b_star):
    """vol(Y_b)/vol(Y_{b*}) - 1, the measure the paper uses."""
    return float(vol_Y(cone, b) / vol_Y(cone, b_star) - 1.0)


def dp2_excess():
    c = dp2()
    b_star = jnp.array([3.0, 0.0, B3_DP2_STAR])
    return relative_excess(c, B_DP2_REG, b_star), b_star


def dp3_ray(deltas=(0.05, 0.15, 0.30)):
    c = dp3()
    b_star = jnp.array([3.0, 0.0, 0.0])
    return [relative_excess(c, jnp.array([3.0, d, 0.0]), b_star) for d in deltas]


if __name__ == "__main__":
    e2, b_star = dp2_excess()
    g = float(jax.grad(lambda x: vol_Y(dp2(), jnp.array([3.0, 0.0, x])))(B3_DP2_STAR))
    print(f"dP2  b3* = {B3_DP2_STAR:.10f}   dVol/db3 there = {g:.2e}  (vanishes: it is the minimiser)")
    print(f"dP2  relative volume excess at the regular Reeb = {e2:.4e}")
    for floor in (4.2e-2, 4.0e-2):
        print(f"     floor {floor:.1e}  ->  floor/excess = {floor/e2:.3f}")
    print("\ndP3 ray, same convention (paper quotes 2.26-2.69):")
    for d, e, f in zip((0.05, 0.15, 0.30), dp3_ray(), (1.20e-3, 1.06e-2, 4.00e-2)):
        print(f"     delta={d:<5} excess={e:.4e}  floor={f:.2e}  ratio={f/e:.3f}")
