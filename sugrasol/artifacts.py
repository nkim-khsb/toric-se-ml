"""Readers for the persisted dP3 metric artifacts.

dp3_G_deg14.npz / dp3_G_deg18.npz ARE the metrics quoted in paper I: every dP3
number there is recomputed from them, never by retraining (which would silently
depend on the current default SVD tolerance -- the 2026-07-26 rank trap).

These two readers used to live in `dp3smooth.py`, which also carries the
non-toric rung-C smoothing machinery and its `legendre` dependency.  They are
split out here so that consuming the artifacts does not drag in that rung:
the toric release ships this module, not that one.  `dp3smooth` re-exports both
names, so existing imports keep working.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .cone import B_DP3, dp3
from .ypq import cone_potential, slice_chart, sym_ortho_psi

jax.config.update("jax_enable_x64", True)


def load_psi_dp3(npz_path):
    """Reconstruct the persisted D6-invariant slice potential psi_dP3.

    Returns (psi, chart, meta) with meta the raw npz record (degree, number of
    parameters, train/held-out loss)."""
    dat = np.load(npz_path)
    powers = [tuple(int(a) for a in row) for row in dat["powers"]]
    W = jnp.asarray(dat["W"])
    group = jnp.asarray(dat["group"])
    coeffs = jnp.asarray(dat["coeffs"])
    chart = slice_chart(dp3(), B_DP3)
    return sym_ortho_psi(coeffs, powers, W, group), chart, dat


def load_G_dp3(npz_path):
    """Reconstruct the persisted symplectic potential G_dP3 (+ constant c_D)."""
    psi, chart, dat = load_psi_dp3(npz_path)
    return cone_potential(chart, psi), float(dat["c"]), chart
