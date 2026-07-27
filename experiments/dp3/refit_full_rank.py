"""Refit and REPLACE the persisted dP3 metrics at full invariant rank.

Why (log.md 2026-07-26).  The hexagon's D_6 invariant ring is free on two
generators of degree 2 and 6, so the invariant space is 14-dimensional at
degree 14 and 21-dimensional at degree 18.  The Reynolds-averaged monomial
basis of ypq.whiten_sym_poly is severely collinear -- its singular values span
ten orders -- so the relative SVD tolerance decides how many genuine invariant
directions survive, and the previously persisted artifacts kept only 11 of 14
and 17 of 21.  invariant_rank_check.py showed the missing directions are NOT
inert: at full rank the held-out Monge-Ampere loss drops ~25x and DHHKW's
independent error measure D at the hexagon vertex drops ~3.5x, with held-out
tracking train (no overfitting).

This script rewrites dp3_G_deg14.npz and dp3_G_deg18.npz at tol = 1e-14, which
recovers exactly the theoretical dimension at both degrees (14 and 21; at
degree 14, tol 1e-12 and 1e-14 agree, so the value is not on a knife edge).
Everything else -- sample seeds, counts, boundary margin, optimizer settings --
is unchanged from the runs that produced the originals (log.md 2026-07-20).

The old files are recoverable from git (commit 597721f).  Downstream flux /
resolved / smooth rungs are NOT re-run: their bottlenecks are the facet
polynomial basis (~2%) and topology, not the background metric; log.md records
which of their numbers came from the superseded 11/17-direction metrics.

Usage: PYTHONPATH=. python experiments/dp3/refit_full_rank.py
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)

from sugrasol.cone import B_DP3, dp3                                # noqa: E402
from sugrasol.ypq import (dihedral_matrices, loss_fn,               # noqa: E402
                          residual_on_slice, sample_slice, slice_chart,
                          sym_ortho_psi, whiten_sym_poly)

TOL = 1e-14        # recovers the full invariant rank (14 at deg 14, 21 at 18)
RUNS = [(14, 2048, 14), (18, 8192, 21)]     # (degree, samples, expected dim)

ch = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(ch.verts_s)

for deg, nsamp, expect in RUNS:
    ss = sample_slice(jax.random.PRNGKey(1), ch, nsamp, eps=2e-3)
    ss_test = sample_slice(jax.random.PRNGKey(7), ch, nsamp, eps=2e-3)
    powers, W, grp = whiten_sym_poly(deg, ss, group, tol=TOL)
    nc = W.shape[1]
    assert nc == expect, f"degree {deg}: got dim {nc}, expected {expect}"

    def L(v, batch, nc=nc, powers=powers, W=W, grp=grp):
        return loss_fn(ch, sym_ortho_psi(v[:nc], powers, W, grp), v[nc], batch)

    vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
    r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    t0 = time.time()
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
    v = jnp.asarray(res.x)
    l_test = float(jax.jit(lambda v: L(v, ss_test))(v))
    print(f"deg {deg}: {nc} invariant params (space is {expect}-dim), "
          f"train {res.fun:.3e}, held-out {l_test:.3e}, "
          f"{res.nit} it, {time.time() - t0:.0f}s")

    npz = ROOT / "experiments" / "dp3smooth" / f"dp3_G_deg{deg}.npz"
    np.savez(npz, degree=deg, powers=np.asarray(powers), W=np.asarray(W),
             group=np.asarray(group), coeffs=np.asarray(v[:nc]),
             c=float(v[nc]), loss_train=res.fun, loss_test=l_test,
             note=f"deg-{deg} D6-invariant, FULL invariant rank {nc} "
                  f"(= dim of U^i V^j, deg U=2 deg V=6, 2<=2i+6j<={deg}); "
                  f"whiten tol={TOL:.0e}, {nsamp} samples, seeds 1/7. "
                  f"Supersedes the rank-truncated fit of 2026-07-20 "
                  f"(log.md 2026-07-26). G = cone_potential(slice_chart("
                  f"dp3(), B_DP3), sym_ortho_psi(coeffs, powers, W, group)).")
    print(f"   wrote {npz.name}")
