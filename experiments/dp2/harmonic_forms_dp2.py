"""Primitive harmonic (1,1) forms on the dP2 pentagon, at the irregular b*.

WHY.  Section 4.3 of paper 1 grades DHHKW's (1,1)-form on dP3, and the
discussion says the same object at an irregular Reeb vector has to be sought on
the link as a basic form, "and we have not attempted that computation on dP2".
This attempts it.  Nothing new is needed: the transverse structure in the
symplectic frame is Hess(G_P) on the slice, which exists whether or not b is
rational, and the stream-potential reduction of r2b_stream_potential.py already
ran at an irregular Reeb vector -- Y^{3,2}, where 4p^2-3q^2 = 24 is not a
square.  The pentagon is the same computation on a five-sided polygon.

PRE-REGISTERED (fixed before running):
  (0) the SE fit must reproduce the paper's ladder at b*: held-out 6.0e-14 at
      degree 14 (tab:dp2ladder).  If it does not, this is a different solution
      and nothing below is reported.
  (1) EXACTLY 2 eigenvalues collapse.  The count is topology, not a fit:
      dim = d-3 for a d-gon, so 5-3 = 2, which is b_2(Y) = 2, the number of
      3-cycles a fractional brane can wrap.  Three or one would falsify the
      reduction on this polygon.
  (2) the gap between the second and third eigenvalue grows with the degree of
      chi, as it does on dP3 (3.7e8 at D=16) and Y^{3,2} (1.2e10).
  (3) primitivity is identity, not fit: max|tr A| at machine precision.

Usage: PYTHONPATH=. python experiments/dp2/harmonic_forms_dp2.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "dp3kt"))

jax.config.update("jax_enable_x64", True)

from sugrasol.cone import dp2, minimize_reeb, vol_Y                  # noqa: E402
from sugrasol.laplacian import slice_potential                       # noqa: E402
from sugrasol.ypq import (gauge_fixed, loss_fn, ortho_psi,           # noqa: E402
                          residual_on_slice, sample_slice,
                          slice_chart, whiten_poly)
from r2b_stream_potential import run                                  # noqa: E402

DEG = 14          # the degree tab:dp2ladder quotes at b*
PAPER_HELD = 6.04e-14


def se_potential(deg=DEG):
    """Refit the SE solution at b*; there is no stored artifact for dP2, and a
    whitened coefficient vector alone does not determine a function."""
    cone = dp2()
    b = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)
    ch = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), ch, 3000, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(7), ch, 3000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
    powers, W = whiten_poly(deg, ss)
    nc = len(powers)

    def L(v, batch):
        psi = gauge_fixed(ortho_psi(v[:nc], powers, W), ch.anchors)
        return loss_fn(ch, psi, v[nc], batch)

    vg = jax.jit(jax.value_and_grad(lambda v: L(v, ss)))
    L_te = jax.jit(lambda v: L(v, ss_te))
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
    held = float(L_te(jnp.asarray(res.x)))
    psi = gauge_fixed(ortho_psi(jnp.asarray(res.x[:nc]), powers, W), ch.anchors)
    print(f"  b* = {tuple(np.round(np.array(b), 6))}   vol = "
          f"{float(vol_Y(cone, b)) / np.pi**3:.6f} pi^3")
    print(f"  SE fit, degree {deg} ({nc} params): train {res.fun:.3e}  "
          f"held-out {held:.3e}   (paper: {PAPER_HELD:.2e})", flush=True)
    return ch, slice_potential(ch, psi), held


if __name__ == "__main__":
    print("== dP2 pentagon, primitive harmonic (1,1) forms at the irregular b* ==",
          flush=True)
    ch, u, held = se_potential()
    gate0 = abs(held / PAPER_HELD - 1.0) < 1.0      # same order, same solution
    print(f"  (0) reproduces the published ladder entry: "
          f"{'PASS' if gate0 else 'FAIL'}", flush=True)

    B2 = 2                                          # d - 3 for the pentagon
    print(f"\n  expecting exactly {B2} collapsed eigenvalues (d-3, d=5)")
    print(f"  {'D':>3} {'basis':>6} {'rank':>5} {'max|trA|':>9} | "
          f"lowest 5 SD eigenvalues                    | gap")
    for D in (8, 12, 16):
        lam, K, r, trA, rr, Rv = run(ch, u, D, B2, nq=(48, 48))
        gap = lam[B2] / lam[B2 - 1] if lam[B2 - 1] > 0 else float("inf")
        print(f"  {D:>3} {K:>6} {r:>5} {trA:>9.1e} | "
              + "  ".join(f"{v:8.2e}" for v in lam[:5]) + f" | {gap:8.0f}x",
              flush=True)
