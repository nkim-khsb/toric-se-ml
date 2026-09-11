"""Re-score the dP2 b* Abreu number with the estimator section 2.3 requires.

The paper quotes "S = 12 to 0.002%" for the Sasaki-Einstein metric at the
pentagon's irregular b*.  That 0.002% is a SAMPLE MAXIMUM (note-dp2.tex,
"maximum deviation of 1.9e-5"), and section 2.3 rules that estimator out: a
maximum over finitely many points is biased low and does not settle as the grid
is refined, because the worst deviation sits near the polygon boundary.  On dP3
the same replacement moved the worst deviation from 0.011% to 0.019%, a factor
1.7, and section 4 has quoted worst/mean on the 240x240 grid ever since.  No
dP2 script had called abreu_grid, so this does.

The fit is not redone here in a new parametrization: se_potential() is the same
refit the harmonic-form work uses, and its own gate (0) requires it to
reproduce the held-out 6.0e-14 of tab:dp2ladder at degree 14 before anything is
reported.  abreu_grid wants psi and se_potential returns u = G_can + psi, so
psi is recovered by subtracting G_can, which is exact and not a second fit.

PRE-REGISTERED (fixed before running):
  (A) the fit gate: held-out within a factor 2 of 6.0e-14.  Otherwise stop.
  (B) grid max >= the 0.002% sample maximum, since a sample maximum is biased
      low.  By the dP3 precedent expect roughly 0.002-0.006%.
  (C) grid mean well below the grid max, and it is the mean the text should
      compare targets by (section 2.3).
  (D) if the grid max comes out BELOW 0.002%, that is the interesting outcome:
      it would say the old sample maximum was not biased low here, and the
      reason would have to be found before either number is quoted.

Usage: PYTHONPATH=. python experiments/dp2/abreu_grid_bstar.py
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harmonic_forms_dp2 import se_potential          # noqa: E402
from sugrasol.laplacian import abreu_grid            # noqa: E402

PAPER_SAMPLE_MAX_PCT = 0.002       # what the paper quotes, as a sample maximum
PAPER_HELD = 6.0e-14

print("== dP2 b*: Abreu S=12 on the 240x240 grid ==\n", flush=True)
ch, u, held = se_potential()

gate_a = 0.5 <= held / PAPER_HELD <= 2.0
print(f"  (A) fit gate: held-out {held:.3e} vs paper {PAPER_HELD:.1e}  "
      f"-> {'PASS' if gate_a else 'FAIL'}", flush=True)
if not gate_a:
    print("\n  STOP: this is not the solution the paper reports.")
    raise SystemExit(1)


def g_can(s):
    lt = ch.cone.ells(ch.t0 + s @ ch.f)
    return 0.5 * jnp.sum(lt * jnp.log(lt))


psi = lambda s: u(s) - g_can(s)                       # exact inverse of u

# sanity: psi must be small against G_can, as it is on every target
smp = np.asarray(ch.verts_s).mean(0)
print(f"  psi/G_can at the centroid: {float(psi(jnp.asarray(smp))):.3e} / "
      f"{float(g_can(jnp.asarray(smp))):.3e}", flush=True)

out = abreu_grid(psi, ch)
print(f"\n  grid {out['n_points']:,} interior points of 240x240")
print(f"  worst |S/12 - 1| = {out['max_pct']:.4f}%")
print(f"  mean  |S/12 - 1| = {out['mean_pct']:.4f}%")
print(f"  paper's sample maximum was {PAPER_SAMPLE_MAX_PCT}%  -> "
      f"grid max is {out['max_pct']/PAPER_SAMPLE_MAX_PCT:.2f}x it")

np.savez(Path(__file__).resolve().parent / "abreu_grid_bstar.npz",
         max_pct=out["max_pct"], mean_pct=out["mean_pct"],
         n_points=out["n_points"], held=held, degree=14)

if out["max_pct"] >= PAPER_SAMPLE_MAX_PCT:
    print("\n  => (B) as expected: the sample maximum was biased low.")
else:
    print("\n  => (D) UNEXPECTED: grid max below the sample maximum. Explain "
          "before quoting either.")
