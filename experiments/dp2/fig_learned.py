"""The learned dP2 correction psi on the pentagon at b*, and its Abreu scalar.

WHY.  The paper pictures the learned psi on the hexagon (fig:dp3learned), which
is the positive control, and pictures the harmonic forms on both polygons
(fig:streampot).  It did not picture the metric it constructs: the pentagon's
psi at the irregular minimizer, which is the new object of the paper.  This is
the dP2 counterpart of experiments/dp3/figures.py, same two panels.

WHERE THE POTENTIAL COMES FROM.  se_potential() of harmonic_forms_dp2, which is
the refit the harmonic-form work and the Abreu grid already use -- there is no
stored artifact for dP2 -- so the figure cannot drift from the metric the paper
reports.  Its gate requires the held-out residual within a factor two of the
6.0e-14 the paper quotes, and this script stops if that fails.  psi is recovered
as u - G_can, which is exact and not a second fit.

THE COLOUR WINDOW is derived from the measured deviation, never hardcoded: on
the hexagon a fixed window tuned at degree 14 outlived the switch to degree 18
and left the caption quoting the wrong metric's accuracy (experiments/dp3/
figures.py).  The window printed below is what the caption must quote.

Usage: PYTHONPATH=. python experiments/dp2/fig_learned.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

jax.config.update("jax_enable_x64", True)

from harmonic_forms_dp2 import se_potential            # noqa: E402
from sugrasol.laplacian import abreu_scalar            # noqa: E402

DRAFT = Path(__file__).resolve().parents[2] / "writing" / "paper1-new"
PAPER_HELD = 6.0e-14
N = 240          # the grid of abreu_grid(), so the caption and section 5.2
                 # quote one number for one quantity


def _targets(name):
    here = Path(__file__).resolve().parent / name
    return [here] + ([DRAFT / name] if DRAFT.is_dir() else [])


ch, u, held = se_potential()
gate = 0.5 <= held / PAPER_HELD <= 2.0
print(f"fit gate: held-out {held:.3e} vs paper {PAPER_HELD:.1e} -> "
      f"{'PASS' if gate else 'FAIL'}")
if not gate:
    raise SystemExit("STOP: this is not the solution the paper reports.")


def g_can(s):
    lt = ch.cone.ells(ch.t0 + s @ ch.f)
    return 0.5 * jnp.sum(lt * jnp.log(lt))


psi = lambda s: u(s) - g_can(s)
S_of = abreu_scalar(ch, psi)

verts = np.asarray(ch.verts_s)
lo, hi = verts.min(0) - 0.02, verts.max(0) + 0.02
GX, GY = np.meshgrid(np.linspace(lo[0], hi[0], N), np.linspace(lo[1], hi[1], N))
pts = jnp.asarray(np.stack([GX.ravel(), GY.ravel()], axis=1))
t_of = lambda s: ch.t0 + s @ ch.f
inside = np.asarray(
    jax.vmap(lambda s: ch.cone.is_interior(t_of(s), 1e-3))(pts)).reshape(GX.shape)
Zpsi = np.array(jax.vmap(psi)(pts)).reshape(GX.shape)
ZS = np.array(jax.vmap(S_of)(pts)).reshape(GX.shape)
Zpsi[~inside] = np.nan
ZS[~inside] = np.nan

dev = np.abs(ZS[inside] / 12.0 - 1.0)
_k = np.nanargmin(Zpsi)
_ext = np.unravel_index(_k, Zpsi.shape)
_at = (GX[_ext], GY[_ext])
_d = np.hypot(verts[:, 0] - _at[0], verts[:, 1] - _at[1])
print(f"psi: sup |psi| = {np.nanmax(np.abs(Zpsi)):.4e}, "
      f"min {np.nanmin(Zpsi):.4e}, max {np.nanmax(Zpsi):.4e}")
print(f"     extremum at s = ({_at[0]:.4f}, {_at[1]:.4f}); nearest vertex is "
      f"({verts[_d.argmin(), 0]:.4f}, {verts[_d.argmin(), 1]:.4f}) at distance "
      f"{_d.min():.4f}, polygon size {np.ptp(verts):.3f}")
print(f"Abreu on the plotted region: max |S/12-1| = {dev.max()*100:.4f}%, "
      f"mean {dev.mean()*100:.5f}%  ({inside.sum()} grid points)")

w = 1.15 * dev.max() * 12.0
print(f"Abreu colour window: 12 +- {w:.2e}  (caption must quote "
      f"[{12-w:.5f}, {12+w:.5f}])")


def pentagon(ax):
    c = verts.mean(0)
    order = np.argsort(np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0]))
    ax.add_patch(Polygon(verts[order], fill=False, ec="k", lw=1.4, zorder=5))
    ax.scatter(verts[:, 0], verts[:, 1], c="k", s=22, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
c1 = a1.contourf(GX, GY, Zpsi, levels=18, cmap="RdBu_r")
a1.contour(GX, GY, Zpsi, levels=18, colors="k", linewidths=0.3, alpha=0.4)
pentagon(a1)
fig.colorbar(c1, ax=a1, fraction=0.046)
a1.set_title(r"$\psi$", fontsize=11)

c2 = a2.contourf(GX, GY, ZS, levels=np.linspace(12 - w, 12 + w, 25),
                 cmap="RdBu_r", extend="both")
pentagon(a2)
fig.colorbar(c2, ax=a2, fraction=0.046, ticks=[12 - w, 12.0, 12 + w],
             format="%.5f")
a2.set_title(r"$S$", fontsize=11)
fig.tight_layout()
for p in _targets("fig_dp2_learned.png"):
    fig.savefig(p, dpi=160)
    print("wrote", p)
