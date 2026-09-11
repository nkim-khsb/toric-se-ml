"""Figure 7's width ladder, with the optimizer budget released.

negative_control_nn.py runs the ladder at LBFGS_IT = 5000 with maxfun left at
scipy's default, so every one of its four points was stopped early rather than
at its capacity.  Measured afterwards: at fixed width 16, going from 5,000 to
90,919 iterations improves the held-out residual at b* by 85x, while the whole
width ladder at the old budget improved it by 2x for 11x the parameters.  The
ladder was therefore measuring the budget, not the capacity.  This reruns it
with maxfun = 100,000 so that what varies along it is the width.

The floor claim of section 5.5 does not depend on the outcome -- it was checked
separately (nn_floor_budget_probe.log: 3.70e-2 at width 32, inside the
polynomial's own seed spread of 3.45-3.76e-2).  What this decides is whether
the ladder is flat because the geometry stops it or because the budget did.

Writes the log, the data, and the figure the paper embeds.

The first version of this script printed the in-sample residual and the exit
status but saved neither, so the paper's in-sample claims -- agreement to five
percent at b*, factors 2.6-25 at the regular Reeb vector -- had no artifact
behind them.  The npz now carries params, held, capped, train and nit for both
Reeb vectors.  Every seed here is fixed (samples 1 and 7, init 0), so a rerun
reproduces the held-out column and is a check on it.

Usage: PYTHONPATH=. python experiments/dp2/width_ladder_released.py [maxfun]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.optimize import minimize

from sugrasol.cone import B_DP2_REG, dp2, minimize_reeb
from sugrasol.nets import init_mlp, mlp
from sugrasol.ypq import (gauge_fixed, loss_fn, residual_on_slice,
                          sample_slice, slice_chart)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "writing" / "paper1-new"
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
MAXFUN = int(_args[0]) if _args else 100_000
WIDTHS = (8, 16, 24, 32)
ADAM_IT = 800                      # as in negative_control_nn.py
POLY_FLOOR, POLY_BSTAR = 4.2e-2, 6.04e-14
BLUE, RED = "#4c72b0", "#c44e52"

cone = dp2()
b_se = minimize_reeb(cone, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)


def one(b, h):
    t0 = time.time()
    ch = slice_chart(cone, b)
    ss = sample_slice(jax.random.PRNGKey(1), ch, 3000, eps=2e-3)
    ss_te = sample_slice(jax.random.PRNGKey(7), ch, 3000, eps=2e-3)
    r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)

    def total(params, batch):
        mp, c = params
        return loss_fn(ch, gauge_fixed(lambda s: mlp(mp, s), ch.anchors), c, batch)

    params = (init_mlp(jax.random.PRNGKey(0), sizes=(2, h, h, 1), scale=1e-2),
              -float(jnp.mean(r0)))
    npar = int(sum(np.size(x) for x in jax.tree.leaves(params)))
    opt = optax.adam(5e-3)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        l, g = jax.value_and_grad(total)(params, batch)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l

    for _ in range(ADAM_IT):
        params, state, _ = step(params, state, ss)
    flat0, unravel = jax.flatten_util.ravel_pytree(params)
    vg = jax.jit(jax.value_and_grad(lambda fv: total(unravel(fv), ss)))
    res = minimize(lambda fv: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(fv))),
                   np.asarray(flat0), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=10 * MAXFUN, maxfun=MAXFUN,
                                ftol=1e-18, gtol=1e-16))
    params = unravel(jnp.asarray(res.x))
    held = float(total(params, ss_te))
    capped = res.nfev >= MAXFUN or res.status == 1
    train = float(res.fun)
    print(f"    width {h:2d} ({npar:5d} params): train {train:.3e}  "
          f"held-out {held:.3e}   ratio {held/train:.2f}   {res.nit:,} it, "
          f"{'CAPPED' if capped else 'converged'}, {time.time()-t0:.0f}s", flush=True)
    return npar, held, capped, train, int(res.nit)


def draw(out):
    """Markers, not a line, on the regular column.

    The four regular-b points are 5.0e-2, 3.0e-3, 3.4e-2, 2.7e-1: non-monotone,
    a factor 90 apart, and worst at the largest width.  Connecting them draws a
    trend through what is scatter, so they are plotted as markers inside a band
    spanning their range.  The b* column does saturate, so it keeps its line.
    The two reference lines carry the polynomial's exit status, since that is
    the difference between them: at b* it stops on its own convergence test, at
    the regular Reeb vector it does not stop at all.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pr = [r[0] for r in out["regular"]]
    hr = [r[1] for r in out["regular"]]
    ax.axhspan(min(hr), max(hr), color=RED, alpha=0.10, zorder=1)
    ax.semilogy(pr, hr, "s", ms=8, color=RED, zorder=4,
                label=r"KE attempt at $b=(3,0,0)$ (no line: see caption)")
    pb = [r[0] for r in out["b*"]]
    hb = [r[1] for r in out["b*"]]
    ax.semilogy(pb, hb, "o-", ms=7, lw=2, color=BLUE, zorder=4,
                label=r"SE at $b^{\ast}$")
    ax.axhline(POLY_FLOOR, color=RED, ls="--", lw=1.2, alpha=0.85,
               label=r"polynomial at $b=(3,0,0)$, never converges "
                     r"($4.2\times10^{-2}$)")
    ax.axhline(2.467e-15, color=BLUE, ls="--", lw=1.2, alpha=0.85,
               label=r"polynomial at $b^{\ast}$, converged "
                     r"($2.5\times10^{-15}$)")
    ax.set_xscale("log")
    ax.set_xticks(pr)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("network parameters")
    ax.set_ylabel("held-out residual")
    ax.set_ylim(3e-16, 3e0)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    for o in ([ROOT / "experiments" / "dp2" / "fig_dp2_width.png"] +
              ([DRAFT / "fig_dp2_width.png"] if DRAFT.is_dir() else [])):
        fig.savefig(o, dpi=160)
        print("wrote", o)
    plt.close(fig)


def replot():
    """Redraw from the saved run, so the figure can be restyled for free."""
    d = np.load(ROOT / "experiments" / "dp2" / "width_ladder_released.npz")
    out = {k: list(zip(d[f"{k}_params"], d[f"{k}_held"])) for k in ("regular", "b*")}
    draw(out)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
        raise SystemExit
    print(f"== dP2 width ladder, maxfun = {MAXFUN:,} "
          f"(negative_control_nn.py: maxiter 5000, maxfun at scipy's 15,000) ==\n",
          flush=True)
    out = {}
    for tag, b in (("regular", B_DP2_REG), ("b*", b_se)):
        print(f"  {tag}:", flush=True)
        out[tag] = [one(b, h) for h in WIDTHS]
        print(flush=True)

    np.savez(ROOT / "experiments" / "dp2" / "width_ladder_released.npz",
             widths=np.array(WIDTHS), maxfun=MAXFUN,
             **{f"{k}_{f}": np.array([r[i] for r in v])
                for k, v in out.items()
                for i, f in enumerate(("params", "held", "capped", "train", "nit"))})

    draw(out)
