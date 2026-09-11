"""Draw the DHHKW D-profile figure from the data dhhkw_D_compare.py saved.

The measurement is expensive (curvature of two stored metrics along a ray), so
the drawing lives here and reads dhhkw_D_profile.npz.  dhhkw_D_compare.py calls
draw() at the end of its run; rerun this file alone to restyle the figure
without recomputing anything.

The parameter counts in the legend are the invariant dimensions of the two
degrees, fixed by Molien's series (appendix B), not run-dependent.

Usage: python experiments/dp3/fig_D_profile.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "writing" / "paper1-new"
BLUE, RED, GREY = "#4c72b0", "#c44e52", "#8c8c8c"


def draw(npz=None):
    d = np.load(npz or ROOT / "experiments" / "dp3" / "dhhkw_D_profile.npz")
    x1 = d["x1"]
    m = x1 > 0
    # x1 -> 1 is the vertex, so ln(1-x1) runs vertex (left) to centre (right).
    t = np.log(1 - x1[m])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.semilogy(t, d["D_deg14"][m], "o-", ms=3, color=BLUE,
                label="ours, deg-14 (14 params)")
    ax.semilogy(t, d["D_deg18"][m], "s-", ms=3, color=RED,
                label="ours, deg-18 (21 params)")
    ax.semilogy(t, d["noise"][m], ":", color=GREY,
                label="numerical floor (conifold, $D=0$ exactly)")
    ax.axhline(1e-3, color="k", ls="--", lw=1)
    ax.text(0.02, 0.955, "DHHKW 18th order, at the vertex", fontsize=8,
            transform=ax.transAxes, va="top")
    ax.set_ylim(1e-16, 2e-2)
    ax.set_xlabel(r"$\ln(1-x_1)$   (hexagon vertex $\to$ centre)")
    ax.set_ylabel(r"$D=\sqrt{\frac{1}{4}(R_{\mu\nu}-g_{\mu\nu})^2}$")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for out in ([ROOT / "experiments" / "dp3" / "fig_dp3_D_profile.png"] +
                ([DRAFT / "fig_dp3_D_profile.png"] if DRAFT.is_dir() else [])):
        fig.savefig(out, dpi=160)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    draw()
