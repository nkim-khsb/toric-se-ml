"""dP2 negative control figure: KE-attempt (no solution -> held-out floor) vs
SE (solution exists -> machine floor). Data from negative_control.py (2026-07-14).

Usage: PYTHONPATH=. python experiments/dp2/fig_negative.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

deg = [6, 8, 10, 12, 14, 16]
ke_train = [7.86e-2, 6.45e-2, 5.10e-2, 3.13e-2, 1.99e-2, 1.03e-2]
ke_held = [8.00e-2, 7.07e-2, 5.87e-2, 4.43e-2, 4.36e-2, 4.23e-2]
se_deg = [6, 8, 10, 12, 14]
se_held = [1.20e-7, 2.58e-9, 6.23e-11, 1.75e-12, 6.04e-14]
# degree 16 at b* is 1.5-3.4e-15 over three independent sampling draws
# (tab:dp2ladder).  Drawn as the range, with the line carried to its geometric
# centre, so the figure does not invent a single value the table does not give.
SE16_LO, SE16_HI = 1.5e-15, 3.4e-15
SE16_MID = (SE16_LO * SE16_HI) ** 0.5

BLUE, RED = "#4c72b0", "#c44e52"
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.semilogy(se_deg + [16], se_held + [SE16_MID], "o-", color=BLUE, lw=2,
            markevery=list(range(len(se_deg))),
            label=r"SE, irregular Reeb (held-out) — solution exists")
ax.errorbar([16], [SE16_MID], yerr=[[SE16_MID - SE16_LO], [SE16_HI - SE16_MID]],
            fmt="none", ecolor=BLUE, elinewidth=1.6, capsize=4, capthick=1.6,
            label=r"degree 16: range over three sampling draws")
ax.semilogy(deg, ke_held, "s-", color=RED, lw=2,
            label=r"KE-attempt, $b=(3,0,0)$ (held-out) — no solution")
ax.semilogy(deg, ke_train, "s--", color=RED, lw=1.2, alpha=0.6,
            label=r"KE-attempt (in-sample) — overfits")
ax.axhspan(3e-2, 6e-2, color=RED, alpha=0.06)
ax.text(6.2, 4.0e-3, "held-out floor $\\approx 4\\times10^{-2}$\n(no KE on dP2)",
        fontsize=8.5, color=RED, va="center")
ax.text(10.4, 2.2e-15, "converged\n(solution exists)", fontsize=8.5,
        color=BLUE, ha="center", va="center")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("mean-squared residual")
ax.legend(fontsize=8.2, loc="center right")
ax.set_ylim(3e-16, 3e-1)
fig.tight_layout()
from pathlib import Path

for _out in (["experiments/dp2/fig_dp2_negative.png"] +
             (["writing/paper1-new/fig_dp2_negative.png"]
              if Path("writing/paper1-new").is_dir() else [])):
    fig.savefig(_out, dpi=160)
    print("wrote", _out)
