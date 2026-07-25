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

BLUE, RED = "#4c72b0", "#c44e52"
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.semilogy(se_deg, se_held, "o-", color=BLUE, lw=2,
            label=r"SE, irregular Reeb (held-out) — solution exists")
ax.semilogy(deg, ke_held, "s-", color=RED, lw=2,
            label=r"KE-attempt, $b=(3,0,0)$ (held-out) — no solution")
ax.semilogy(deg, ke_train, "s--", color=RED, lw=1.2, alpha=0.6,
            label=r"KE-attempt (train) — overfits, does not generalize")
ax.axhspan(3e-2, 6e-2, color=RED, alpha=0.06)
ax.text(6.2, 4.6e-2, "held-out floor $\\approx 4\\times10^{-2}$\n(no KE on dP2)",
        fontsize=8.5, color=RED, va="center")
ax.text(11.5, 3e-13, "machine floor\n(genuine solution)", fontsize=8.5,
        color=BLUE, ha="center")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("mean-squared residual")
ax.set_title("dP2 negative control: same pentagon, two Reeb vectors")
ax.legend(fontsize=8.2, loc="center left")
ax.set_ylim(1e-15, 3e-1)
fig.tight_layout()
fig.savefig("experiments/dp2/fig_dp2_negative.png", dpi=160)
print("wrote experiments/dp2/fig_dp2_negative.png")
