"""The symmetry-free dP3 network in the (held-out residual, D6 defect) plane.

Data: nn_seeds_and_kick_sweep.log beside this file, the 769 s run
that trained eight seeds directly and then perturbed and relaxed four seeds at
each of five perturbation sizes.  Nothing is retrained here; the log carries
every (residual, defect) pair the figure needs.

What the figure is for is one fact: perturbed states sit at defect of order one
and come back to the converged cluster at ~1e-2 when the Monge-Ampere residual
alone is minimised.  The earlier version also drew a fitted slope and the
polynomial's 2.8e-16 floor; the slope is quoted in the text and spans little
more than a decade, and the floor line stretched the axis over thirteen decades
and squashed the data into the top three.  Both are gone.

Usage: python experiments/dp3/fig_defect_residual.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG = Path(__file__).resolve().parent / "nn_seeds_and_kick_sweep.log"
DRAFT = ROOT / "writing" / "paper1-new"
BLUE, RED, GREY = "#4c72b0", "#c44e52", "#8c8c8c"


def read(log=LOG):
    txt = log.read_text()
    direct = [(float(a), float(b)) for a, b in re.findall(
        r"seed \d+: init defect [\d.]+\s+->\s+held-out ([\d.eE+-]+)\s+defect ([\d.eE+-]+)", txt)]
    perturbed, recovered, failed = [], [], []
    for m in re.finditer(r"rel ([\d.]+) seed (\d+): kicked held ([\d.eE+-]+) defect "
                         r"([\d.eE+-]+)\s+->\s+restored held ([\d.eE+-]+) defect "
                         r"([\d.eE+-]+)\s+(RECOVERED|failed)", txt):
        eps, _, hk, dk, hr, dr, verdict = m.groups()
        perturbed.append((float(hk), float(dk)))
        (recovered if verdict == "RECOVERED" else failed).append((float(hr), float(dr)))
    return direct, perturbed, recovered, failed


def draw():
    direct, perturbed, recovered, failed = read()
    assert (len(direct), len(perturbed), len(recovered), len(failed)) == (8, 20, 16, 4), \
        "the log no longer holds 8 direct runs and 20 perturbations"

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    xy = lambda p: ([a for a, _ in p], [b for _, b in p])
    ax.scatter(*xy(perturbed), marker="x", s=34, color=GREY, lw=1.2,
               label=f"perturbed ({len(perturbed)})", zorder=2)
    ax.scatter(*xy(direct), marker="o", s=34, color=BLUE,
               label=f"directly trained ({len(direct)})", zorder=4)
    ax.scatter(*xy(recovered), marker="o", s=38, facecolors="none",
               edgecolors=BLUE, lw=1.3,
               label=f"relaxed, symmetry recovered ({len(recovered)})", zorder=4)
    ax.scatter(*xy(failed), marker="s", s=34, facecolors="none",
               edgecolors=RED, lw=1.3,
               label=f"relaxed, did not recover ({len(failed)})", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("held-out residual")
    ax.set_ylabel(r"$D_6$ equivariance defect")
    ax.set_xlim(5e-8, 2e1)
    ax.set_ylim(1e-3, 6e0)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for out in ([ROOT / "experiments" / "dp3" / "fig_nn_defect_residual.png"] +
                ([DRAFT / "fig_nn_defect_residual.png"] if DRAFT.is_dir() else [])):
        fig.savefig(out, dpi=160)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    draw()
