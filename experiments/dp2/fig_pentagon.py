"""How the dP2 pentagon is drawn: toric diagram (fixed lattice rays) + the
Reeb-slice working polygon, with its genuine Z2 lattice symmetry.

Usage: PYTHONPATH=. python experiments/dp2/fig_pentagon.py
"""
import itertools

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from sugrasol.cone import MomentCone, minimize_reeb
from sugrasol.ypq import slice_chart

jax.config.update("jax_enable_x64", True)
BLUE, RED, GREY = "#4c72b0", "#c44e52", "#999999"

w = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1)]  # dP2 fan rays
removed = (0, -1)                                # the dP3 vertex we blew down
W = np.array(w)
dp2 = MomentCone(jnp.array([[1.0, a, b] for a, b in w]))
b = minimize_reeb(dp2, jnp.array([0.0, 0.33]), steps=8000, lr=2e-3)
ch = slice_chart(dp2, b)
vs = np.array(ch.verts_s)

# genuine lattice Z2 (brute force GL(2,Z) preserving the ray set)
G = [np.array([[a, bb], [c, d]]) for a, bb, c, d in itertools.product(range(-2, 3), repeat=4)
     if abs(round(np.linalg.det(np.array([[a, bb], [c, d]])))) == 1
     and set(map(tuple, (np.array([[a, bb], [c, d]]) @ W.T).T.tolist())) == set(map(tuple, w))]
R_w = [A for A in G if not np.array_equal(A, np.eye(2))][0]
# Z2 action on slice vertices (permutation) for colouring the orbits
img = (R_w @ W.T).T
perm = [list(map(tuple, w)).index(tuple(p)) for p in img.tolist()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# ---- panel 1: toric diagram
hexrays = w + [removed]
for p in hexrays:
    ax1.plot([0, p[0]], [0, p[1]], color=GREY, lw=0.8, zorder=1)
ax1.add_patch(Polygon(W[np.argsort(np.arctan2(W[:, 1], W[:, 0]))], fill=False,
                      ec=BLUE, lw=2, zorder=3))
ax1.scatter(W[:, 0], W[:, 1], c=BLUE, s=70, zorder=4, label="dP2 rays (kept)")
ax1.scatter(*removed, facecolors="none", edgecolors=RED, s=110, lw=2,
            zorder=4, label="removed (blow-down)")
for p in w:
    ax1.annotate(f"{p}", p, textcoords="offset points", xytext=(6, 5), fontsize=9)
ax1.annotate("(0,-1)", removed, textcoords="offset points", xytext=(6, -12),
             fontsize=9, color=RED)
for gx in range(-2, 3):
    ax1.axhline(gx, color="#eee", lw=0.5, zorder=0)
    ax1.axvline(gx, color="#eee", lw=0.5, zorder=0)
ax1.set_aspect("equal")
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_title("toric diagram: dP2 = dP3 hexagon minus one ray")
ax1.legend(fontsize=8, loc="lower left")

# ---- panel 2: Reeb-slice working polygon, coloured by Z2 orbit
order = np.argsort(np.arctan2(vs[:, 1] - vs[:, 1].mean(), vs[:, 0] - vs[:, 0].mean()))
ax2.add_patch(Polygon(vs[order], fill=True, fc="#eaf0f7", ec=BLUE, lw=2, zorder=1))
colors = {}
palette = ["#c44e52", "#4c72b0", "#55a868"]
ci = 0
for i in range(len(vs)):
    orb = frozenset({i, perm[i]})
    if orb not in colors:
        colors[orb] = ("k" if perm[i] == i else palette[ci]); ci += perm[i] != i
for i, p in enumerate(vs):
    ax2.scatter(*p, c=colors[frozenset({i, perm[i]})], s=80, zorder=4)
ax2.scatter(*vs.mean(0), marker="x", c="k", s=90, zorder=5, label="centroid (off-centre)")
ax2.set_aspect("equal")
ax2.set_title("working polygon (Reeb slice): only $Z_2$\n(same colour = $Z_2$ pair; black = fixed)")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("experiments/dp2/fig_dp2_pentagon.png", dpi=160)
print("wrote experiments/dp2/fig_dp2_pentagon.png")
print("slice vertices:", [tuple(np.round(p, 3)) for p in vs])
print("Z2 vertex permutation:", perm, " reflection R_w=", R_w.tolist())
