"""How close to exactly 1/2 are the non-kernel self-dual energies?

The draft states, of the zero-period (bubble) directions, that "we see that value
to 1e-10 on every polygon".  That number was asserted, not measured: the tables
print three digits.  This measures it, in two spaces:

  (a) the exactly-admissible POLYNOMIAL subspace, affine + prod_a ell_a * q, which
      is what the sentence is about;
  (b) the Wachspress space eq:admspace actually used, where the same statement
      applies to every direction above the d-3 kernel.

For each we print max |E - 1/2| over the non-kernel directions, at two quadrature
orders, so that integration error can be told from function-space error.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_half_check.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments" / "dp3kt", ROOT / "experiments" / "dp2"):
    sys.path.insert(0, str(p))

from r2b_ypq_calibration import load_case                            # noqa: E402
from r2b_holomorphic import load_dp3                                 # noqa: E402
from r2b_stream_admissible import (admissible_basis, sd_design, solve, RHO,  # noqa: E402
                                   powers)
from r2b_stream_wachspress import wachspress_space, design           # noqa: E402


def bubble_only(verts, D, d):
    """Coefficients of prod_a ell_a * (monomials of degree <= D-d) in the monomial
    basis of the scaled chart -- the exactly admissible polynomials, modulo affine."""
    V = np.asarray(verts) / RHO[0]
    # ell_a(s) = c_a + n_a . s, from consecutive vertices
    lines = []
    for a in range(d):
        P0, P1 = V[a - 1], V[a]
        t = P1 - P0
        n = np.array([-t[1], t[0]])
        lines.append((-n @ P0, n))
    pw = powers(D)
    idx = {pq: k for k, pq in enumerate(pw)}
    # multiply out prod ell_a as a polynomial dict {(i,j): coeff}
    prod = {(0, 0): 1.0}
    for c, n in lines:
        new = {}
        for (i, j), v in prod.items():
            for dk, cc in (((0, 0), c), ((1, 0), n[0]), ((0, 1), n[1])):
                key = (i + dk[0], j + dk[1])
                new[key] = new.get(key, 0.0) + v * cc
        prod = new
    cols = []
    for tot in range(0, D - d + 1):
        for i in range(tot + 1):
            j = tot - i
            col = np.zeros(len(pw))
            ok = True
            for (a, b), v in prod.items():
                key = (a + i, b + j)
                if key in idx:
                    col[idx[key]] += v
                elif abs(v) > 0:
                    ok = False
            if ok:
                cols.append(col)
    return np.array(cols).T if cols else np.zeros((len(pw), 0))


def report(tag, ch, u, d, Ds=(8, 12, 16)):
    k = d - 3
    verts = np.asarray(ch.verts_s)
    print(f"\n=== {tag}  (d = {d}, kernel k = {k}) ===")
    for D in Ds:
        RHO[0] = float(np.max(np.linalg.norm(verts, axis=1)))
        # (a) bubble-only polynomial subspace
        N = bubble_only(verts, D, d)
        line = f"  D={D:>3}  bubbles={N.shape[1]:>4}"
        for ngl in (24, 96):
            X, Xs, _ = sd_design(ch, u, D, ngl, N=N)
            lam, _, r = solve(X, Xs)
            line += f" | poly bubble max|E-1/2| @ngl{ngl}: {np.max(np.abs(lam - 0.5)):.2e}"
        print(line, flush=True)
        # (b) Wachspress space, directions above the kernel
        B, K, rho = wachspress_space(verts, D)
        line = f"  D={D:>3}  trial={K:>4}"
        for ngl in (24, 96):
            X, Xs, _, _ = design(ch, u, B, ngl)
            lam, _, r = solve(X, Xs)
            line += f" | Wachspress max|E-1/2| above kernel @ngl{ngl}: {np.max(np.abs(lam[k:] - 0.5)):.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    ch, u, _, _ = load_case("conifold")
    report("conifold (square)", ch, u, 4)
    ch, u, y1, y2 = load_case((3, 2))
    report("Y^{3,2}", ch, u, 4)
    ch, u, _, _ = load_dp3()
    report("dP3 (hexagon)", ch, u, 6, Ds=(8, 12, 16))
    from harmonic_forms_dp2 import se_potential
    ch, u, held = se_potential()
    report("dP2 (pentagon) at b*", ch, u, 5)
