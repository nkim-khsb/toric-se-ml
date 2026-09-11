"""Round-2 checks on the admissible space (review 2026-09-11, items 5, 6, 7).

(5) Structure of the spectrum.  With M = <theta_i,theta_j>, T = int theta_i ^ theta_j
    (= <theta_i, * theta_j>) and S = <theta_i+, theta_j+>, the Hodge decomposition
    gives S = (M + T)/2 identically.  Stokes kills T on the zero-period (bubble)
    directions, so rank T = d-3 =: k when the vertex directions span the primitive
    classes, and the generalized eigenproblem has N-k eigenvalues at EXACTLY 1/2 and
    k below.  The count is therefore structural; what is measured is how small the k
    are.  Here: numerical rank of T in the trial basis (gap in its singular values),
    sign of its nonzero eigenvalues (negative definite), and S = (M+T)/2 to roundoff.
(6) Harmonic distance.  For smooth closed primitive trial theta on a surface with
    b2+ = 1, theta = h + d alpha with h the harmonic representative of [theta] for
    the metric used, and  ||theta - h|| / ||theta|| = sqrt(2E).  Printed for the
    kernel vectors.
(7) Quadrature table for the FIXED kernel vectors: absolute numerator N = ||theta+||^2,
    denominator ||theta||^2 (=1 at the reference), and E, at ngl 24/48/96/128, for
    dP3 D = 16 and D = 20 and dP2 D = 16; the Y^{3,2} and conifold kernels are at
    roundoff and are reported as floors.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_stream_wachspress_checks.py
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

from sugrasol.forms import hodge                                     # noqa: E402
from r2b_ypq_calibration import load_case, base_metric               # noqa: E402
from r2b_holomorphic import load_dp3                                 # noqa: E402
from r2b_stream_potential import poly_quad                           # noqa: E402
from r2b_stream_admissible import forms_from_hess, flat_weighted, solve  # noqa: E402
from r2b_stream_wachspress import wachspress_space, hess_basis       # noqa: E402


def designs(ch, u, B, ngl, coef=None):
    smp, wt = poly_quad(ch, ngl)
    wt = wt / wt.sum()
    gs = jax.vmap(lambda s: base_metric(u, s))(smp)
    H = hess_basis(B, smp)
    if coef is not None:
        H = np.einsum("nkij,km->nmij", H, coef)
    _, W = forms_from_hess(H)
    st = jax.vmap(lambda O, g: jax.vmap(lambda o: hodge(o, g))(O))(jnp.asarray(W), gs)
    X = flat_weighted(W, gs, wt)
    Xst = flat_weighted(np.asarray(st), gs, wt)
    Xs = flat_weighted(0.5 * (jnp.asarray(W) + st), gs, wt)
    return X, Xs, Xst


def structure(tag, ch, u, d, D):
    k = d - 3
    B, K, rho = wachspress_space(ch.verts_s, D)
    X, Xs, Xst = designs(ch, u, B, 48)
    lam, coef, r = solve(X, Xs)
    # orthonormalize the trial basis in L2 (M = I) and express T, S there
    Q, R = np.linalg.qr(X)
    Rinv = np.linalg.inv(R)
    Tm = Rinv.T @ (X.T @ Xst) @ Rinv
    Sm = Rinv.T @ (Xs.T @ Xs) @ Rinv
    Tm = 0.5 * (Tm + Tm.T)
    ident = np.max(np.abs(Sm - 0.5 * (np.eye(r) + Tm)))
    ev = np.linalg.eigvalsh(Tm)
    sv = np.sort(np.abs(ev))[::-1]
    gap = sv[k - 1] / sv[k] if sv[k] > 0 else np.inf
    print(f"\n=== {tag}, D = {D}, trial dim {r}, k = d-3 = {k} ===")
    print(f"  S = (M+T)/2 identity: max deviation {ident:.1e}")
    print(f"  |eig T| sorted: " + "  ".join(f"{v:.2e}" for v in sv[:k + 3]) + f"   gap at k: {gap:.1e}")
    print(f"  the k largest eigenvalues of T: " + "  ".join(f"{v:+.4f}" for v in np.sort(ev)[:k])
          + "   (all negative = primitive classes negative definite)")
    print(f"  kernel E: " + "  ".join(f"{v:.3e}" for v in lam[:k]) + "   harmonic distance sqrt(2E): "
          + "  ".join(f"{np.sqrt(2*v):.2e}" for v in lam[:k]))
    return B, coef, lam


def scan(tag, ch, u, d, D, B, coef, ngls=(24, 48, 96, 128)):
    k = d - 3
    print(f"  quadrature table, fixed kernel vectors ({tag}, D = {D}):")
    print(f"  {'ngl':>4} | " + " | ".join(f"{'N_'+str(i):>12} {'den_'+str(i):>10} {'E_'+str(i):>10}" for i in range(k)))
    for ngl in ngls:
        X, Xs, _ = designs(ch, u, B, ngl, coef=coef[:, :k])
        N = np.sum(Xs ** 2, 0); Dn = np.sum(X ** 2, 0)
        print(f"  {ngl:>4} | " + " | ".join(f"{N[i]:12.6e} {Dn[i]:10.7f} {N[i]/Dn[i]:10.4e}" for i in range(k)), flush=True)


if __name__ == "__main__":
    ch, u, _, _ = load_dp3()
    for D in (16, 20):
        B, coef, lam = structure("dP3 (hexagon)", ch, u, 6, D)
        scan("dP3", ch, u, 6, D, B, coef)
    from harmonic_forms_dp2 import se_potential
    ch2, u2, held = se_potential()
    B, coef, lam = structure("dP2 (pentagon) at b*", ch2, u2, 5, 16)
    scan("dP2", ch2, u2, 5, 16, B, coef)
    ch3, u3, y1, y2 = load_case((3, 2))
    B, coef, lam = structure("Y^{3,2}", ch3, u3, 4, 16)
    scan("Y32", ch3, u3, 4, 16, B, coef, ngls=(24, 48, 96))
    chc, uc, _, _ = load_case("conifold")
    B, coef, lam = structure("conifold", chc, uc, 4, 8)
