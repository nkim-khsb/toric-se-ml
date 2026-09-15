"""Review 2026-09-13, item 8: arithmetic accuracy of the quoted energies.

The eigensolver path builds the trial forms, applies the Hodge star, L^2-
orthonormalizes by pivoted QR and diagonalizes S = Xs^T Xs.  Quadrature
stability (tab:quadscan) does not by itself bound the arithmetic error of
that basis change.  This script re-evaluates the SAME low-energy vectors,
fixed at ngl = 48, by an independent path that never forms a 4x4 form, a
Hodge star or a QR factor: the pointwise identity (eq:sdensity)

    ||theta_+||^2 = (det H / 2) (u^{ij} chi_ij)^2 ,   ||theta||^2 = tr(u A H A^T),

integrated directly from the stream-function coefficients.  The type residual
r = u^{ij} chi_ij is evaluated by two roundoff paths (inv(H).C and adj(H).C/det H),
and the sums are repeated with float64 node values accumulated in longdouble.
What is shared with the eigensolver path is only the evaluation of the basis
Hessians and of Hess G_P at the nodes.

PRE-REGISTERED: the density-path energy agrees with the eigensolver energy to
at least 6 significant digits for hexagon D = 20 and pentagon D = 16, at ngl
48 and 96; the two residual paths agree to 1e-12 relative; longdouble
accumulation changes E by less than 1e-12 relative.  Failures are reported.

Usage: PYTHONPATH=. python experiments/dp3kt/r2b_arith_check.py
"""
import sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/"experiments"/"dp3kt", ROOT/"experiments"/"dp2", ROOT/"experiments"/"dp3"):
    sys.path.insert(0, str(p))
from r2b_stream_wachspress import wachspress_space, design, hess_basis   # noqa: E402
from r2b_stream_admissible import solve                                  # noqa: E402
from r2b_stream_potential import poly_quad                               # noqa: E402
from r2b_holomorphic import load_dp3                                     # noqa: E402
from harmonic_forms_dp2 import se_potential                              # noqa: E402

J = np.array([[0., -1.], [1., 0.]])
FAILS = []

def density_energy(ch, u, B, coef, ngl, ld=False):
    smp, wt = poly_quad(ch, ngl); wt = np.asarray(wt); wt = wt / wt.sum()
    Hb = hess_basis(B, smp)                                   # (n,K,2,2)
    C = np.einsum("nkij,km->nmij", Hb, coef)                  # (n,m,2,2)
    Hm = np.asarray(jax.vmap(jax.hessian(u))(jnp.asarray(smp)))  # (n,2,2)
    f = np.longdouble if ld else np.float64
    C, Hm, wt = C.astype(f), Hm.astype(f), wt.astype(f)
    detH = Hm[:, 0, 0]*Hm[:, 1, 1] - Hm[:, 0, 1]*Hm[:, 1, 0]
    adj = np.empty_like(Hm); adj[:, 0, 0] = Hm[:, 1, 1]; adj[:, 1, 1] = Hm[:, 0, 0]
    adj[:, 0, 1] = -Hm[:, 0, 1]; adj[:, 1, 0] = -Hm[:, 1, 0]
    uinv = adj / detH[:, None, None]
    r1 = np.einsum("nij,nmji->nm", uinv, C)                   # tr(u C)
    r2 = np.einsum("nij,nmji->nm", adj, C) / detH[:, None]     # tr(adj C)/det
    A = np.einsum("nmij,jk->nmik", C, J.astype(f))
    den = np.einsum("n,nij,nmjk,nkl,nmil->m", wt, uinv, A, Hm, A)   # tr(u A H A^T)
    Np1 = np.einsum("n,n,nm->m", wt, detH/2, r1**2)
    Np2 = np.einsum("n,n,nm->m", wt, detH/2, r2**2)
    return Np1/den, Np2/den, Np1, den

def run(name, loader, D, k):
    t0 = time.time()
    out = loader()
    ch, u = out[0], out[1]
    B, K, rho = wachspress_space(ch.verts_s, D)
    X, Xs, smp, _ = design(ch, u, B, 48)
    lam, coef, r = solve(X, Xs)
    c = coef[:, :k]
    print(f"\n=== {name}, D = {D}, trial dim {K} (rank {r}), vectors fixed at ngl = 48   [{time.time()-t0:.0f}s]")
    print(f"  eigensolver E (ngl 48):      " + "  ".join(f"{v:.9e}" for v in lam[:k]))
    X96, Xs96, _, _ = design(ch, u, B, 96); lam96, _, _ = solve(X96, Xs96)
    print(f"  eigensolver E (ngl 96):      " + "  ".join(f"{v:.9e}" for v in lam96[:k]))
    for ngl in (48, 96):
        E1, E2, Np, den = density_energy(ch, u, B, c, ngl)
        E1l, _, _, _ = density_energy(ch, u, B, c, ngl, ld=True)
        ref = lam[:k]
        rel = np.abs(E1 - ref) / ref
        digits = -np.log10(np.maximum(rel, 1e-300))
        print(f"  density path, ngl {ngl:3d}:       " + "  ".join(f"{v:.9e}" for v in E1))
        print(f"    agreement with eigensolver:  " + "  ".join(f"{d:.1f} digits" for d in digits))
        print(f"    residual paths inv vs adj:   " + "  ".join(f"{abs(a-b)/a:.1e}" for a, b in zip(E1, E2)))
        print(f"    longdouble accumulation:     " + "  ".join(f"{abs(float(a)-b)/b:.1e}" for a, b in zip(E1l, E1)))
        if np.any(digits < 6): FAILS.append(f"{name} ngl {ngl}: agreement {digits}")
        if np.any(np.abs(E1-E2)/E1 > 1e-12): FAILS.append(f"{name} ngl {ngl}: residual paths differ")
        if np.any(np.abs(E1l.astype(float)-E1)/E1 > 1e-12): FAILS.append(f"{name} ngl {ngl}: longdouble moves E")
    print(f"  [{time.time()-t0:.0f}s]")

run("hexagon dP3", load_dp3, 20, 3)
run("pentagon dP2 at b*", se_potential, 16, 2)
print("\nFAILURES:", FAILS if FAILS else "none")
