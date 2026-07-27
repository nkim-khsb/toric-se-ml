"""Does the D6-invariant basis we actually fit span the full invariant space?

The hexagon's D_6 acts on the slice coordinates as a rank-2 reflection group, so
its ring of invariants is free on two generators of degree 2 and 6 (DHHKW's
U = x1^2+x1x2+x2^2 and V = x1^2 x2^2 (x1+x2)^2, their eq. (5.1)).  The number of
invariants of degree <= d, constants excluded, is therefore the count of
U^i V^j with 2 <= 2i+6j <= d:  FOURTEEN at degree 14, TWENTY-ONE at degree 18.

Our basis is instead built by Reynolds-averaging all monomials and taking the
SVD column space with a RELATIVE tolerance (ypq.whiten_sym_poly).  The averaged
monomials are severely collinear -- the singular values span ten orders -- so
that one relative threshold decides how many genuine invariant directions
survive.  The persisted artifacts kept 11 of 14 (deg-14, tol 1e-9) and 17 of 21
(deg-18, tol 1e-12).  This is the third appearance of the conditioning trap
already recorded in log.md (2026-07-12 Y^{p,q} "degree-10 mirage", 2026-07-20
"tol=1e-9 truncates deg>14"); the 2026-07-20 fix moved the tolerance but did not
remove the truncation, and the "degree-22 saturation" noted there is what
re-truncation looks like.

The question here is NOT whether the count is wrong -- it is, and the singular
value spectrum shows a clean gap exactly at 14 and at 21.  It is whether the
missing directions CARRY ANYTHING.  They sit at relative singular values ~2e-10
(deg-14) and ~1.5e-13 (deg-18), i.e. three to six digits of float64 resolution,
so they may well be inert, or actively harmful to condition.

PRE-REGISTERED verdict (fixed before running).  Refit at each tolerance and
compare against the persisted artifact.  The extra directions are INERT -- and
the paper is then corrected by wording alone, artifacts and every downstream
number left untouched -- iff, at BOTH degrees:
  (i)   the held-out Monge-Ampere loss changes by less than a factor 2,
  (ii)  D at the hexagon vertex (DHHKW's error measure, the quantity the paper
        quotes) changes by less than 10% -- its numerical floor is ~1e-10, so
        10% is far above noise,
  (iii) lambda2 changes by less than 1e-4 relative (the tolerance of the
        regression test in tests/test_laplacian.py).
Any violation means the persisted artifacts must be replaced and every
downstream dP3 number recomputed.  If the enlarged basis makes these WORSE,
that is equally an answer: the truncation was protective, we keep 11/17, and we
say so in the text rather than pretending the count was intended.

Usage: PYTHONPATH=. python experiments/dp3/invariant_rank_check.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))

from sugrasol.cone import B_DP3, dp3                              # noqa: E402
from sugrasol.curvature import ricci, toric_metric                # noqa: E402
from sugrasol.laplacian import (laplace_spectrum, orthonormal_basis,  # noqa: E402
                                polygon_quadrature, slice_potential)
from sugrasol.ypq import (_monomials, _poly_powers,               # noqa: E402
                          dihedral_matrices, loss_fn, residual_on_slice,
                          sample_slice, slice_chart, sym_ortho_psi,
                          whiten_sym_poly)

# Setup of the persisted artifacts (log.md 2026-07-20): deg-14 from 2048
# samples at tol 1e-9, deg-18 from 8192 samples at tol 1e-12; held-out on key 7.
RUNS = [(14, 2048, (1e-9, 1e-12, 1e-14), 11, 2.585e-12),
        (18, 8192, (1e-12, 1e-14), 17, 2.545e-15)]

ch = slice_chart(dp3(), B_DP3)
group = dihedral_matrices(ch.verts_s)
verts = np.array(ch.verts_s)
nodes, wts = polygon_quadrature(ch.verts_s, ngl=24)
cloud = sample_slice(jax.random.PRNGKey(5), ch, 40000, eps=1e-4)
probe = sample_slice(jax.random.PRNGKey(9), ch, 200, eps=1e-2)


def theory_dim(deg):
    """# of U^i V^j with 2 <= 2i+6j <= deg  (deg U = 2, deg V = 6)."""
    return sum(1 for i in range(deg // 2 + 1) for j in range(deg // 6 + 1)
               if 2 <= 2 * i + 6 * j <= deg)


def fit(deg, nsamp, tol):
    ss = sample_slice(jax.random.PRNGKey(1), ch, nsamp, eps=2e-3)
    st = sample_slice(jax.random.PRNGKey(7), ch, nsamp, eps=2e-3)
    powers, W, grp = whiten_sym_poly(deg, ss, group, tol=tol)
    nc = W.shape[1]
    vg = jax.jit(jax.value_and_grad(
        lambda v: loss_fn(ch, sym_ortho_psi(v[:nc], powers, W, grp), v[nc], ss)))
    r0 = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
    v0 = np.zeros(nc + 1)
    v0[nc] = -float(jnp.mean(r0))
    res = minimize(lambda v: (lambda l, g: (float(l), np.asarray(g)))(*vg(jnp.asarray(v))),
                   v0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-18, gtol=1e-16))
    v = jnp.asarray(res.x)
    psi = sym_ortho_psi(v[:nc], powers, W, grp)
    held = float(loss_fn(ch, psi, v[nc], st))
    return psi, nc, float(res.fun), held


def diagnostics(psi):
    """(D at the vertex, Abreu S, lambda1, lambda2) -- as in the paper."""
    u = slice_potential(ch, psi)
    gp_fn = lambda x: 3.0 * toric_metric(u, 2)(x)        # noqa: E731  (DHHKW Ric=g)

    def D(s):
        x = jnp.concatenate([s, jnp.zeros(2)])
        gp = gp_fn(x)
        T = ricci(gp_fn, x) - gp
        gpi = jnp.linalg.inv(gp)
        return jnp.sqrt(0.25 * jnp.einsum("ab,ac,bd,cd->", T, gpi, gpi, T))

    Dv = float(jax.jit(D)(jnp.asarray(0.99999 * verts[3])))
    hinv = lambda s: jnp.linalg.inv(jax.hessian(u)(s))   # noqa: E731
    S = float(jnp.mean(jax.vmap(lambda s: -jnp.einsum(
        "jkjk->", jax.jacfwd(jax.jacfwd(hinv))(s)))(probe)))

    pw = _poly_powers(20)

    def raw(s):
        gs = jnp.einsum("gij,j->gi", group, s)
        sym = jnp.mean(jax.vmap(lambda g: _monomials(g, pw))(gs), axis=0)
        return jnp.concatenate([jnp.ones(1), sym])

    w = laplace_spectrum(u, orthonormal_basis(raw, cloud, tol=1e-12),
                         nodes, weights=wts)
    return Dv, S, float(w[1]) * 4 / S, float(w[2]) * 4 / S


out = {}
for deg, nsamp, tols, kept, held_ref in RUNS:
    print(f"\n=== degree {deg}: invariant space is {theory_dim(deg)}-dimensional; "
          f"the persisted artifact keeps {kept} ===")
    print(f"{'tol':>8} {'dim':>4} {'train':>11} {'held-out':>11} "
          f"{'D(vertex)':>11} {'S':>10} {'lambda1':>10} {'lambda2':>10}")
    for tol in tols:
        psi, nc, tr, held = fit(deg, nsamp, tol)
        Dv, S, l1, l2 = diagnostics(psi)
        out[(deg, tol)] = (nc, tr, held, Dv, l1, l2)
        print(f"{tol:8.0e} {nc:4d} {tr:11.3e} {held:11.3e} {Dv:11.3e} "
              f"{S:10.6f} {l1:10.6f} {l2:10.5f}")
    print(f"   [reproduction check] artifact held-out {held_ref:.3e}")

# ------------------------------------------------- pre-registered verdict
print("\n=== verdict (criteria fixed in the docstring before running) ===")
inert = True
for deg, _, tols, kept, _ in RUNS:
    base = out[(deg, tols[0])]
    full = out[(deg, tols[-1])]
    assert base[0] == kept, f"deg {deg}: baseline dim {base[0]} != artifact {kept}"
    r_held, r_D = full[2] / base[2], full[3] / base[3]
    r_l2 = abs(full[5] / base[5] - 1.0)
    ok = (0.5 < r_held < 2.0) and (0.9 < r_D < 1.1) and r_l2 < 1e-4
    inert &= ok
    print(f"  deg {deg}: {base[0]} -> {full[0]} directions | "
          f"held-out x{r_held:.3f} | D(vertex) x{r_D:.3f} | "
          f"lambda2 rel {r_l2:.2e}  -> {'inert' if ok else 'MOVES'}")
    if r_D > 1.1 or r_held > 2.0:
        print("        (the enlarged basis is WORSE here -- truncation was "
              "protective, which is itself the answer)")
print("  => " + ("INERT: correct the wording only, leave artifacts and every "
                 "downstream number untouched."
                 if inert else
                 "MOVES: the artifacts must be replaced and every downstream "
                 "dP3 number recomputed."))
