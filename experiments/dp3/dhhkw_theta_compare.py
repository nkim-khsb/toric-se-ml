"""Grade our dP3 metric against DHHKW's PUBLISHED harmonic (1,1)-form.

Why this is a ground truth (rule 2).  DHHKW hep-th/0703057 did not merely
attempt theta_a: their sec. 6.2 prints a least-squares fit of theta_2 to sixth
order -- fifteen coefficients, eq. (6.13).  Unlike lambda_2, which they left
unconverged, this is a published, checkable object, and it comes with an EXACT
protected anchor, their (6.15).

Their construction (sec. 3.7 and 6.2, transcribed):

    theta_a = i d dbar mu_a ,   mu_a = ln(1 + v_a . x) + sum c_nm x1^n x2^m
                                                                       (6.11)
    harmonic  <=>  d_i ( G^{ij} d_j mu ) = const                        (3.46)
    omega = (1/2) sum_a theta_a                                         (6.14)
    (theta_a)_{ij} F^{ij} = 2/3   exactly; their fit gave 0.6672         (6.15)
    M_theta = sum_p |Delta mu - const|^2 ~ 1e-4 over 100 points          (6.12)
              => rms |Delta mu - const| ~ 1e-3, and they warn the error
                 "gets much worse outside the fitting domain |x_i| > 0.9"

(3.46) is precisely the torus-invariant Laplacian laplacian.py already implements
and anchors on the conifold, so no new machinery is needed: their theta is
harmonic iff L[mu] is constant, and the constant is fixed exactly.

Two conventions to bridge, and they turn out to be the same one.

  Coordinates.  Their polytope is {x : l_a = 1 + v_a.x >= 0} -- the log in (6.11)
  fixes lambda_a = 1 -- and ours is l_a = 1/3 + v_a.s.  Hence  x = 3 s  exactly;
  their vertices then sit at |x_i| <= 1, matching their "boundary x1 = 1" and
  their fitting domain 0 < |x_i| < 0.9.  Independent confirmation of the bridge:
  their choice c_nm = c_mn is the x1 <-> x2 reflection, which is exactly the
  stabilizer of their facet a=2, whose normal is v_2 = (1,1).

  Normalization.  Under x = 3s the Guillemin potential scales as
  u^them(x) = 3 u^ours(s) + affine, so (Hess_x u^them)^{-1} = 3 (Hess_s u^ours)^{-1}
  while each d_x = (1/3) d_s: two factors 1/3 against one 3 give
  L^them = L^ours / 3.  That is the SAME factor as the paper's eigenvalue
  conversion lambda_DHHKW = lambda_ours * 4/S = lambda_ours / 3 (their Ric=g has
  S=4, ours S=12), so the two conventions check each other.

  Sign.  L = d_i(u^{ij} d_j .) is the analyst's Laplacian: the eigenproblem
  solved elsewhere is -L f = lambda f with lambda > 0 (conifold lambda_1 = +6),
  so L is negative-definite, whereas DHHKW's g^{i jbar} theta_{i jbar} is
  positive.  Hence  g^{i jbar} theta_{i jbar} = - L[mu] / 3.  The sign is global
  and identical in every block below, so it cannot absorb an error in one of them.

What is measured
  [A] their published theta_2, tested on OUR metric: is L[mu] constant, and is
      the constant 2/3?
  [anti] the same mu on the canonical Guillemin metric (psi = 0, not Einstein),
      which must fail -- otherwise the grade is not psi-sensitive.
  [B] the protected omega anchor (6.14): summing mu_2 over the D_6 orbit gives
      2 sum_a mu_a, so (1/4) sum_g mu_2 o g is the potential of omega and its
      constant is exactly 2.
  [C] our own fit of the identical ansatz on our metric: the constant, the
      harmonicity residual, and the coefficients against (6.13) -- with least-
      squares standard errors and the design-matrix condition number, which
      decide whether a coefficient-by-coefficient comparison means anything.
  [D] the gauge-free comparison: mu_theirs - mu_ours pointwise.  The log terms
      cancel exactly in the difference, so the difference is a pure polynomial
      and the honest denominator is the spread of the polynomial part, not of mu.
  [E] an order ladder (6, 8, 10) and the R2b+ facet weights ell_a^{5/2}, to see
      how far below their residual the same construction can be driven.

Usage: PYTHONPATH=. python experiments/dp3/dhhkw_theta_compare.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from sugrasol.artifacts import load_psi_dp3
from sugrasol.cone import B_DP3, dp3
from sugrasol.laplacian import slice_potential
from sugrasol.ypq import dihedral_matrices, sample_slice, slice_chart

ROOT = Path(__file__).resolve().parents[2]
NPZ = {"deg-14": ROOT / "experiments/dp3smooth/dp3_G_deg14.npz",
       "deg-18": ROOT / "experiments/dp3smooth/dp3_G_deg18.npz"}

V2 = np.asarray(dp3().normals)[:, 1:]        # inward normals, cyclic
A_FACET = 1                                  # their a=2 (1-indexed) -> v=(1,1)
SCALE = 3.0                                  # x = 3 s
C_ELL = 1.0 / 3.0                            # our l_a = 1/3 + v_a . s
TO_THEM = -1.0 / 3.0                         # L^ours  ->  g^{ijbar} theta_{ijbar}

# DHHKW eq. (6.13), c_nm = c_mn, sixth order.  Transcribed verbatim.
# c_00 is absent: an additive constant in mu is invisible to both theta and (3.46).
C613 = {(1, 0): -0.2250,
        (2, 0): 0.0638, (1, 1): 0.0311,
        (3, 0): -0.0301, (2, 1): 0.0059,
        (4, 0): 0.0126, (3, 1): 0.0005, (2, 2): -0.0073,
        (5, 0): -0.0150, (4, 1): -0.0245, (3, 2): -0.0240,
        (6, 0): 0.0088, (5, 1): 0.0223, (4, 2): 0.0281, (3, 3): 0.0196}
POWERS6 = sorted(C613)
CD = np.array([C613[p] for p in POWERS6])


def sym_powers(order):
    """(n,m) with n >= m, 1 <= n+m <= order: the symmetrized monomials of (6.11)."""
    return sorted((n, m) for n in range(order + 1) for m in range(n + 1)
                  if 1 <= n + m <= order)


def mu_powers(s, powers):
    """The monomial part of (6.11) in THEIR x = 3s, symmetrized c_nm = c_mn:
    each (n,m) with n > m contributes x1^n x2^m + x1^m x2^n."""
    x = SCALE * s
    out = []
    for n, m in powers:
        t = x[0] ** n * x[1] ** m
        out.append(t if n == m else t + x[0] ** m * x[1] ** n)
    return jnp.stack(out)


def facet_powers(s):
    """R2b+ facet weights l_a^{5/2} (note-r2b-facet.md)."""
    lt = jnp.stack([C_ELL + V2[a, 0] * s[0] + V2[a, 1] * s[1] for a in range(6)])
    return lt ** 2.5


def mu_log(s, a=A_FACET):
    """ln(1 + v_a . x) = ln(3 l_a(s)); the ln 3 is an additive constant."""
    x = SCALE * s
    return jnp.log(1.0 + V2[a, 0] * x[0] + V2[a, 1] * x[1])


def mu_dhhkw(s):
    return mu_log(s) + jnp.dot(jnp.asarray(CD), mu_powers(s, POWERS6))


def laplacian_op(chart, psi):
    """L[f](s) = d_i(u^{ij} d_j f), i.e. DHHKW (3.46) in our variables."""
    u = slice_potential(chart, psi)
    Hinv = lambda z: jnp.linalg.inv(jax.hessian(u)(z))

    def L(f):
        flux = lambda z: Hinv(z) @ jax.grad(f)(z)
        return lambda z: jnp.trace(jax.jacfwd(flux)(z))

    return L


def report(tag, Lvals, exact):
    """Lvals are L^ours[mu]; everything printed in DHHKW units."""
    v = TO_THEM * np.asarray(Lvals)
    mean = v.mean()
    print(f"    {tag:31s} mean {mean:.6f}  (exact {exact:.6f}, "
          f"off by {abs(mean - exact):.1e})   spread: rms "
          f"{np.sqrt(((v - mean) ** 2).mean()):.2e}  max "
          f"{np.abs(v - mean).max():.2e}")
    return mean


def build_design(L, pts, powers, facet):
    cols = [jax.jit(jax.vmap(L(lambda s, k=k, p=powers: mu_powers(s, p)[k])))(pts)
            for k in range(len(powers))]
    if facet:
        cols += [jax.jit(jax.vmap(L(lambda s, k=k: facet_powers(s)[k])))(pts)
                 for k in range(6)]
    A = np.concatenate([np.asarray(jnp.stack(cols, axis=1)),
                        -np.ones((len(pts), 1))], axis=1)
    b = -np.asarray(jax.jit(jax.vmap(L(mu_log)))(pts))
    return A, b


def refit(L, pts, powers=POWERS6, facet=False):
    """L is LINEAR in mu, so fitting (6.11) to (3.46) is linear least squares --
    no optimizer and no local minima."""
    A, b = build_design(L, pts, powers, facet)
    sol, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ sol - b
    rms = float(np.sqrt((resid ** 2).mean())) / 3.0        # -> DHHKW units
    dof = max(len(pts) - A.shape[1], 1)
    cov = np.linalg.pinv(A.T @ A) * float(resid @ resid) / dof
    sigma = np.sqrt(np.clip(np.diag(cov)[:len(powers)], 0.0, None))
    return dict(c=sol[:len(powers)], const=TO_THEM * sol[-1], rms=rms,
                cond=float(sv[0] / sv[-1]), sigma=sigma)


CHART = slice_chart(dp3(), B_DP3)
GROUP = dihedral_matrices(CHART.verts_s)


def mu_omega(s):
    """(1/4) sum_{g in D_6} mu_2(g s) = (1/2) sum_a mu_a = the potential of
    omega: the orbit of facet 2 is all six facets, with a stabilizer of order 2."""
    return 0.25 * jnp.sum(jax.vmap(lambda g: mu_dhhkw(g @ s))(GROUP))


def main():
    chart, group = CHART, GROUP
    print(f"coordinate bridge: x = {SCALE:.0f} s; our vertices in their x = "
          f"{np.round(SCALE * np.asarray(chart.verts_s), 6).tolist()}")
    print(f"their a=2 normal v = {V2[A_FACET].tolist()}; fixed by x1<->x2, which is "
          f"their c_nm = c_mn: {bool(np.allclose(V2[A_FACET], V2[A_FACET][::-1]))}")

    ss = sample_slice(jax.random.PRNGKey(5), chart, 20000, eps=1e-3)
    inner = np.asarray(jnp.max(jnp.abs(SCALE * ss), axis=1)) < 0.9
    ss_in = ss[jnp.asarray(inner)]
    print(f"samples: {len(ss)} in the hexagon, {int(inner.sum())} inside their "
          f"fitting domain |x_i| < 0.9\n")

    # ---- [A] + [anti]: their published theta_2, on the KE metric and on a wrong one
    for name, path in NPZ.items():
        psi, ch, dat = load_psi_dp3(path)
        L = laplacian_op(ch, psi)
        print(f"[A] DHHKW (6.13) theta_2 on our {name} metric "
              f"({dat['W'].shape[1]} params): harmonicity (3.46) needs a constant, "
              f"fixed to 2/3 by (6.15)")
        report("their domain |x_i|<0.9", jax.jit(jax.vmap(L(mu_dhhkw)))(ss_in), 2 / 3)
        report("whole hexagon", jax.jit(jax.vmap(L(mu_dhhkw)))(ss), 2 / 3)

    psi, ch, dat = load_psi_dp3(NPZ["deg-18"])
    L = laplacian_op(ch, psi)
    L0 = laplacian_op(ch, lambda s: 0.0)
    print("    their own fit reported 0.6672 for this constant, and rms ~1e-3 for "
          "the spread.")
    print("[anti-test] the same mu on the canonical Guillemin metric (psi=0, NOT "
          "Einstein) -- must fail:")
    report("whole hexagon", jax.jit(jax.vmap(L0(mu_dhhkw)))(ss), 2 / 3)

    # ---- [B] protected omega anchor
    print("\n[B] protected anchor omega = (1/2) sum_a theta_a (6.14): constant "
          "exactly 2")
    report("whole hexagon", jax.jit(jax.vmap(L(mu_omega)))(ss), 2.0)
    report("Guillemin psi=0 (anti-test)", jax.jit(jax.vmap(L0(mu_omega)))(ss), 2.0)

    # ---- [C] our own fit of the identical ansatz
    print("\n[C] refit of the IDENTICAL sixth-order ansatz on our deg-18 metric")
    f_hex, f_in = refit(L, ss), refit(L, ss_in)
    for tag, f in (("whole hexagon", f_hex), ("|x_i|<0.9", f_in)):
        print(f"    {tag:15s} constant {f['const']:.6f} (exact 0.666667, off by "
              f"{abs(f['const'] - 2/3):.1e})   residual rms {f['rms']:.2e}   "
              f"design cond {f['cond']:.1e}")
    print(f"    {'(n,m)':>7} {'DHHKW':>9} {'ours':>9} {'+-1sigma':>9} "
          f"{'ours-DHHKW':>11} {'in sigmas':>10} {'domain shift':>13}")
    for p, cd, co, sg, ci in zip(POWERS6, CD, f_hex["c"], f_hex["sigma"], f_in["c"]):
        print(f"    {str(p):>7} {cd:9.4f} {co:9.4f} {sg:9.4f} {co - cd:11.4f} "
              f"{abs(co - cd) / max(sg, 1e-12):10.1f} {abs(co - ci):13.4f}")
    print(f"    rms |ours-DHHKW| {np.sqrt(((f_hex['c']-CD)**2).mean()):.4f}  vs  "
          f"our own domain shift {np.sqrt(((f_hex['c']-f_in['c'])**2).mean()):.4f}  "
          f"vs  our own 1sigma {np.sqrt((f_hex['sigma']**2).mean()):.4f}")

    # ---- [D] gauge-free: the potential itself.  The logs cancel in the difference.
    print("\n[D] pointwise mu_DHHKW - mu_ours on |x_i|<0.9 (the ln terms cancel "
          "exactly, so this is a pure polynomial)")
    poly_only = jax.jit(jax.vmap(lambda s: jnp.dot(jnp.asarray(CD),
                                                   mu_powers(s, POWERS6))))(ss_in)
    den = float(jnp.max(poly_only) - jnp.min(poly_only))
    mu_all = jax.jit(jax.vmap(mu_dhhkw))(ss_in)
    print(f"    spread of the polynomial part of mu: {den:.4f}   "
          f"(spread of mu itself {float(jnp.max(mu_all) - jnp.min(mu_all)):.3f}, "
          f"dominated by the log)")
    for tag, f in (("hexagon fit", f_hex), ("|x|<0.9 fit", f_in)):
        d = jax.jit(jax.vmap(lambda s, c=jnp.asarray(f["c"]): jnp.dot(
            jnp.asarray(CD) - c, mu_powers(s, POWERS6))))(ss_in)
        d = d - jnp.mean(d)
        print(f"    {tag:12s} rms {float(jnp.sqrt(jnp.mean(d**2))):.2e} "
              f"({float(jnp.sqrt(jnp.mean(d**2))) / den:.1e} of the polynomial "
              f"spread)   max {float(jnp.max(jnp.abs(d))):.2e}")

    # ---- [E] order ladder + facet weights
    print("\n[E] how far below their residual the same construction goes "
          "(deg-18 metric, whole hexagon)")
    print(f"    {'ansatz':>26} {'n_par':>6} {'constant':>10} {'off by':>9} "
          f"{'resid rms':>10} {'cond':>9}")
    print(f"    {'DHHKW (6.13), as published':>26} {len(CD):6d} "
          f"{TO_THEM * float(np.mean(jax.jit(jax.vmap(L(mu_dhhkw)))(ss))):10.6f} "
          f"{abs(TO_THEM * float(np.mean(jax.jit(jax.vmap(L(mu_dhhkw)))(ss))) - 2/3):9.1e} "
          f"{float(np.std(TO_THEM * np.asarray(jax.jit(jax.vmap(L(mu_dhhkw)))(ss)))):10.2e} "
          f"{'--':>9}")
    for order in (6, 8, 10):
        for facet in (False, True):
            pw = sym_powers(order)
            f = refit(L, ss, powers=pw, facet=facet)
            tag = f"order {order}" + (" + facet l^5/2" if facet else "")
            print(f"    {tag:>26} {len(pw) + (6 if facet else 0):6d} "
                  f"{f['const']:10.6f} {abs(f['const'] - 2/3):9.1e} "
                  f"{f['rms']:10.2e} {f['cond']:9.1e}")


if __name__ == "__main__":
    main()
