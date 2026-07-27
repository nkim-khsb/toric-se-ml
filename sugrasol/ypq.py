"""Y^{p,q} rung: learn the cone potential correction psi on the Reeb slice.

Architecture (fixed by the T^{1,1} cone verification, see log.md 2026-07-11):

    G_theta(y) = (1/2) l_b ln l_b
               + l_b * [ (1/2) sum_a <v_a,t> ln <v_a,t>  +  psi_theta(s(t)) ],
    t = y / l_b,   s = 2d affine coordinates on the doubled slice {<b,t>=1}.

  - The Guillemin part carries all facet log singularities (MSY eq. (2.59)).
  - psi_theta is smooth on the slice polygon.  For the conifold, psi = 0 is
    exact (machine-precision residual); for Y^{p,q} (q != 0) psi != 0 must be
    LEARNED — this is the first rung where the network does real work.

Loss = MSY eq. (2.54) residual with learnable constant c:
    L(theta, c) = mean_t [ ln det Hess_y G + 2 dG/dy_1 + c ]^2 .
Residual is exactly l_b-independent for this ansatz, so sampling on the
slice suffices.

Gauge structure: adding an affine function of t to G^T is a linear function
of y — an exact flat direction (verified for the transverse problem, same
mechanism here; the constant is degenerate with c).  We therefore gauge-fix
psi by subtracting the affine function interpolating psi at three fixed
anchor points (ansatz-level fix, CLAUDE.md rule 1/4).

Irregular Reeb (e.g. Y^{2,1}) needs nothing special: the moment-cone
description does not care whether b is rational (MSY Sec. 2.5; notes §3.3).
"""
from __future__ import annotations

import warnings
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cone import MomentCone

jax.config.update("jax_enable_x64", True)


class SliceChart(NamedTuple):
    """2d affine chart s on the doubled Reeb slice {<b, t> = 1}."""

    cone: MomentCone
    b: jnp.ndarray
    t0: jnp.ndarray  # base point, <b, t0> = 1
    f: jnp.ndarray  # (2, 3) orthonormal basis of ker(b)
    verts_s: jnp.ndarray  # (d, 2) polygon vertices in s-coordinates
    anchors: jnp.ndarray  # (3, 2) gauge anchor points


def slice_chart(cone: MomentCone, b: jnp.ndarray) -> SliceChart:
    _, _, Vt = jnp.linalg.svd(b[None, :])
    f = Vt[1:]
    t0 = b / (b @ b)
    v = cone.normals
    d = v.shape[0]
    rays = jnp.stack([jnp.cross(v[a], v[(a + 1) % d]) for a in range(d)])
    rays = rays * jnp.sign(rays @ b)[:, None]
    verts = rays / (rays @ b)[:, None]
    verts_s = (verts - t0) @ f.T
    centroid = jnp.mean(verts_s, axis=0)
    anchors = jnp.stack(
        [centroid,
         centroid + 0.4 * (verts_s[0] - centroid),
         centroid + 0.4 * (verts_s[1] - centroid)]
    )
    return SliceChart(cone, b, t0, f, verts_s, anchors)


def t_of_s(chart: SliceChart, s: jnp.ndarray) -> jnp.ndarray:
    return chart.t0 + s @ chart.f


def sample_slice(key, chart: SliceChart, n_samples: int, eps: float = 1e-3):
    """Uniform s-samples in the slice polygon (rejection from bounding box)."""
    lo = jnp.min(chart.verts_s, axis=0)
    hi = jnp.max(chart.verts_s, axis=0)
    m = 16 * n_samples
    u = jax.random.uniform(key, (m, 2), dtype=jnp.float64)
    s = lo + u * (hi - lo)
    t = jax.vmap(lambda ss: t_of_s(chart, ss))(s)
    ok = jax.vmap(lambda tt: chart.cone.is_interior(tt, eps))(t)
    idx = jnp.nonzero(ok, size=m, fill_value=-1)[0]
    if int(jnp.sum(ok)) < n_samples:
        raise RuntimeError("increase oversampling")
    return s[idx[:n_samples]]


def gauge_fixed(psi: Callable, anchors: jnp.ndarray) -> Callable:
    """Subtract the affine function matching psi at the three anchors."""

    def fixed(s):
        vals = jnp.stack([psi(a) for a in anchors])  # (3,)
        # affine u(s) = m0 + m1.s solving u(anchor_i) = vals_i
        M = jnp.concatenate([jnp.ones((3, 1)), anchors], axis=1)  # (3,3)
        coef = jnp.linalg.solve(M, vals)
        return psi(s) - coef[0] - coef[1:] @ s

    return fixed


def cone_potential(chart: SliceChart, psi: Callable) -> Callable:
    """G_theta(y); psi must already be gauge-fixed (or zero)."""
    cone, b, t0, f = chart.cone, chart.b, chart.t0, chart.f

    def G(y):
        lb = b @ y
        t = y / lb
        lt = cone.ells(t)
        GT = 0.5 * jnp.sum(lt * jnp.log(lt))
        s = f @ (t - t0)
        return 0.5 * lb * jnp.log(lb) + lb * (GT + psi(s))

    return G


def residual_on_slice(chart: SliceChart, psi: Callable, s: jnp.ndarray):
    """MSY (2.54) residual without c, evaluated at y = t(s) (l_b = 1)."""
    G = cone_potential(chart, psi)
    y = t_of_s(chart, s)
    grad = jax.grad(G)(y)
    hess = jax.hessian(G)(y)
    _, logdet = jnp.linalg.slogdet(hess)
    return logdet + 2.0 * grad[0]


def loss_fn(chart: SliceChart, psi: Callable, c, ss: jnp.ndarray):
    r = jax.vmap(lambda s: residual_on_slice(chart, psi, s))(ss)
    return jnp.mean((r + c) ** 2)


# ---------------------------------------------------------------- ansaetze
def mlp_psi(params) -> Callable:
    def psi(s):
        h = s
        for W, bb in params[:-1]:
            h = jnp.tanh(W @ h + bb)
        W, bb = params[-1]
        return (W @ h + bb)[0]

    return psi


def poly_psi(coeffs: jnp.ndarray, degree: int) -> Callable:
    """Polynomial in (s1, s2), monomials with 2 <= i+j <= degree only
    (constant and linear terms are pure gauge, excluded by construction)."""
    powers = [(i, j) for tot in range(2, degree + 1)
              for i in range(tot + 1) for j in [tot - i]]

    def psi(s):
        mons = jnp.stack([s[0] ** i * s[1] ** j for i, j in powers])
        return coeffs @ mons

    return psi


def n_poly_coeffs(degree: int) -> int:
    return sum(tot + 1 for tot in range(2, degree + 1))


# --------------------------------------------------- conditioned poly basis
def _poly_powers(degree: int):
    return [(i, j) for tot in range(2, degree + 1)
            for i in range(tot + 1) for j in [tot - i]]


def _monomials(s, powers):
    return jnp.stack([s[0] ** i * s[1] ** j for i, j in powers])


def whiten_poly(degree: int, samples: jnp.ndarray):
    """Orthonormalize the degree-<=deg monomial basis over `samples`.

    Returns (powers, W) such that  features(s) = _monomials(s, powers) @ W
    have (near-)orthonormal columns on `samples`.  This is a pure LINEAR
    reparametrization of the same polynomial space (constant/linear excluded
    by construction, so gauge structure is untouched) — it only fixes the
    Vandermonde ill-conditioning of raw monomials on the off-center slice
    polygon, which was stalling L-BFGS (see log.md 2026-07-12: deg-12 failed
    to beat deg-10).  CLAUDE.md rule 1: same ansatz space, better conditioned.
    """
    powers = _poly_powers(degree)
    A = jax.vmap(lambda s: _monomials(s, powers))(samples)  # (N, K)
    _, R = jnp.linalg.qr(A)
    W = jnp.linalg.solve(R, jnp.eye(R.shape[0]))            # R^{-1}
    return powers, W


def ortho_psi(coeffs: jnp.ndarray, powers, W: jnp.ndarray) -> Callable:
    def psi(s):
        return coeffs @ (_monomials(s, powers) @ W)

    return psi


# ------------------------------------------------ dihedral-symmetric ansatz
def dihedral_matrices(verts_s: jnp.ndarray) -> jnp.ndarray:
    """The dihedral symmetry group of a polygon, as 2x2 linear maps on the
    slice coordinates s (vertices assumed centred: centroid = 0).

    The polygon (e.g. the dP3 hexagon, DHHKW's D6) is combinatorially regular
    but metrically skewed in the orthonormal s-frame, so its symmetries are
    NOT Euclidean rotations — they are the linear maps that cyclically permute
    the vertices. We recover them directly: the "rotation" sends vertex a to
    a+1, one "reflection" reverses the cycle fixing vertex 0. Returns (2d,2,2)."""
    V = verts_s
    d = V.shape[0]

    def linmap(src, dst):  # 2x2 A with A @ V[src_i] = V[dst_i], i=0,1
        S = jnp.stack([V[src[0]], V[src[1]]], axis=1)
        D = jnp.stack([V[dst[0]], V[dst[1]]], axis=1)
        return D @ jnp.linalg.inv(S)

    A_rot = linmap((0, 1), (1, 2 % d))
    A_ref = linmap((0, 1), (0, d - 1))
    mats, R = [], jnp.eye(2)
    for _ in range(d):
        mats.append(R)
        mats.append(R @ A_ref)
        R = R @ A_rot
    return jnp.stack(mats)  # (2d, 2, 2)


def invariant_dimension(group: jnp.ndarray, degree: int, tol: float = 1e-6):
    """Dimension of the G-invariant polynomials of degree 2..`degree`, exactly,
    by Molien's series -- the ground truth the SVD rank below must reproduce.

        sum_k dim(Inv_k) t^k = (1/|G|) sum_{g in G} 1 / det(1 - t g),

    and for 2x2 g the expansion 1/det(1 - t g) = sum_k c_k t^k obeys
    c_0 = 1, c_1 = tr g, c_k = (tr g) c_{k-1} - (det g) c_{k-2}.  For the dP3
    hexagon this returns 14 at degree 14 and 21 at degree 18, matching the count
    of DHHKW's U^i V^j (deg U = 2, deg V = 6, their eq. (5.1)).

    Returns None when the dimensions come out non-integral, which means `group`
    is not a finite matrix group in this representation -- e.g.
    `dihedral_matrices` applied to a polygon whose centroid is not the origin,
    where the symmetries are affine and not linear.  Better no check than a
    wrong one.
    """
    G = np.asarray(group, dtype=float)
    dims = np.zeros(degree + 1)
    for g in G:
        tr, dt = float(np.trace(g)), float(np.linalg.det(g))
        c = np.zeros(degree + 1)
        c[0] = 1.0
        if degree >= 1:
            c[1] = tr
        for k in range(2, degree + 1):
            c[k] = tr * c[k - 1] - dt * c[k - 2]
        dims += c
    dims /= len(G)
    if np.max(np.abs(dims - np.round(dims))) > tol:
        return None
    return int(round(dims[2:].sum()))


def whiten_sym_poly(degree: int, samples: jnp.ndarray, group: jnp.ndarray,
                    tol: float = 1e-14):
    """Orthonormal basis of the GROUP-INVARIANT degree-<=deg polynomials over
    `samples`.  Features are group-averaged monomials; symmetrization makes the
    monomial family rank-deficient, so we orthonormalize its column space by SVD
    (not QR).  Constant/linear are absent by construction (monomials start at
    degree 2) AND linear carries no invariant under D6 -> the affine gauge is
    removed by the symmetry itself; no anchor gauge-fixing (which would break
    invariance).  Returns (powers, W, group); psi = coeffs @ (feat(s) @ W).

    On `tol`.  The group-averaged monomials are severely collinear -- on the dP3
    hexagon their singular values span ten orders -- so this one RELATIVE
    threshold decides how many genuine invariant directions survive, and the
    default used to silently discard some of them: 11 of 14 at degree 14, 17 of
    21 at degree 18, which cost a factor 25 in held-out residual and 3.5 in the
    Einstein-condition error (log.md 2026-07-26; the same conditioning trap as
    the 2026-07-12 Y^{p,q} "degree-10 mirage" and the 2026-07-20 "tol=1e-9
    truncates deg>14", whose fix moved the tolerance without removing the
    truncation).  The default is now 1e-14, and the retained rank is checked
    against Molien's exact count so that a truncation cannot pass unnoticed.
    """
    powers = _poly_powers(degree)

    def feat(s):
        gs = jnp.einsum("gij,j->gi", group, s)               # orbit of s
        return jnp.mean(jax.vmap(lambda g: _monomials(g, powers))(gs), axis=0)

    A = jax.vmap(feat)(samples)                               # (N, K)
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))                         # invariant dim
    expect = invariant_dimension(group, degree)
    if expect is not None and r < expect:
        warnings.warn(
            f"whiten_sym_poly: kept {r} of the {expect} invariant directions at "
            f"degree {degree} (tol={tol:.0e}); the discarded ones are real, not "
            f"noise. Lower tol or use more samples.", RuntimeWarning, stacklevel=2)
    W = Vt[:r].T / S[:r]                                     # (K, r), cols orthonormal on samples
    return powers, W, group


def sym_ortho_psi(coeffs: jnp.ndarray, powers, W: jnp.ndarray,
                  group: jnp.ndarray) -> Callable:
    def psi(s):
        gs = jnp.einsum("gij,j->gi", group, s)
        feat = jnp.mean(jax.vmap(lambda g: _monomials(g, powers))(gs), axis=0)
        return coeffs @ (feat @ W)

    return psi
