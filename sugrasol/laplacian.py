"""Scalar Laplacian spectrum on the transverse toric Kaehler base (the
torus-invariant sector), by Rayleigh-Ritz.  Used to grade the learned dP3
metric against DHHKW hep-th/0703057 (their lambda1 = 6.322), the replacement
for the GMSW curvature-invariant collapse that dP3's cohomogeneity-2 forbids.

Geometry.  In action-angle (symplectic) coordinates the toric Kaehler metric
has unit determinant (det g = 1), so for a torus-invariant function f(s),

    Delta f = (1/sqrt g) d_j( sqrt g g^{jk} d_k f ) = d_j( u^{jk} d_k f ),
    u^{jk} = (Hess_s u)^{-1},   u = transverse symplectic potential.

The Dirichlet form  int_P u^{jk} d_j f d_k h d^2s  is self-adjoint w.r.t.
int_P f h d^2s: the boundary term vanishes because u^{jk} (coefficient of the
normal second derivative) degenerates on dP.  So the eigenproblem -Delta f =
lambda f becomes the generalized symmetric eigenproblem

    A c = lambda B c,
    A_mn = <u^{jk} d_j phi_m, d_k phi_n>_P   (stiffness),
    B_mn = <phi_m, phi_n>_P                  (mass),

for a function basis {phi_m}; the lowest eigenvalue is 0 (constant mode) and
the next is lambda1.  Being Rayleigh-Ritz, lambda1 is approached from ABOVE as
the basis grows.

Normalization.  Eigenvalues carry a 1/length^2 scale set by the Einstein
convention, so they are convention-dependent.  The dimensionless ratio
lambda/S (S = Abreu scalar curvature, = 12 in our doubled-slice normalization,
= 4 for DHHKW's Ric=g) is convention-FREE:  compare  lambda_ours * 4 / S_ours
against DHHKW's lambda.  (Anchor: for the conifold base CP^1 x CP^1 this gives
lambda1 = 6 in our normalization, computable in closed form.)
"""
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh


def polygon_quadrature(verts_s, ngl: int = 24):
    """Deterministic quadrature for the uniform (Lebesgue) measure on the convex
    slice polygon: a centroid fan triangulation + a tensor Gauss--Legendre rule
    on each triangle via the Duffy (collapsed-square) map.  Returns (nodes,
    weights) with nodes (M,2) and weights (M,) summing to the polygon area.

    Why this beats Monte-Carlo for `laplace_spectrum`/`link_charged_spectrum`.
    The stiffness/mass integrands are smooth in the interior, so a GL rule
    converges geometrically instead of as 1/sqrt(N); the conifold anchor
    (lambda1 = 6, E = 21) comes out at machine precision, versus ~0.1-0.7% and
    seed-dependent for MC.  It also removes the eps boundary margin entirely:
    the stiffness integrand carries u^{jk} = (Hess u)^{-1}, which DEGENERATES
    on partial P (the same fact that kills the boundary term), so the natural
    weight already vanishes there -- no margin, no boundary-exclusion bias
    (log.md 2026-07-24).  `ngl` is the 1d node count per direction per triangle
    (total M = d * ngl^2 for a d-gon); ngl~24 is converged to ~1e-13 for dP3.
    """
    V = np.asarray(verts_s, dtype=float)
    c = V.mean(0)
    d = len(V)
    x, wx = leggauss(ngl)
    x = 0.5 * (x + 1.0)              # nodes on [0, 1]
    wx = 0.5 * wx                    # weights on [0, 1]
    u, v = np.meshgrid(x, x, indexing="ij")
    u, v = u.ravel(), v.ravel()
    w2 = np.outer(wx, wx).ravel()
    b0, b1, b2 = 1.0 - u, u * (1.0 - v), u * v   # barycentric, Duffy Jacobian = u
    nodes, wts = [], []
    for a in range(d):
        P0, P1, P2 = c, V[a], V[(a + 1) % d]
        area = 0.5 * abs((P1[0] - P0[0]) * (P2[1] - P0[1])
                         - (P1[1] - P0[1]) * (P2[0] - P0[0]))
        pts = b0[:, None] * P0 + b1[:, None] * P1 + b2[:, None] * P2
        nodes.append(pts)
        wts.append(w2 * u * (2.0 * area))
    return jnp.asarray(np.concatenate(nodes)), jnp.asarray(np.concatenate(wts))


def _accumulate(vals, weights):
    """Contract per-node matrices `vals` (N, ...) into an integral: the plain
    Monte-Carlo mean when `weights` is None, else the quadrature-weighted sum
    sum_n w_n vals_n (weights = 1/N recovers the mean)."""
    if weights is None:
        return jnp.mean(vals, axis=0)
    return jnp.tensordot(weights, vals, axes=(0, 0))


def slice_potential(chart, psi):
    """Transverse symplectic potential u(s) = (1/2) sum_a l_a ln l_a + psi(s)
    on the (doubled) Reeb slice — the same u graded by the Abreu S check."""

    def u(s):
        t = chart.t0 + s @ chart.f
        lt = chart.cone.ells(t)
        return 0.5 * jnp.sum(lt * jnp.log(lt)) + psi(s)

    return u


def abreu_scalar(chart, psi):
    """Abreu's transverse scalar curvature S(s) = -d_j d_k u^{jk} (math/0004122),
    fourth order in u and never part of the second-order training loss."""
    u = slice_potential(chart, psi)

    def S(s):
        Hinv = lambda z: jnp.linalg.inv(jax.hessian(u)(z))
        return -jnp.einsum("jkjk->", jax.jacfwd(jax.jacfwd(Hinv))(s))

    return S


def abreu_grid(psi, chart, n: int = 240, pad: float = 0.02, eps: float = 1e-3,
               target: float = 12.0):
    """max/mean of |S/target - 1| on an n x n grid over the slice polygon.

    A DETERMINISTIC estimator, deliberately not a sample maximum: on 2026-07-26 a
    200-point Monte-Carlo maximum was found to understate the polynomial's worst
    Abreu deviation by a factor ~2 (0.011% -> 0.019%), so anything compared with
    the polynomial's number must be scored the same way.  Returns a dict."""
    verts = np.asarray(chart.verts_s)
    lo, hi = verts.min(0) - pad, verts.max(0) + pad
    GX, GY = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    pts = jnp.asarray(np.stack([GX.ravel(), GY.ravel()], axis=1))
    t_of = lambda s: chart.t0 + s @ chart.f
    inside = np.asarray(jax.vmap(lambda s: chart.cone.is_interior(t_of(s), eps))(pts))
    vals = np.asarray(jax.vmap(abreu_scalar(chart, psi))(pts))[inside.astype(bool)]
    dev = np.abs(vals / target - 1.0)
    return {"max_pct": 100 * float(dev.max()), "mean_pct": 100 * float(dev.mean()),
            "n_points": int(inside.sum())}


def laplace_spectrum(u, basis, samples, weights=None):
    """Torus-invariant scalar Laplacian eigenvalues of the metric with
    symplectic potential `u`, in the span of `basis` (s -> (m,) values).
    `samples`: (N,2) evaluation points on the slice polygon.  With
    `weights=None` these are treated as a uniform Monte-Carlo cloud (the mean);
    pass the (nodes, weights) of `polygon_quadrature` for deterministic
    quadrature (machine-precision integrals, no boundary margin -- preferred).
    Returns eigenvalues ascending; [0] ~ 0 (constant), [1] = lambda1, ..."""
    uinv = lambda s: jnp.linalg.inv(jax.hessian(u)(s))
    jac = jax.jacfwd(basis)  # s -> (m, 2)

    def a_int(s):
        J = jac(s)
        return J @ uinv(s) @ J.T

    def b_int(s):
        b = basis(s)
        return jnp.outer(b, b)

    A = np.asarray(_accumulate(jax.vmap(a_int)(samples), weights))
    B = np.asarray(_accumulate(jax.vmap(b_int)(samples), weights))
    A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
    return eigh(A, B, eigvals_only=True)


# --------------------------------------------------------------------------
# Charged sector on the SE_5 LINK (rung: KK spectrum / paper II)
# --------------------------------------------------------------------------
#
# The link Y = {<b, y> = level} of the toric Kaehler cone, full T^3 kept.
# In MSY symplectic coordinates the cone metric is block diagonal,
#     g = G_ij dy dy + G^{ij} dphi dphi,   G_ij = Hess_y G,
# so the induced link metric in chart (s, phi), y(s) = level * t(s):
#     g_ss = J^T (Hess G) J    (J = level * f^T, the affine slice Jacobian),
#     g_ss-phi = 0,
#     g_phiphi = (Hess G)^{-1}  (full 3x3 — the Reeb circle is inside T^3).
# A charged mode  psi = e^{i m.phi} F(s),  m in Z^3, has Rayleigh quotient
#     E[F] = int [ dF g_ss^{-1} dF + (m^T Hess G m) F^2 ] w d^2s
#            / int F^2 w d^2s,      w = sqrt(det g_ss / det Hess G),
# i.e. the invariant 2d problem plus a POTENTIAL term m Hess m — the charge
# feels the angular metric.  Facet regularity: near <v_a, y> = 0 the circle
# generated by v_a degenerates, and smoothness of e^{i m.phi} F forces
#     F ~ l_a^{|<v_a, m>|/2}   (flat-C^3 model: z^m ~ y^{m/2} e^{i m phi}),
# built into the basis as a weight (rule 1), not imposed as a penalty.
#
# Anchors that pin all conventions (level, charge normalization) at once:
# holomorphic monomials z^m (m in the dual cone sigma* = cone{v_a}) are
# harmonic on the cone, homogeneous of Reeb weight lambda = <b, m>, so the
# lowest link eigenvalue in the chiral sector m is EXACTLY
#     E_min(m) = lambda (lambda + 4),   lambda = <b, m>
# — no external input; misconventions break the linear/quadratic pieces
# differently and cannot fake this across several m.


def link_charged_spectrum(G3d, chart, m, basis, samples, level: float = 0.5,
                          weights=None):
    """Charged scalar Laplacian spectrum on the SE_5 link.

    G3d: full 3d cone symplectic potential y -> G(y) (ypq.cone_potential);
    chart: SliceChart (doubled Reeb slice conventions);
    m: (3,) integer T^3 charge; basis: s -> (M,) trial functions (should
    carry the facet weight — see charged_basis); samples: (N,2) evaluation
    points.  `weights=None` -> Monte-Carlo (uniform cloud mean); pass the
    (nodes, weights) of `polygon_quadrature` for deterministic quadrature
    (the chiral anchor E = lambda(lambda+4) then comes out exact -- preferred).
    Returns eigenvalues ascending (E ~ lambda(lambda+4))."""
    J = level * chart.f.T                       # (3, 2) = dy/ds
    mv = jnp.asarray(m, dtype=jnp.float64)

    def mats(s):
        y = level * (chart.t0 + s @ chart.f)
        H = jax.hessian(G3d)(y)                 # (3, 3)
        gss = J.T @ H @ J                       # (2, 2)
        sign, logdetH = jnp.linalg.slogdet(H)
        w = jnp.sqrt(jnp.linalg.det(gss)) * jnp.exp(-0.5 * logdetH)
        return jnp.linalg.inv(gss), mv @ H @ mv, w

    jac = jax.jacfwd(basis)

    def a_int(s):
        gi, pot, w = mats(s)
        Jb, bb = jac(s), basis(s)
        return w * (Jb @ gi @ Jb.T + pot * jnp.outer(bb, bb))

    def b_int(s):
        _, _, w = mats(s)
        bb = basis(s)
        return w * jnp.outer(bb, bb)

    A = np.asarray(_accumulate(jax.vmap(a_int)(samples), weights))
    B = np.asarray(_accumulate(jax.vmap(b_int)(samples), weights))
    A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
    return eigh(A, B, eigvals_only=True)


def orthonormal_basis(raw, samples, tol: float = 1e-9):
    """SVD-orthonormalize a raw basis s -> (M,) on the given samples
    (conditioning fix, same logic as ypq.whiten_poly); drops directions
    below tol * S[0].  Returns s -> (r,)."""
    A = jax.vmap(raw)(samples)
    _, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    r = int(jnp.sum(S > tol * S[0]))
    W = Vt[:r].T / S[:r]
    return lambda s: raw(s) @ W


def charged_basis(chart, m, degree, samples, tol: float = 1e-9):
    """Facet-weighted polynomial basis for charge sector m:
        W(s) * {s^i} ,   W = prod_a l_a(t(s))^{|<v_a, m>|/2},
    orthonormalized on the samples.  W carries the exact boundary behaviour
    of smooth charged modes; polynomials carry the rest."""
    v = chart.cone.normals
    expo = 0.5 * jnp.abs(v @ jnp.asarray(m, dtype=jnp.float64))
    powers = [(i, j) for tot in range(degree + 1)
              for i in range(tot + 1) for j in [tot - i]]

    def raw(s):
        t = chart.t0 + s @ chart.f
        W = jnp.prod(chart.cone.ells(t) ** expo)
        return W * jnp.stack([s[0] ** i * s[1] ** j for i, j in powers])

    return orthonormal_basis(raw, samples, tol)


def chiral_lattice_points(cone, b, lam_max: float, box: int = 6):
    """T^3 charges of the chiral (holomorphic) tower: integer points of the
    MOMENT CONE sigma itself, <v_a, m> >= 0 for all facet normals, with
    Reeb weight lambda = <b, m> in (0, lam_max].

    Why sigma and not its dual: in symplectic coordinates the angle phi_i is
    conjugate to y_i, so a monomial's charge vector lives in the SAME space
    as y.  Worked example (flat C^3, CY gauge, flat_cn_cy_gauge): the
    charges of z_1, z_2, z_3 are (1,-1,-1), (0,1,0), (0,0,1) — the *edge
    rays of the moment cone* (all with <b', m> = 1, as Delta(z_i) = 1
    demands) — and NOT points of the dual cone{v_a}.  Testing against the
    dual cone silently drops the entire chiral tower (found the hard way).
    Returns [(lambda, m)] sorted by lambda."""
    v = np.asarray(cone.normals)
    bb = np.asarray(b)
    out = []
    rng = range(-box, box + 1)
    for m1 in rng:
        for m2 in rng:
            for m3 in rng:
                mm = np.array([m1, m2, m3], dtype=float)
                lam = float(bb @ mm)
                if lam <= 1e-12 or lam > lam_max:
                    continue
                if np.all(v @ mm >= -1e-12):
                    out.append((lam, (m1, m2, m3)))
    return sorted(out)
