"""Toric Kaehler cones a la MSY (hep-th/0503183). Equation-checked (rule 6).

Conventions (MSY; their moment coordinates are y, matching this module):
  Moment cone C = {y : l_a(y) = <v_a, y> >= 0}, primitive inward normals v_a.
  CY (Gorenstein) gauge, eq. (2.55):  v_a = (1, w_a),  w_a in Z^{n-1}.
  Reeb vector b in interior of dual cone; Sasakian radius {r=1} is the
  characteristic hyperplane  <b, y> = 1/2   (eq. (2.68)).

Ricci-flatness in symplectic coordinates, eq. (2.54):
      det(G_ij) = exp(2 gamma . dG/dy - c),
with gamma fixed by smoothness (eqs. (2.59)-(2.61)) to gamma = (-1, 0, ..., 0),
whence (b, gamma) = -n gives b_1 = n (eq. (2.62)).  We implement the residual

      R[G](y) = ln det Hess G + 2 dG/dy_1 + c .

Volume (protected quantity), eqs. (2.74) and (3.26) (n = 3):
      vol(Y) = 2n (2pi)^n vol(Delta_b),
      vol(Y) = (pi^3/b_1) sum_a det(v_{a-1},v_a,v_{a+1})
               / [ det(b,v_{a-1},v_a) det(b,v_a,v_{a+1}) ]        (cyclic).
Reeb determination: minimize vol(Delta_b) over b with b_1 = n (their Z[b];
strictly convex, unique critical point — Section 3.4).

Regular/quasi-regular cone potential (decomposition ansatz):
      G(y) = (1/2) l_b ln l_b + l_b * G^T(t),   t = y / l_b,
with G^T defined on the "doubled slice" Q~ = C cap {<b,t> = 1} using the
restricted facet functions <v_a, t>.  Exact for flat C^n (check in tests);
for the conifold this reproduces the Ricci-flat cone of T^{1,1} (tested).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MomentCone:
    """Facet normals (d, n), rows v_a in cyclic (adjacent-facet) order."""

    normals: jnp.ndarray

    @property
    def n(self) -> int:
        return self.normals.shape[1]

    def ells(self, y: jnp.ndarray) -> jnp.ndarray:
        return self.normals @ y

    def is_interior(self, y: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
        return jnp.all(self.ells(y) > eps)


# ---------------------------------------------------------------- toric data
def conifold() -> MomentCone:
    """MSY eq. (3.27), cyclic order. Cone over T^{1,1}."""
    return MomentCone(jnp.array(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    ))


def ypq(p: int, q: int) -> MomentCone:
    """MSY eq. (3.31), cyclic order. Cone over Y^{p,q}."""
    return MomentCone(jnp.array(
        [[1.0, 0.0, 0.0],
         [1.0, float(p - q - 1), float(p - q)],
         [1.0, float(p), float(p)],
         [1.0, 1.0, 0.0]]
    ))


B_CONIFOLD = jnp.array([3.0, 1.5, 1.5])  # MSY eq. (3.29)


def b_ypq(p: int, q: int) -> jnp.ndarray:
    """MSY eqs. (3.33)-(3.34)."""
    linv = (3.0 * q**2 - 2.0 * p**2 + p * jnp.sqrt(4.0 * p**2 - 3.0 * q**2)) / q
    return jnp.array([3.0, 0.5 * (3 * p - 3 * q + linv), 0.5 * (3 * p - 3 * q + linv)])


def vol_ypq(p: int, q: int) -> jnp.ndarray:
    """MSY eq. (3.35)."""
    s = jnp.sqrt(4.0 * p**2 - 3.0 * q**2)
    return (q**2 * (2 * p + s)) / (3 * p**2 * (3 * q**2 - 2 * p**2 + p * s)) * jnp.pi**3


def dp3() -> MomentCone:
    """Cone over the SE_5 whose Kaehler-Einstein base is dP3 (degree-6 del Pezzo,
    P^2 blown up at 3 points). Normals v_a = (1, w_a), w_a = dP3 fan rays in
    cyclic order, from DHHKW hep-th/0703057 eq. (3.49). Six facets — the first
    rung with a polytope beyond the quadrilateral. Smooth: det[w_a, w_{a+1}] = 1
    for all a (verified). dP3 is the largest del Pezzo admitting a Kaehler-
    Einstein metric (dP1, dP2 do not) -> regular SE, DHHKW numerical reference."""
    w = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]
    return MomentCone(jnp.array([[1.0, float(a), float(b)] for (a, b) in w]))


# Regular (rational) Reeb by MSY volume minimization + hexagon D6 symmetry:
# b = (3, 0, 0). Protected: vol(Y) = 2 pi^3/9, vol(Sigma_a) = 2 pi^2/9 (x6).
# (DHHKW surface normalization: vol(dP3) = 2 pi^2 c1^2 = 12 pi^2, c1^2 = 6.)
B_DP3 = jnp.array([3.0, 0.0, 0.0])


def dp2() -> MomentCone:
    """Cone over dP2 (degree-7 del Pezzo, P^2 blown up at 2 points). Fan rays =
    dP3 hexagon minus one ray (blow down the (0,-1) point). Five facets, smooth
    (adjacent-pair det[w_a, w_{a+1}] = 1). Only a Z2 lattice symmetry (the
    crystallographic restriction forbids a 5-fold rotation).

    dP2 admits NO Kaehler-Einstein metric (Matsushima/Futaki obstruction). Two
    experiments live on this one cone, differing only in the Reeb:
      - B_DP2_REG = (3,0,0): the REGULAR Reeb, base = the smooth surface dP2.
        The transverse equation is 'KE on dP2' -> NO solution (negative control).
      - the volume-minimizing (irregular) Reeb from minimize_reeb: the genuine
        Sasaki-Einstein metric (Futaki-Ono-Wang existence) -> a solution exists.
    See experiments/dp2/lit-survey.md."""
    w = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1)]
    return MomentCone(jnp.array([[1.0, float(a), float(b)] for (a, b) in w]))


# Regular Reeb (base = smooth dP2). NOT the volume minimizer (dP2 has no KE), so
# the transverse KE equation has no solution here -> the honest negative control.
# The Sasaki-Einstein (irregular) Reeb b~(3,0,0.3312) comes from minimize_reeb.
B_DP2_REG = jnp.array([3.0, 0.0, 0.0])


# ------------------------------------------------------------------- volumes
def vol_Y(cone: MomentCone, b: jnp.ndarray) -> jnp.ndarray:
    """vol(Y), MSY eq. (3.26). n = 3 only; normals must be in cyclic order."""
    v = cone.normals
    d = v.shape[0]
    total = 0.0
    for a in range(d):
        vm, v0, vp = v[(a - 1) % d], v[a], v[(a + 1) % d]
        num = jnp.linalg.det(jnp.stack([vm, v0, vp]))
        den = jnp.linalg.det(jnp.stack([b, vm, v0])) * jnp.linalg.det(
            jnp.stack([b, v0, vp])
        )
        total = total + num / den
    # |.|: the formula is orientation-sensitive; MSY list (3.27) and (3.31)
    # use opposite cyclic orientations, volume must be positive either way.
    return jnp.abs(jnp.pi**3 / b[0] * total)


def vol_sigma(cone: MomentCone, b: jnp.ndarray) -> jnp.ndarray:
    """vol(Sigma_a) for all facets, MSY eq. (3.25) (n = 3, cyclic order):
        vol(Sigma_a) = 2 pi^2 (v_{a-1},v_a,v_{a+1})
                       / [ (b,v_{a-1},v_a)(b,v_a,v_{a+1}) ].
    Sigma_a = link of the toric divisor over facet a — a COMPLEX submanifold's
    link, hence calibrated: its volume is symplectic (b-only, psi-independent).
    Protected quantity: grades b (refined, per-facet), NOT the learned metric.
    Identity: vol(Y) = (pi / 2 b_1) sum_a vol(Sigma_a)."""
    v = cone.normals
    d = v.shape[0]
    out = []
    for a in range(d):
        vm, v0, vp = v[(a - 1) % d], v[a], v[(a + 1) % d]
        num = jnp.linalg.det(jnp.stack([vm, v0, vp]))
        den = jnp.linalg.det(jnp.stack([b, vm, v0])) * jnp.linalg.det(
            jnp.stack([b, v0, vp])
        )
        out.append(jnp.abs(2.0 * jnp.pi**2 * num / den))
    return jnp.stack(out)


def vol_delta(cone: MomentCone, b: jnp.ndarray) -> jnp.ndarray:
    """Euclidean volume of Delta_b = C cap {<b,y> <= 1/2}, via eq. (2.74)."""
    n = cone.n
    return vol_Y(cone, b) / (2 * n * (2 * jnp.pi) ** n)


def minimize_reeb(cone: MomentCone, b23_init, steps: int = 400, lr: float = 5e-3):
    """Reeb by volume minimization at b_1 = n = 3 (MSY Sec. 3.4; strictly
    convex, unique critical point in the interior of the dual cone)."""
    import optax

    def obj(b23):
        return vol_Y(cone, jnp.concatenate([jnp.array([3.0]), b23]))

    b23 = jnp.asarray(b23_init, dtype=jnp.float64)
    opt = optax.adam(lr)
    state = opt.init(b23)
    g = jax.jit(jax.grad(obj))
    for _ in range(steps):
        upd, state = opt.update(g(b23), state)
        b23 = optax.apply_updates(b23, upd)
    return jnp.concatenate([jnp.array([3.0]), b23])


# ----------------------------------------------------- Ricci-flat MA residual
def cone_ma_residual(G: Callable, y: jnp.ndarray, c: float) -> jnp.ndarray:
    """MSY eq. (2.54) with gamma = (-1,0,...,0) (eq. (2.61)):
    R = ln det Hess G + 2 dG/dy_1 + c ; zero iff the cone is Ricci-flat."""
    grad = jax.grad(G)(y)
    hess = jax.hessian(G)(y)
    sign, logdet = jnp.linalg.slogdet(hess)
    return logdet + 2.0 * grad[0] + c


# ------------------------------------------------------------ cone potentials
def flat_cn_cy_gauge(n: int = 3) -> tuple[Callable, MomentCone, jnp.ndarray]:
    """Flat C^n written in the CY gauge v_a = (1, w_a).

    Octant frame: G(x) = (1/2) sum x_i ln x_i, normals e_a, Reeb (1,...,1).
    Change of moment coordinates y' = A y transforms normals by A^{-T}.
    With A^{-1} = [[1,0,...],[1,1,0,...],[1,0,1,...],...] the new normals are
    (1,0,..),(1,1,0,..),(1,0,1,..) — CY gauge — and b' = A^{-T} (1,..,1)
    has b'_1 = n (checks eq. (2.62))."""
    Ainv = jnp.eye(n).at[:, 0].set(1.0)  # columns: first column all ones
    A = jnp.linalg.inv(Ainv)
    normals = (jnp.linalg.inv(A).T @ jnp.eye(n)).T  # rows A^{-T} e_a
    b = jnp.linalg.inv(A).T @ jnp.ones(n)

    def G(yp):
        y = Ainv @ yp
        return 0.5 * jnp.sum(y * jnp.log(y))

    return G, MomentCone(normals), b


def regular_cone_potential(cone: MomentCone, b: jnp.ndarray) -> Callable:
    """Decomposition ansatz G = (1/2) l_b ln l_b + l_b G^T(y / l_b),
    with G^T = (1/2) sum_a <v_a,t> ln <v_a,t> on the doubled slice.
    Exact for flat C^n; conjectured (and tested) exact for the conifold."""

    def G(y):
        lb = b @ y
        t = y / lb
        lt = cone.ells(t)
        GT = 0.5 * jnp.sum(lt * jnp.log(lt))
        return 0.5 * lb * jnp.log(lb) + lb * GT

    return G


def sample_cone(key, cone: MomentCone, b: jnp.ndarray, n_samples: int,
                eps: float = 1e-3, lb_range=(0.5, 1.5)) -> jnp.ndarray:
    """Points y = l_b * t with t uniform on the doubled slice {<b,t>=1} and
    l_b uniform in lb_range. (Convenience measure for residual checks, not
    the geometric measure.)"""
    n = cone.n
    # basis of ker(b)
    _, _, Vt = jnp.linalg.svd(b[None, :])
    f = Vt[1:]  # (n-1, n)
    t0 = b / (b @ b)  # <b, t0> = 1
    # adaptive bounding box from the vertices of the doubled slice {<b,t>=1}:
    # edge rays u_a = v_a x v_{a+1} (n=3, cyclic), oriented so <b,u_a> > 0.
    v = cone.normals
    d = v.shape[0]
    rays = jnp.stack([jnp.cross(v[a], v[(a + 1) % d]) for a in range(d)])
    rays = rays * jnp.sign(rays @ b)[:, None]
    verts = rays / (rays @ b)[:, None]  # t-vertices of the slice polygon
    sv = (verts - t0) @ f.T  # slice coordinates of the vertices
    lo_s, hi_s = jnp.min(sv, axis=0), jnp.max(sv, axis=0)
    m = 16 * n_samples
    k1, k2 = jax.random.split(jax.random.PRNGKey(0) if key is None else key)
    s = lo_s + jax.random.uniform(k1, (m, n - 1), dtype=jnp.float64) * (hi_s - lo_s)
    t = t0 + s @ f
    ok = jax.vmap(lambda tt: cone.is_interior(tt, eps))(t)
    idx = jnp.nonzero(ok, size=m, fill_value=-1)[0]
    if int(jnp.sum(ok)) < n_samples:
        raise RuntimeError("increase oversampling / bounding box")
    t = t[idx[:n_samples]]
    lb = jax.random.uniform(k2, (n_samples, 1), minval=lb_range[0],
                            maxval=lb_range[1], dtype=jnp.float64)
    return lb * t
