"""Direct measurement of the claim behind section 2.3 / 3.3 of paper1: the raw
monomial basis on an off-centre slice polygon is ill-conditioned, and the
orthonormalization of `sugrasol.ypq.whiten_poly` cures exactly that.

Section 3.3 argues ill-conditioning from symptoms (degree 12 not beating degree
10; fitted coefficients of order 1 against sup|psi| ~ 0.05).  The condition
number of the design matrix is the direct measurement, and that is what this
script reports.

Design matrix: A_{pk} = (s_1^(p))^{i_k} (s_2^(p))^{j_k}, one row per sample point
and one column per monomial.  Its entries are numbers, not polynomials: A is the
matrix of the linear map from a coefficient vector a to the sampled values,
(A a)_p = psi(s^(p)), so ||A a||^2 = sum_p psi(s^(p))^2 and the singular values of
A measure how much that map can stretch or shrink a coefficient vector.  QR gives
A = QR with Q^T Q = I, so the orthonormalized basis is q_k = sum_l m_l (R^{-1})_{lk}
and cond(Q) = 1 by construction.  Nothing about the function space changes.

Self-validating: the Reeb vectors are found here by volume minimization (MSY
hep-th/0503183, Delta = C ∩ {2<b,y> <= 1}) and asserted against the closed forms
b_2 = sqrt(13)-1 for Y^{2,1} and 28/3 for Y^{7,3}.  The slice chart mirrors
sugrasol/ypq.py::slice_chart (SVD of b, t0 = b/|b|^2, s = f (t - t0)); it is
reimplemented in numpy so the script needs no jax.

Run:  python experiments/ypq/basis_conditioning.py
"""
import numpy as np

DEGREES = (8, 10, 12, 14)
NSAMPLE = 4096          # the sample size of the Y^{p,q} production runs
SEED = 0


def normals(w):
    """Calabi-Yau gauge v_a = (1, w_a)."""
    return np.array([(1.0, float(a), float(b)) for a, b in w])


def ypq_normals(p, q):
    """MSY section 3: v_1..v_4 for Y^{p,q}."""
    return normals([(0, 0), (p - q - 1, p - q), (p, p), (1, 0)])


def slice_verts(v, b):
    """Vertices of P = {t in C : <b,t> = 1} in the chart s, or None if b is not
    interior to the fan cone (in which case the slice is not a finite polygon
    and any area computed from it is meaningless -- the trap that a naive
    volume minimization walks into)."""
    d = len(v)
    rays = np.stack([np.cross(v[a], v[(a + 1) % d]) for a in range(d)])
    pb = rays @ b
    if np.any(np.abs(pb) < 1e-12):
        return None
    verts = (rays * np.sign(pb)[:, None]) / ((rays * np.sign(pb)[:, None]) @ b)[:, None]
    if not np.all(np.isfinite(verts)) or (verts @ v.T).min() < -1e-9:
        return None
    _, _, vt = np.linalg.svd(b[None, :])
    return (verts - b / (b @ b)) @ vt[1:].T


def polygon_area(V):
    n = len(V)
    return abs(sum(V[i][0] * V[(i + 1) % n][1] - V[(i + 1) % n][0] * V[i][1]
                   for i in range(n))) / 2


def reeb(p, q):
    """b = (3, t, t) minimizing vol(Delta); Y^{p,q} has b_2 = b_3 by symmetry.

    vol(Delta) = (1/3) * dist(0, plane) * area = area(P) / (24 |b|), since the
    link slice sits at <b,y> = 1/2, i.e. at half of P."""
    v = ypq_normals(p, q)

    def vol(t):
        b = np.array([3.0, t, t])
        V = slice_verts(v, b)
        return np.inf if V is None else polygon_area(V) / (24 * np.linalg.norm(b))

    grid = np.linspace(1e-3, 20.0, 40000)
    t = grid[int(np.nanargmin([vol(x) for x in grid]))]
    lo, hi = max(grid[0], t - 0.02), t + 0.02
    for _ in range(300):                       # golden-section-style trisection
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, m2) if vol(m1) < vol(m2) else (m1, hi)
    return (lo + hi) / 2, v


def uniform_in_polygon(V, n, seed=SEED):
    rng = np.random.default_rng(seed)
    P = V if sum(V[i][0] * V[(i + 1) % len(V)][1] - V[(i + 1) % len(V)][0] * V[i][1]
                 for i in range(len(V))) > 0 else V[::-1]

    def inside(pt):
        return all((P[(i + 1) % len(P)][0] - P[i][0]) * (pt[1] - P[i][1])
                   - (P[(i + 1) % len(P)][1] - P[i][1]) * (pt[0] - P[i][0]) >= 0
                   for i in range(len(P)))

    lo, hi = V.min(axis=0), V.max(axis=0)
    out = []
    while len(out) < n:
        c = rng.uniform(lo, hi)
        if inside(c):
            out.append(c)
    return np.array(out)


def design_matrix(S, degree):
    powers = [(i, j) for tot in range(2, degree + 1)
              for i in range(tot + 1) for j in [tot - i]]
    return np.stack([S[:, 0] ** i * S[:, 1] ** j for i, j in powers], axis=1), powers


def main():
    # ---- validate the Reeb finder against closed forms -------------------
    for (p, q), ref, name in [((2, 1), np.sqrt(13) - 1, "sqrt(13)-1"),
                              ((7, 3), 28 / 3, "28/3")]:
        t, _ = reeb(p, q)
        rel = abs(t - ref) / ref
        print(f"Y^({p},{q}): b_2 = {t:.10f}  vs {name} = {ref:.10f}  (rel {rel:.1e})")
        assert rel < 1e-7, (p, q, t, ref)

    # ---- conditioning on the Y^{3,2} slice --------------------------------
    t, v = reeb(3, 2)
    V = slice_verts(v, np.array([3.0, t, t]))
    print(f"\nY^(3,2): b = (3, {t:.7f}, {t:.7f});  "
          f"s_1 in [{V[:,0].min():.3f}, {V[:,0].max():.3f}]  "
          f"(off-centre, as quoted in section 2.3)")
    assert abs(V[:, 0].min() + 0.861) < 5e-3 and abs(V[:, 0].max() - 0.204) < 5e-3

    S = uniform_in_polygon(V, NSAMPLE)
    print(f"\n{NSAMPLE} uniform samples on that polygon:\n")
    print("  deg   #monomials   cond(A_monomial)   cond(Q_orthonormal)")
    for deg in DEGREES:
        A, powers = design_matrix(S, deg)
        Q, _ = np.linalg.qr(A)
        cA, cQ = np.linalg.cond(A), np.linalg.cond(Q)
        print(f"  {deg:3d}   {len(powers):10d}   {cA:16.2e}   {cQ:19.4f}")
        assert cQ < 1.0001, cQ
    print("\ncond(A) grows by ~2 orders of magnitude every 2 degrees; "
          "orthonormalization sets it to 1 by construction.")

    # The ratio is a property of (polygon, degree), not of the sample: (1/N)A^T A
    # estimates the L^2(P) Gram matrix of the monomials, so sigma_max/sigma_min
    # converges as N grows and does not depend on the seed.
    print("\nsample-independence of the ratio:\n")
    print(f"  {'N':>7}" + "".join(f"{'seed ' + str(k):>13}" for k in (0, 1, 2))
          + "    deg")
    for deg in (10, 12):
        for N in (256, 1024, 4096, 16384):
            row = []
            for seed in (0, 1, 2):
                A, _ = design_matrix(uniform_in_polygon(V, N, seed=seed), deg)
                sv = np.linalg.svd(A, compute_uv=False)
                row.append(sv[0] / sv[-1])
            print(f"  {N:>7}" + "".join(f"{r:13.2e}" for r in row) + f"    {deg}")
            if N == 16384:
                assert max(row) / min(row) < 1.05, row      # seed spread < 5%

    # What the ratio means: equal-size coefficient changes move psi by amounts
    # differing by that factor.  This is the false floor, seen from the inside.
    A, _ = design_matrix(S, 10)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    rms = lambda a: np.linalg.norm(A @ a) / np.sqrt(len(S))
    print(f"\ndegree 10, unit coefficient vector along the weakest singular "
          f"direction: rms(psi) = {rms(Vt[-1]):.2e}")
    print(f"                                     strongest direction: "
          f"rms(psi) = {rms(Vt[0]):.2e}")
    print(f"  ratio {rms(Vt[0]) / rms(Vt[-1]):.2e} = sigma_max/sigma_min, as it must be.")


if __name__ == "__main__":
    main()
