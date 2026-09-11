"""Full torus-invariant Laplacian spectrum on dP3, labelled by D6 irreps
(review item 3, 2026-09-11).

The paper quotes lambda_1 = 6.3228 (DHHKW units, S = 4) as "the" first
eigenvalue.  DHHKW sec. 6.1 restrict to eigenfunctions invariant under D6, and
their (3.40) shows the symplectic coordinates themselves are eigenfunctions of
-Laplacian with eigenvalue 2 Lambda = 2 in their units -- torus-invariant, not
D6-invariant, and BELOW 6.32.  So 6.3228 is the lowest eigenvalue of the
TRIVIAL D6 sector, not of the full torus-invariant spectrum.  This script
computes the full torus-invariant spectrum on the persisted deg-18 metric in a
general (non-symmetrized) polynomial basis, labels each level by its D6
irreducible representation from the character of the induced representation on
the eigenspace, and prints three normalizations side by side:

    lambda_DHHKW = lambda_slice * 4/S = lambda_slice / 3   (their Ric = g),
    lambda_link  = 2 lambda_slice = 6 lambda_DHHKW          (r = 1 link, S_T = 24),

since our slice l_b = 1 carries twice the transverse metric of the r = 1 link.

PRE-REGISTERED:
  (S1) the lowest nonzero level is doubly degenerate at lambda_DHHKW = 2.000
       (the moment maps, irrep E1), to 1e-6.
  (S2) the lowest TRIVIAL (A1) level is 6.3228 and the second 17.094, matching
       the paper's D6-sector numbers to the digits quoted.
  (S3) characters of every level are integers to 1e-3.

Usage: PYTHONPATH=. python experiments/dp3/laplacian_full_sector.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

jax.config.update("jax_enable_x64", True)

from sugrasol.artifacts import load_psi_dp3                          # noqa: E402
from sugrasol.laplacian import polygon_quadrature, slice_potential   # noqa: E402
from sugrasol.ypq import dihedral_matrices, sample_slice             # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "experiments/dp3smooth/dp3_G_deg18.npz"
NGL = 24
BDEG = 16
S_OURS = 12.0

# D6 character table.  Classes: e | r^{+-1} | r^{+-2} | r^3 | s_v (3 refl. through
# vertices) | s_e (3 refl. through edge midpoints).
CHARS = {
    "A1": (1, 1, 1, 1, 1, 1),
    "A2": (1, 1, 1, 1, -1, -1),
    "B1": (1, -1, 1, -1, 1, -1),
    "B2": (1, -1, 1, -1, -1, 1),
    "E1": (2, 1, -1, -2, 0, 0),
    "E2": (2, -1, -1, 2, 0, 0),
}


def classify(g, verts):
    det = np.linalg.det(g)
    tr = np.trace(g)
    if det > 0:
        return {2: 0, 1: 1, -1: 2, -2: 3}[int(round(tr))]
    # reflection: does it fix a vertex?
    fixes = any(np.allclose(g @ v, v, atol=1e-8) for v in verts)
    return 4 if fixes else 5


def main():
    psi, ch, dat = load_psi_dp3(NPZ)
    u = slice_potential(ch, psi)
    verts = np.asarray(ch.verts_s)
    group = np.asarray(dihedral_matrices(ch.verts_s))
    classes = [classify(g, verts) for g in group]
    print(f"deg-18 metric ({dat['W'].shape[1]} params), basis: all monomials to degree {BDEG}, ngl = {NGL}")
    print(f"D6 class sizes found: {[classes.count(c) for c in range(6)]}  (expect [1,2,2,1,3,3])")

    # general polynomial basis, SVD-orthonormalized on a cloud for conditioning
    pw = [(i, t - i) for t in range(0, BDEG + 1) for i in range(t + 1)]
    raw = lambda s: jnp.stack([s[0] ** i * s[1] ** j for i, j in pw])
    cloud = sample_slice(jax.random.PRNGKey(5), ch, 40000, eps=1e-4)
    Araw = jax.vmap(raw)(cloud)
    _, sv, Vt = jnp.linalg.svd(Araw, full_matrices=False)
    r = int(jnp.sum(sv > 1e-12 * sv[0]))
    Wb = Vt[:r].T / sv[:r]
    basis = lambda s: raw(s) @ Wb
    print(f"basis dimension {r} of {len(pw)}")

    nodes, wts = polygon_quadrature(ch.verts_s, ngl=NGL)
    uinv = lambda s: jnp.linalg.inv(jax.hessian(u)(s))
    jac = jax.jacfwd(basis)
    A = np.asarray(jnp.tensordot(wts, jax.vmap(lambda s: jac(s) @ uinv(s) @ jac(s).T)(nodes), axes=(0, 0)))
    B = np.asarray(jnp.tensordot(wts, jax.vmap(lambda s: jnp.outer(basis(s), basis(s)))(nodes), axes=(0, 0)))
    A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
    w, vec = eigh(A, B)                   # B-orthonormal eigenvectors = L^2-orthonormal
    lam_dh = w * 4.0 / S_OURS

    # representation matrices on each degenerate cluster
    fvals = np.asarray(jax.vmap(basis)(nodes)) @ vec          # (n, m) eigenfunctions at nodes
    gvals = [np.asarray(jax.vmap(basis)(jnp.asarray(nodes @ g.T))) @ vec for g in group]  # f(g s)
    wts_np = np.asarray(wts)

    print(f"\n{'level':>5} {'lam_DHHKW':>11} {'lam_slice':>11} {'lam_link':>11} {'deg':>4} "
          f"{'irrep':>6}  character (e, r1, r2, r3, s_v, s_e)")
    fails = []
    i = 1
    nshow = 0
    first_A1 = []
    while nshow < 14 and i < len(w):
        j = i
        while j + 1 < len(w) and abs(lam_dh[j + 1] - lam_dh[i]) < 1e-5 * max(1.0, lam_dh[i]):
            j += 1
        idx = list(range(i, j + 1))
        chi = np.zeros(6)
        cnt = np.zeros(6)
        for g, c in zip(gvals, classes):
            M = fvals[:, idx].T @ (wts_np[:, None] * g[:, idx])
            chi[c] += np.trace(M)
            cnt[c] += 1
        chi = chi / np.maximum(cnt, 1)
        # decompose into irreps (multiplicities)
        names = []
        for name, ch_tab in CHARS.items():
            tab = np.array(ch_tab, float)
            sizes = np.array([1, 2, 2, 1, 3, 3], float)
            mult = np.sum(sizes * chi * tab) / 12.0
            if abs(mult - round(mult)) > 1e-3:
                fails.append(f"(S3) level {lam_dh[i]:.4f}: non-integer multiplicity {mult:.4f} for {name}")
            if round(mult) > 0:
                names.append(name if round(mult) == 1 else f"{int(round(mult))}{name}")
        label = "+".join(names)
        if "A1" in names:
            first_A1.append(lam_dh[i])
        print(f"{nshow + 1:>5} {lam_dh[i]:11.5f} {w[i]:11.5f} {6 * lam_dh[i]:11.4f} {len(idx):>4} "
              f"{label:>6}  " + " ".join(f"{c:6.3f}" for c in chi))
        i = j + 1
        nshow += 1

    # gates
    lvl1 = lam_dh[1:3]
    if not (abs(lvl1[0] - 2.0) < 1e-6 and abs(lvl1[1] - 2.0) < 1e-6):
        fails.append(f"(S1) lowest nonzero level not 2.000 x2: {lvl1}")
    if len(first_A1) < 2 or abs(first_A1[0] - 6.3228) > 5e-4 or abs(first_A1[1] - 17.094) > 5e-3:
        fails.append(f"(S2) trivial-sector levels {first_A1[:2]} vs paper 6.3228, 17.094")
    print(f"\nlowest trivial (A1) levels, DHHKW units: {[f'{v:.5f}' for v in first_A1[:3]]}")
    print(f"moment-map level: lambda_DHHKW = {lvl1[0]:.8f}, {lvl1[1]:.8f}  (exact 2 = 2 Lambda);"
          f" link value {6 * lvl1[0]:.6f} (exact 12)")
    print(f"\nFAILURES: {fails if fails else 'none'}")


if __name__ == "__main__":
    main()
