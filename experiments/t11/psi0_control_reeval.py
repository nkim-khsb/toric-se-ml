"""Round-2 review, item 9: (a) what statistic the T^{1,1} psi = 0 control numbers
are (the draft quotes 1.8e-13 cone form / 1.0e-13 transverse form without saying
whether they are rms or mean-square departures from constancy); (b) positivity of
the slice Hessian for the metrics actually used, with the protocol stated.

(a) is re-evaluated, not recovered from logs: the control is analytic (psi = 0),
so the grader can be fixed and both statistics printed, for both forms and for the
Lambda = 4 anti-test.  Whichever statistic reproduces the draft's digits is the one
the draft meant; if neither does, the draft's digits are replaced by these.

(b) min eigenvalue of Hess_s G_P over (i) the training-type sample (eps = 2e-3
margin) and (ii) the deterministic quadrature nodes (ngl = 48, which approach the
edges to ~1e-3 of the polygon size), for the dP3 deg-14 and deg-18 potentials and
for the dP2 SE potential at b* (deg 14, the one section 6 uses).

Usage: PYTHONPATH=. python experiments/t11/psi0_control_reeval.py
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

from sugrasol import t11                                             # noqa: E402
from sugrasol.losses import ke_residual                              # noqa: E402
from sugrasol.sampling import sample_polytope                        # noqa: E402
from sugrasol.toric import guillemin_potential                       # noqa: E402
from sugrasol.cone import conifold, B_CONIFOLD                       # noqa: E402
from sugrasol.ypq import slice_chart, sample_slice, residual_on_slice  # noqa: E402
from sugrasol.artifacts import load_psi_dp3                          # noqa: E402
from sugrasol.laplacian import slice_potential, polygon_quadrature   # noqa: E402


def stats(r, tag):
    r = np.asarray(r)
    dev = r - r.mean()
    print(f"  {tag:34s} rms {np.sqrt((dev**2).mean()):.2e}   mean-square {(dev**2).mean():.2e}   max {np.abs(dev).max():.2e}")


def part_a():
    print("(a) T^{1,1} psi = 0 control, departure of the residual from its mean")
    # cone form, our slice machinery
    ch = slice_chart(conifold(), B_CONIFOLD)
    for n, eps in ((1024, 1e-3), (4096, 2e-3)):
        ss = sample_slice(jax.random.PRNGKey(1), ch, n, eps=eps)
        r = jax.vmap(lambda s: residual_on_slice(ch, lambda s: 0.0, s))(ss)
        stats(r, f"cone form, n={n}, eps={eps:g}")
    # transverse form, the t11 module (P^1 x P^1 base, Lambda = 6)
    p = t11.transverse_polytope()
    prob = t11.transverse_problem()
    LO, HI = -t11.S * jnp.ones(2), t11.S * jnp.ones(2)
    G = lambda x: guillemin_potential(p, x)
    for n, eps in ((1024, 1e-4), (1024, 1e-3)):
        xs = sample_polytope(jax.random.PRNGKey(1), p, LO, HI, n, eps=eps)
        r = jax.vmap(lambda x: ke_residual(prob, G, x))(xs)
        stats(r, f"transverse form, n={n}, eps={eps:g}")
        bad = prob._replace(lam=4.0)
        rb = jax.vmap(lambda x: ke_residual(bad, G, x))(xs)
        stats(rb, f"  anti-test Lambda=4, n={n}")


def min_eig(u, pts):
    ev = jax.vmap(lambda s: jnp.linalg.eigvalsh(jax.hessian(u)(s)))(jnp.asarray(pts))
    return float(ev.min()), float(ev.max())


def part_b():
    print("\n(b) positivity of the slice Hessian Hess_s G_P")
    for tag in ("deg14", "deg18"):
        psi, ch, dat = load_psi_dp3(ROOT / f"experiments/dp3smooth/dp3_G_{tag}.npz")
        u = slice_potential(ch, psi)
        ss = sample_slice(jax.random.PRNGKey(1), ch, 2048, eps=2e-3)
        nodes, _ = polygon_quadrature(ch.verts_s, ngl=48)
        print(f"  dP3 {tag}: min eig over 2048 samples (eps 2e-3) {min_eig(u, ss)[0]:.4f};"
              f" over {len(nodes)} quadrature nodes (ngl 48) {min_eig(u, nodes)[0]:.4f}")
    from harmonic_forms_dp2 import se_potential
    ch, u, held = se_potential()
    ss = sample_slice(jax.random.PRNGKey(1), ch, 3000, eps=2e-3)
    nodes, _ = polygon_quadrature(ch.verts_s, ngl=48)
    print(f"  dP2 SE deg14 (held-out {held:.2e}): min eig over 3000 samples (eps 2e-3) {min_eig(u, ss)[0]:.4f};"
          f" over {len(nodes)} quadrature nodes (ngl 48) {min_eig(u, nodes)[0]:.4f}")
    # how close do the nodes get to the boundary, in facet distance
    V = np.asarray(ch.verts_s)
    def facet_dist(pts):
        d = np.full(len(pts), np.inf)
        for a in range(len(V)):
            P0, P1 = V[a - 1], V[a]; t = P1 - P0; nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            d = np.minimum(d, np.abs((np.asarray(pts) - P0) @ nrm))
        return d
    print(f"  nearest approach to an edge: samples {facet_dist(ss).min():.1e}, nodes {facet_dist(nodes).min():.1e}"
          f"  (polygon size ~{np.linalg.norm(V, axis=1).max():.2f})")


if __name__ == "__main__":
    part_a()
    part_b()
