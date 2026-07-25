"""Stress-test the ortho-poly pipeline at extreme deformation lambda=3q/2p.

Goal is NOT completeness (Y^{p,q} is closed-form-known) but failure hunting:
does the method break where the slice polygon is most distorted / psi largest?
Three cases: (4,3),(5,4) [extreme lambda, irregular] and (7,3) [quasi-regular,
rational Reeb -- a different corner]. Each graded coordinate-freely by the GMSW
curvature-invariant collapse AND the independent 4th-order Abreu S=12 check.

Usage: PYTHONPATH=. python experiments/ypq/sweep_extreme_lambda.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sugrasol.cone import b_ypq, ypq
from sugrasol.curvature import invariants, toric_metric
from sugrasol.gmsw import cone6, roots_and_a
from sugrasol.ypq import (
    cone_potential, gauge_fixed, loss_fn, ortho_psi, residual_on_slice,
    sample_slice, slice_chart, t_of_s, whiten_poly,
)

jax.config.update("jax_enable_x64", True)

CASES = [(4, 3), (5, 4), (7, 3)]
DEG_LADDER = [12, 14, 16]
LOSS_OK = 1e-11
NSAMP = 4096


def abreu_S(chart, psi, s):
    def u(ss):
        t = t_of_s(chart, ss)
        lt = chart.cone.ells(t)
        return 0.5 * jnp.sum(lt * jnp.log(lt)) + psi(ss)
    Hinv = lambda ss: jnp.linalg.inv(jax.hessian(u)(ss))
    T = jax.jacfwd(jax.jacfwd(Hinv))(s)
    return -jnp.einsum("jkjk->", T)


def train_case(P, Q):
    chart = slice_chart(ypq(P, Q), b_ypq(P, Q))
    ss = sample_slice(jax.random.PRNGKey(1), chart, NSAMP, eps=2e-3)
    ss_test = sample_slice(jax.random.PRNGKey(7), chart, NSAMP, eps=2e-3)

    best = None
    for DEG in DEG_LADDER:
        powers, W = whiten_poly(DEG, ss)
        nc = len(powers)

        def Lf(v, batch):
            psi = gauge_fixed(ortho_psi(v[:nc], powers, W), chart.anchors)
            return loss_fn(chart, psi, v[nc], batch)

        vg = jax.jit(jax.value_and_grad(lambda v: Lf(v, ss)))
        L_test = jax.jit(lambda v: Lf(v, ss_test))

        r0 = jax.vmap(lambda s: residual_on_slice(chart, lambda s: 0.0, s))(ss)
        v0 = np.zeros(nc + 1)
        v0[nc] = -float(jnp.mean(r0))
        res = minimize(lambda v: tuple(np.asarray(x) if i else float(x)
                                       for i, x in enumerate(vg(jnp.asarray(v)))),
                       v0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=30000, ftol=1e-18, gtol=1e-16))
        v = jnp.asarray(res.x)
        l_tr, l_te = float(res.fun), float(L_test(v))
        best = dict(chart=chart, ss_test=ss_test, powers=powers, W=W, nc=nc,
                    v=v, deg=DEG, l_tr=l_tr, l_te=l_te, iters=res.nit)
        print(f"  deg {DEG:2d} ({nc} coeffs): loss {l_tr:.2e}  held-out {l_te:.2e}"
              f"  ({res.nit} iters)")
        if l_tr < LOSS_OK:
            break
    return best


def grade(P, Q, r):
    chart, v = r["chart"], r["v"]
    psi = gauge_fixed(ortho_psi(v[:r["nc"]], r["powers"], r["W"]), chart.anchors)
    sup_psi = float(jnp.max(jnp.abs(jax.vmap(psi)(r["ss_test"]))))

    # --- GMSW coordinate-free grading
    ss_eval = sample_slice(jax.random.PRNGKey(11), chart, 400, eps=1e-2)
    g6 = toric_metric(cone_potential(chart, psi), 3)

    def inv_at(s):
        y = t_of_s(chart, s) / 2.0
        return jnp.stack(invariants(g6, jnp.concatenate([y, jnp.zeros(3)])))
    inv_after = jax.vmap(inv_at)(ss_eval)

    y1, y2, a = roots_and_a(P, Q)
    g6_gmsw = cone6(float(a))
    ys = jnp.linspace(float(y1) + 1e-4, float(y2) - 1e-4, 200)
    inv_gmsw = jax.vmap(lambda yv: jnp.stack(
        invariants(g6_gmsw, jnp.array([1.0, yv, 0.9, 0.3, 0.7, 0.2]))))(ys)
    order = jnp.argsort(inv_gmsw[:, 0])
    xg, yg = inv_gmsw[order, 0], inv_gmsw[order, 1]
    phi2_ref = jnp.interp(inv_after[:, 0], xg, yg)
    rel = jnp.abs(inv_after[:, 1] - phi2_ref) / jnp.abs(phi2_ref)
    inside = (inv_after[:, 0] > float(jnp.min(xg))) & (inv_after[:, 0] < float(jnp.max(xg)))
    rel_in = rel[inside]
    gmsw_med, gmsw_max = float(jnp.median(rel_in)), float(jnp.max(rel_in))

    # --- Abreu S=12
    S = jax.vmap(lambda s: abreu_S(chart, psi, s))(ss_eval)
    abreu_dev = float(jnp.max(jnp.abs(S - 12.0)) / 12.0)

    return dict(sup_psi=sup_psi, gmsw_med=gmsw_med, gmsw_max=gmsw_max,
                abreu_dev=abreu_dev, inside=int(jnp.sum(inside)))


rows = []
for (P, Q) in CASES:
    lam = 3.0 * Q / (2.0 * P)
    b = b_ypq(P, Q)
    disc = 4 * P**2 - 3 * Q**2
    reg = "quasi-reg" if int(round(disc**0.5))**2 == disc else "irregular"
    print(f"\n== Y^{{{P},{Q}}}  lambda={lam:.3f}  4p^2-3q^2={disc} ({reg})  "
          f"b=({float(b[0]):.0f},{float(b[1]):.4f},{float(b[2]):.4f}) ==")
    r = train_case(P, Q)
    g = grade(P, Q, r)
    print(f"  sup|psi|={g['sup_psi']:.3e}  GMSW med/max {g['gmsw_med']:.2e}/"
          f"{g['gmsw_max']:.2e}  AbreuS dev {g['abreu_dev']:.2e}")
    rows.append((P, Q, lam, reg, r["deg"], r["l_tr"], r["l_te"], g["sup_psi"],
                 g["gmsw_med"], g["gmsw_max"], g["abreu_dev"]))

print("\n\n================ SUMMARY: extreme-lambda stress test ================")
hdr = ("p,q", "lam", "class", "deg", "loss", "held", "sup|psi|",
       "GMSWmed", "GMSWmax", "AbreuDev")
print("{:>5} {:>6} {:>10} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(*hdr))
for (P, Q, lam, reg, deg, ltr, lte, sp, gm, gx, ad) in rows:
    print(f"{P},{Q:>3} {lam:6.3f} {reg:>10} {deg:>4} {ltr:9.1e} {lte:9.1e} "
          f"{sp:9.1e} {gm:9.1e} {gx:9.1e} {ad:9.1e}")
