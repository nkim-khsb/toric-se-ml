"""Pins experiments/dp3/bhhpw_pointwise_compare.py: our dP3 artifacts against
the certified BHHPW potential (arXiv:2609.32655, companion-repo coefficients).
The four convention gates must pass exactly as in the script, and the measured
absolute errors are pinned with a 3x margin (quadrature ngl=12 here, 24 in the
script -- the error is a smooth function, the numbers agree to the digits
quoted)."""
import jax
import jax.numpy as jnp
import numpy as np

from sugrasol.artifacts import load_psi_dp3
from sugrasol.laplacian import polygon_quadrature, slice_potential
from sugrasol.ypq import dihedral_matrices, residual_on_slice, sample_slice

from experiments.dp3 import bhhpw_pointwise_compare as B


def test_bridge_and_convention_gates():
    assert B.G0_RES < 1e-12
    assert abs(abs(np.linalg.det(B.PHI_M)) - 9.0) < 1e-10
    ss = sample_slice(jax.random.PRNGKey(5), B.chart, 400, eps=2e-3)
    r = jax.jit(jax.vmap(lambda s: residual_on_slice(B.chart, B.psi_th, s)))(ss)
    assert float(jnp.std(r)) < 1e-8                     # (G1) their h solves OUR equation
    Hg = jax.jit(jax.vmap(jax.hessian(B.guil_diff)))(ss)
    assert float(jnp.max(jnp.abs(Hg))) < 1e-10          # (G2)
    f = jax.jit(jax.vmap(B.psi_th))
    base = f(ss)
    for g in dihedral_matrices(B.chart.verts_s):
        assert float(jnp.max(jnp.abs(f(ss @ np.asarray(g).T) - base))) < 1e-10   # (G3)


def test_absolute_error_against_certified_truth():
    nodes, w = polygon_quadrature(B.chart.verts_s, ngl=12)
    area = float(jnp.sum(w))
    H_th = jax.jit(jax.vmap(jax.hessian(lambda s: B.u_them(B.phi(s)) / 3.0)))(nodes)
    rms = {}
    for tag, path in B.NPZ.items():
        psi, _, _ = load_psi_dp3(path)
        H_o = jax.jit(jax.vmap(jax.hessian(slice_potential(B.chart, psi))))(nodes)
        e = jax.vmap(B.rel_metric_err)(H_o, H_th)
        rms[tag] = float(jnp.sqrt(jnp.dot(w, e ** 2) / area))
        if tag == "deg-18":
            assert rms[tag] < 3.5e-8 and float(jnp.max(e)) < 3.5e-7   # script: 1.17e-8 / 1.18e-7
        if tag == "deg-14":
            assert rms[tag] < 1.0e-6 and float(jnp.max(e)) < 7.0e-6   # script: 3.37e-7 / 2.34e-6
    assert rms["deg-18"] < rms["deg-14"] / 10                           # the ladder
