# toric-se-ml

Code accompanying **"Numerical Sasaki--Einstein metrics and harmonic forms on
del Pezzo links"**, by Nakwoo Kim, Sejin Kim and Hoseob Shin (2026).

The paper numerically constructs the Sasaki--Einstein metric on the link of the
cone over the second del Pezzo surface, at its irregular volume-minimizing Reeb
vector, together with the two primitive harmonic basic $(1,1)$-forms on it from
which primitive imaginary self-dual $(2,1)$-forms on the cone are built. This
repository holds the code, the random seeds and the stored minimizers behind
every number in the paper.

The method learns the symplectic potential of a toric Ricci-flat Kähler cone
from the Monge--Ampère equation. It is validated against closed forms
($T^{1,1}$, $Y^{p,q}$), against an independent numerical reference (the DHHKW
Kähler--Einstein metric on the cone over the third del Pezzo surface), and
against a non-existence theorem (the obstructed Kähler--Einstein equation at
the regular Reeb vector of the second del Pezzo cone).

## Layout

- `sugrasol/` -- the core library: toric cone data (`cone.py`), the symplectic
  ansatz and boundary conditions (`toric.py`), sampling (`sampling.py`), the
  Monge--Ampère loss (`losses.py`), the MLP ansatz (`nets.py`), $T^{1,1}$ and
  $Y^{p,q}$ closed forms (`t11.py`, `ypq.py`, `gmsw.py`), curvature invariants
  (`curvature.py`), the Laplacian / Abreu-scalar graders (`laplacian.py`),
  differential forms and the Hodge star (`forms.py`), and the readers for the
  persisted $\mathrm{dP}_3$ metrics (`artifacts.py`).
- `experiments/t11/`, `experiments/ypq/`, `experiments/dp3/`, `experiments/dp2/`
  -- the metric calculations, in the order the paper presents them.
- `experiments/dp3kt/` -- the harmonic $(1,1)$-forms of the paper's section 6:
  the stream-potential reduction, the admissible (Wachspress) trial space, the
  self-dual energy, the divisor-period basis and the figure.
- `experiments/dp3smooth/dp3_G_deg{14,18}.npz` -- the two precomputed
  $\mathrm{dP}_3$ symplectic potentials. **These are the metrics quoted in the
  paper**: every $\mathrm{dP}_3$ number is recomputed from them
  rather than by retraining, so that the reported values cannot drift with a
  default tolerance. `experiments/dp3/refit_full_rank.py` regenerates them from
  scratch (a few minutes on a CPU) and should reproduce the held-out residuals
  $9.1\times10^{-14}$ (degree 14, fourteen parameters) and $1.0\times10^{-16}$
  (degree 18, twenty-one parameters).
- `tests/` -- closed-form and consistency tests for the modules above,
  including regression tests that pin the paper's headline numbers
  (`test_invariant_basis.py`, `test_dp3_dhhkw_D.py`, `test_curvature_anchors.py`,
  `test_review_20260911.py`).

The $\mathrm{dP}_2$ Sasaki--Einstein potential is not shipped as a file: it is
refitted in about a minute by `experiments/dp2/harmonic_forms_dp2.py`, which
gates on reproducing the paper's held-out residual $6.0\times10^{-14}$ at
degree 14 before reporting anything downstream.

## Setup

```
pip install -r requirements.txt
PYTHONPATH=. pytest tests/
```

`requirements.txt` gives loose constraints. The runs, timings and stored
minimizers in the paper come from one build, Python 3.11.9 with JAX 0.10.2
and SciPy 1.17.1 on the CPU of an Apple M1 Max; its exact versions are in
`requirements-lock.txt` (`pip install -r requirements-lock.txt`). Runs that
stop on the optimizer's convergence test reproduce across builds to
$0.99$--$1.00$ in the held-out residual; runs that exhaust an evaluation cap
do not, and a different JAX version also draws different sample points from
the same `PRNGKey`.

## Reproducing the main results

- $T^{1,1}$ calibration: `PYTHONPATH=. pytest tests/test_t11_smoke.py`; the
  $\psi=0$ control statistics quoted in the text come from
  `experiments/t11/psi0_control_reeval.py`, which also reports the slice-Hessian
  positivity checks of appendix B.
- $Y^{p,q}$ closed-form comparison and the conditioning fix:
  `experiments/ypq/train_y21.py`, `train_ypq_precise.py`,
  `sweep_extreme_lambda.py`, `basis_conditioning.py`; the held-out losses at the
  raw-monomial plateau are evaluated on the persisted stalled minimizers by
  `raw_floor_heldout.py`, and the Abreu columns of the sweep table by
  `abreu_grid_sweep_rescore.py`.
- $\mathrm{dP}_3$ positive control (DHHKW comparison):
  `experiments/dp3/smoke_dp3.py`, `lambda_compare.py` (the $D_6$-invariant
  eigenvalues and the three-way convergence ladder),
  `laplacian_full_sector.py` (the full torus-invariant spectrum with its $D_6$
  labels), `dhhkw_D_compare.py` (DHHKW's pointwise Einstein-condition measure
  evaluated on our metric), `dhhkw_theta_compare.py` and
  `dhhkw_theta_sdenergy.py`, `dhhkw_theta_sdenergy_ladder.py` and
  `dhhkw_theta_deterministic.py` (their published $(1,1)$-form and refits of
  their ansatz graded on our metric, with deterministic quadrature),
  `dhhkw_eigvec_class.py` (the classes and $D_6$ characters of the low-energy
  directions of their span),
  `invariant_rank_check.py`, `direct_compare.py`, `review_pins_20260911.py`
  (the volume-Hessian and potential-normalization identities), `figures.py`;
  neural-ansatz cross-check: `nn_dp3.py`, `nn_dp3_nosym.py`,
  `nn_budget_probe.py`.
- $\mathrm{dP}_2$: `experiments/dp2/negative_control.py`,
  `negative_control_nn.py`, `width_ladder_released.py` (the released-budget
  width ladder), `floor_vs_volume_excess.py` (the floor against the volume
  excess), `abreu_grid_bstar.py`, `fig_negative.py`.
- Harmonic $(1,1)$-forms: `experiments/dp3kt/r2b_stream_wachspress.py` is the
  production calculation in the admissible space, on all four polygons;
  `r2b_stream_admissible.py` is the diagnosis of the earlier polynomial space
  and of why exactly-admissible polynomials carry no vertex data;
  `r2b_stream_wachspress_checks.py` verifies the structural identities
  $S=\tfrac12(M+T)$ and the harmonic-distance relation and prints the
  quadrature table; `r2b_half_check.py` measures the departure of the
  non-kernel energies from $\tfrac12$; `r2b_arith_check.py` re-evaluates the
  reported energies directly from the fixed stream functions, without the
  orthonormal basis change; `fig_period_basis_adm.py` fixes the
  hexagon basis by divisor periods and the pentagon basis by parity, and draws
  the figure.

## Scope

This repository is deliberately scoped to the toric ($T^3$-invariant) part of a
larger project, and within that to the results of this paper. Follow-on work
(the flux and warp-factor construction on these cones, the non-toric
deformed-conifold rung, and the physics companion paper on the
$\mathrm{dP}_2$ Kaluza--Klein spectrum) lives in separate repositories tied to
their own papers. A few modules here are shared with that work and carry
docstrings referring to it.

## Citation

If you use this code, please cite the paper. The BibTeX entry will be added
when the preprint appears. The code is released under the MIT License (see
`LICENSE`).
