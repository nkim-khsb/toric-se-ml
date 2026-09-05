# toric-se-ml

Code accompanying **"Learning Toric Sasaki--Einstein metrics"**, by Nakwoo
Kim, Sejin Kim and Hoseob Shin (in preparation).

This repository contains exactly the code the paper's results depend on: a
pipeline that learns the symplectic potential of a toric Ricci-flat Kähler
cone from the Monge--Ampère equation, validated against closed forms
($T^{1,1}$, $Y^{p,q}$), an independent numerical reference (the DHHKW
Kähler--Einstein metric on the cone over the third del Pezzo surface), and a
theorem-grade negative control (the obstructed Kähler--Einstein equation on
the cone over the second del Pezzo surface).

## Layout

- `sugrasol/` -- the core library: toric cone data (`cone.py`), the symplectic
  ansatz and boundary conditions (`toric.py`), sampling (`sampling.py`), the
  Monge--Ampère loss (`losses.py`), the MLP ansatz (`nets.py`), $T^{1,1}$ and
  $Y^{p,q}$ closed forms (`t11.py`, `ypq.py`, `gmsw.py`), curvature invariants
  (`curvature.py`), the Laplacian / Abreu-scalar graders (`laplacian.py`), and
  the readers for the persisted $\mathrm{dP}_3$ metrics (`artifacts.py`).
- `experiments/t11/`, `experiments/ypq/`, `experiments/dp3/`, `experiments/dp2/`
  -- the four rungs of the validation ladder, matching the paper's sections.
- `experiments/dp3smooth/dp3_G_deg{14,18}.npz` -- the two precomputed
  $\mathrm{dP}_3$ symplectic potentials. **These are the metrics quoted in the
  paper**: every $\mathrm{dP}_3$ number in its Section 4 is recomputed from them
  rather than by retraining, so that the reported values cannot drift with a
  default tolerance. `experiments/dp3/refit_full_rank.py` regenerates them from
  scratch (a few minutes on a CPU) and should reproduce the held-out residuals
  $9.1\times10^{-14}$ (degree 14, fourteen parameters) and $1.0\times10^{-16}$
  (degree 18, twenty-one parameters).
- `tests/` -- closed-form and consistency tests for the modules above,
  including regression tests that pin the paper's headline numbers
  (`test_invariant_basis.py`, `test_dp3_dhhkw_D.py`, `test_curvature_anchors.py`).
- `paper/` -- the paper source (`paper1-methods.tex`, `refs.bib`) and a
  compiled PDF.

## Setup

```
pip install -r requirements.txt
PYTHONPATH=. pytest tests/
```

## Reproducing the main results

- $T^{1,1}$ smoke tests: `PYTHONPATH=. pytest tests/test_t11_smoke.py`
- $Y^{p,q}$ closed-form comparison and the conditioning fix:
  `experiments/ypq/train_y21.py`, `train_ypq_precise.py`,
  `sweep_extreme_lambda.py`; the held-out losses at the raw-monomial plateau
  (the paper's non-existence diagnostic depends on them) are evaluated on the
  persisted stalled minimizers by `raw_floor_heldout.py`
- $\mathrm{dP}_3$ positive control (DHHKW comparison):
  `experiments/dp3/smoke_dp3.py`, `lambda_compare.py` (eigenvalues and the
  three-way convergence ladder), `dhhkw_D_compare.py` (DHHKW's own pointwise
  Einstein-condition measure $D$ evaluated on our metric),
  `invariant_rank_check.py` (invariant-basis rank against Molien's count),
  `direct_compare.py`, `figures.py`; neural-ansatz cross-check: `nn_dp3.py`,
  `nn_dp3_nosym.py`
- $\mathrm{dP}_2$ negative control: `experiments/dp2/negative_control.py`,
  `negative_control_nn.py`, `fig_negative.py`

Each `experiments/*/note-*.tex` is a standalone derivation note for that rung.

## Scope

This repository is deliberately scoped to the toric ($T^3$-invariant) part of
a larger project. Non-toric follow-on work (the deformed-conifold /
Klebanov--Strassler rung, and the physics companion paper on the $\mathrm{dP}_2$
Sasaki--Einstein metric and its Kaluza--Klein spectrum) lives in separate,
not-yet-released repositories tied to their own papers.

## Citation

If you use this code, please cite the paper (BibTeX entry to follow at
publication; see `paper/refs.bib` for the citation keys used internally).
