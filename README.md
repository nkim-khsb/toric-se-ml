# toric-se-ml

Code accompanying **"Machine-learned toric Sasaki--Einstein metrics: a
validated pipeline with two-sided sensitivity"** (N. Kim, submitted to
*Machine Learning: Science and Technology*).

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
  (`curvature.py`), and the Laplacian / Abreu-scalar graders (`laplacian.py`).
- `experiments/t11/`, `experiments/ypq/`, `experiments/dp3/`, `experiments/dp2/`
  -- the four rungs of the validation ladder, matching the paper's sections.
- `experiments/dp3smooth/dp3_G_deg18.npz` -- a single precomputed checkpoint
  (the degree-18 $\mathrm{dP}_3$ symplectic potential) used only as the
  ground-truth reference for the neural-ansatz de-risking run in the paper's
  Discussion section; it is not part of any non-toric follow-up work.
- `tests/` -- closed-form and consistency tests for the modules above.
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
  `sweep_extreme_lambda.py`
- $\mathrm{dP}_3$ positive control (DHHKW comparison):
  `experiments/dp3/smoke_dp3.py`, `lambda_compare.py`, `direct_compare.py`,
  `figures.py`; neural-ansatz cross-check: `nn_dp3.py`, `nn_dp3_nosym.py`
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
