# Sharpness Aware Minimization Variants

Sharpness-Aware Minimization (SAM) computes the gradient at an adversarially
perturbed point w̃ = w + ε so that training favors flat minima — but doubling
the gradient cost is hard to justify. Momentum-SAM (MSAM) gets the perturbation
for free by reusing the momentum buffer: w̃ = w − ρ·v/‖v‖ (note the sign — the
perturbation points *along* the update direction; the loss still rises there
because momentum overshoots local minima). Its results are okay but leave more
to be desired.

This repo explores a generalization suggested by Gallabytes: build the
perturbation from a **different optimizer's update direction** than the one
used for descent. Concretely, `HybridSAM` lets you pick ascent and descent
independently from {momentum EMA, Adam direction, Muon-orthogonalized
momentum}, e.g. descend with Muon while perturbing with the Adam direction, or
vice versa. All directions are built from shared buffers updated with the
(perturbed-point) gradient, so there is still only one gradient per step.

## Layout

- `hybrid_sam.py` — the optimizer. Weights live at w̃ between steps; `step()`
  removes the cached perturbation exactly, descends from clean w, re-perturbs.
- `muon.py` — Muon baseline (Adam fallback for non-2D / oversized params).
- `train_gpt.py` / `transformer.py` — GPT pretraining harness (OpenWebText,
  ~250M-param GPT-2-medium-shaped model), config via JSON overrides.
- `launch_llm.sh` — Slurm sweep launcher.
- `tests/test_hybrid_sam.py` — correctness suite (`python tests/test_hybrid_sam.py`).
- `EXPERIMENT_JOURNAL.md` — running log of decisions, findings, and planned runs.

## Conventions worth knowing

- **Sign**: `rho > 0` perturbs *forward along the descent direction* — a
  Nesterov-style lookahead (verified: cos(ε, descent step) ≈ +0.9). MSAM calls
  this "ascent" because the loss at that point is *higher*, not because the
  direction is uphill: past ~1–2 steps you overshoot the line-minimum.
  `rho < 0` perturbs along the gradient/uphill direction — classic SAM-style
  ascent. Sweep both.
- **Normalization**: `perturbation_norm="balanced"` (default) gives each param
  ‖ε_p‖ = ρ·√(numel_p/total) with unit per-param directions; total norm = ρ.
  Raw `"global"` (MSAM's choice) is only safe when every param uses the same
  direction family — mixing elementwise Adam directions with Muon-orthogonalized
  ones lets embeddings absorb >99% of the budget.
- **Eval/save**: params are perturbed between steps. Use
  `with optimizer.unperturbed():` around evaluation and
  `optimizer.remove_perturbation()` before the final save (train_gpt.py does
  both, and also logs `eval_sam_gap` = loss(w̃) − loss(w) as a sharpness probe).

## References

- SAM: https://arxiv.org/abs/2010.01412
- Momentum-SAM: https://arxiv.org/abs/2401.12033
- Muon: https://github.com/KellerJordan/Muon
