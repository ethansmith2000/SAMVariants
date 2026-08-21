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

- **Two modes, selected by the sign of ρ.** Each step:
  perturb w → compute the gradient at w̃ → revert → descend from w.

  | | ρ > 0 — **LOOKAHEAD** | ρ < 0 — **ASCENT** |
  |---|---|---|
  | ε points | *forward* along the descent direction | *uphill* along the gradient direction |
  | verified | cos(ε, descent step) ≈ +0.9 | cos(ε, descent step) ≈ −0.9 |
  | analogy | Nesterov / extragradient | classic SAM |
  | loss at w̃ | higher (you overshoot the line-minimum past ~1–2 steps) | higher (you climbed) |

  MSAM calls its ρ>0 perturbation "ascent" because the *loss* there is higher,
  not because the direction is uphill. We reserve "ascent" for ρ<0 and say
  "lookahead" for ρ>0.

- **`ascent=` / `descent=` name geometries, not directions.** `ascent` is which
  optimizer's direction we perturb *along* (aliased `perturb_with=`); `descent`
  is which optimizer performs the actual update (aliased `update_with=`).
  So `ascent="adam", descent="muon", rho=4` = "look 4 Muon-steps ahead along
  the Adam direction, then take a Muon step".

- **What ρ means**: with `perturbation_scale="relative"` (all sweeps since the
  pilot), ‖ε_p‖ = |ρ| · EMA(‖actual update to p‖) — ρ is *multiples of that
  parameter's own recent update length*. This is not MSAM's convention: MSAM
  normalizes by one **global** Frobenius norm over all parameters, so its ρ is
  an absolute distance in weight space (their 1.7–5.5). `"absolute"` mode is
  the MSAM-like setting.

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
