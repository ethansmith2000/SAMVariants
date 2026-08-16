# SAMVariants — Experiment Journal

Running log of design decisions, findings, and planned experiments. Newest entries at the bottom.

## Background & conventions (reference)

**The idea.** SAM perturbs weights adversarially before computing the gradient, at 2× gradient
cost. MSAM (arXiv:2401.12033) makes it free by perturbing with the momentum buffer. We generalize
(per Gallabytes' suggestion): perturb with *another optimizer's* update direction while descending
with your optimizer of choice — Adam-perturb/Muon-descend, Muon-perturb/Adam-descend, etc.

**Key facts from the MSAM paper** (verified against their code + ar5iv, 2026-08-16):

- Perturbation: ε = **−ρ·v/‖v‖**, v = momentum buffer (gradient accumulation, points uphill),
  norm is **global** (single L2 over all params concatenated). The sign means the perturbation
  points *along* the update direction; loss still increases there because momentum overshoots.
  So MSAM is lookahead-flavored, not classic SAM ascent.
- ρ values are larger than intuition suggests: 2.2–3.0 (WRN/CIFAR), 1.7–3.0 (ResNet/ImageNet),
  3.0–5.5 (ViT/ImageNet).
- **Do not scale ρ with lr**: their Appendix A.2 shows NAG-style ρ∝η coupling *hurts*
  generalization. Constant ρ throughout training.
- **ρ = 0 during warmup** (they do this for ViTs) — motivates our `perturbation_start_step`.
- They evaluate and save at the **unperturbed** weights (Algorithm 1 removes the final
  perturbation explicitly).
- Their "Adam variant" perturbs with the raw momentum EMA — i.e. our `ascent="momentum"` with
  Adam descent is exactly MSAM-on-Adam. `ascent="adam"` (preconditioned direction) and
  `ascent="muon"` are new territory.

**Our conventions** (`hybrid_sam.py`):

- `rho > 0` = MSAM lookahead sign; `rho < 0` = classic SAM ascent. Signed ρ is the ablation.
- `perturbation_norm="balanced"` (default): per-param unit directions scaled √(numel_p/total),
  total norm = ρ, equal per-element RMS everywhere. `"global"` reproduces MSAM but is only safe
  for homogeneous direction families (see 2026-08-16 finding below). `"per_param"`: every tensor
  gets norm ρ.
- Weights live at w̃ between steps. Eval inside `optimizer.unperturbed()`; final save after
  `optimizer.remove_perturbation()`. `eval_sam_gap` = loss(w̃) − loss(w) is logged at each
  validation as a free sharpness probe.

---

## 2026-06 — initial implementations (pre-journal, reconstructed from git)

First versions: `muon_adam_perturb.py` (Muon descent, +Adam-direction perturbation),
`adam_two_momentum_perturb.py` (Adam descent, perturbation from a second faster-β EMA),
`adam_wd_perturb.py` (extra weight-decay shrink as perturbation), then the configurable
`HybridSAM` in `adam_muon_perturb.py`. The `wtf1`–`wtf5` commit streak was debugging the
perturbed-weight bookkeeping. A sweep (`launch_llm.sh`) over ρ ∈ {0.1…1.7} for
adam→muon and muon→muon configs was launched on Slurm (wandb: `sam-variants-llm`);
no checkpoints or journal survive from that period.

## 2026-08-16 — code review: three confounds found, sweep results untrustworthy

Full review of the codebase (Claude), verified against the MSAM reference implementation:

1. **Eval and final save happened at w̃, not w.** Validation ran right after `optimizer.step()`,
   when params hold the perturbed weights; baselines evaluated at their true iterates. Bias
   ≈ ρ·⟨d̂,∇L⟩ + O(ρ²), first-order flattering for SAM runs, growing with ρ — corrupts exactly
   the ρ-trend being measured.
2. **Global norm + muon ascent + adam fallback was pathological.** NS-orthogonalized directions
   have Frobenius norm ≈ √min(m,n) (~32 for our matrices) while elementwise Adam directions on
   the 50257×1024 embedding/lm_head have norm ~7000. Under one global norm, embeddings absorbed
   ≳99% of the perturbation budget: the "muon-muon" runs were effectively *embedding-only Adam
   perturbation*. The transformer matrices received ~3e-4·ρ relative perturbation (nothing).
3. **Sign conventions disagreed across files.** `HybridSAM` used the MSAM sign (−ρd̂);
   `muon_adam_perturb.py` / `adam_two_momentum_perturb.py` perturbed +Adam-direction (classic
   SAM ascent). Also in the older files: perturbation removal used the *new* scheduler lr and
   bias-correction scale, so removal ≠ application → per-step drift (worst during warmup); and
   an off-by-one applied a spurious removal at `perturbation_start_step + 1`.

**Consequence:** any results from the June sweep should be discarded; rerun after fixes.

## 2026-08-16 — refactor

- Consolidated everything into `hybrid_sam.py`; deleted `muon_adam_perturb.py`,
  `adam_two_momentum_perturb.py`, `adam_wd_perturb.py` (git history keeps them). The
  two-momentum idea survives as `ascent_beta1` (separate EMA feeding the ascent direction —
  controls the "staleness" of the lookahead, which per MSAM's own analysis is what makes it an
  ascent; nobody has swept this).
- Perturbations are now **cached** (`state["perturb"]`) and removal subtracts the cached tensor:
  exact by construction, robust to intermittently-frozen params, halves the extra Newton-Schulz
  cost for muon ascent, and survives checkpoint save/resume (ε rides in the optimizer state).
- Added `unperturbed()` context manager + `remove_perturbation()`; train_gpt.py now evals at
  clean w, logs `eval_loss_perturbed`/`eval_sam_gap`, and unperturbs before the final save.
- Added `perturbation_norm="balanced"` (new default), `perturbation_start_step`,
  signed-ρ support, per-step stats (`optim/perturb_norm`, `optim/cos_perturb_grad`,
  per-family norms), lazy `exp_avg_sq` allocation.
- Fixed: `zeropower_via_newtonschulz5` could mutate its input in place when handed a bf16
  tensor; run-name fallback when no override JSON is provided.
- New test suite `tests/test_hybrid_sam.py` (11 tests, all passing): bit-exact ρ=0 ≡ Muon,
  ρ=0 ≡ AdamW (wd=0), sign/norm semantics, balanced-norm budget, remove/unperturbed round-trip,
  no-leak on grad-None, start-step gating, ascent_beta1, state_dict resume, lazy second moment,
  stats.

## Planned experiments (next sweep)

Priority order; all at the 1024×12 config, constant lr, 250k steps unless noted:

1. **A/A anchors** (cheap, run first): hybrid_sam muon-muon ρ=0 must overlay the Muon baseline
   curve; adam-adam ρ=0 must overlay AdamW. If not, stop and debug.
2. **Signed-ρ sweep**: ρ ∈ {−1.0, −0.3, −0.1, +0.1, +0.3, +1.0} for adam→muon and muon→muon.
   Decides lookahead (MSAM) vs ascent (SAM) mechanism. Note balanced-norm ρ is not comparable
   to MSAM's global-norm ρ values; the sweep range may need widening after seeing
   `optim/perturb_norm` relative to update norms.
3. **muon-nesterov control**: if ρ>0 wins, the honest baseline is Muon with nesterov=True
   (built-in cheap lookahead). Included in launch script.
4. **ascent_beta1 sweep** (after 2): {0.5, 0.8, 0.95, 0.99} at the best ρ — staleness of the
   lookahead as the mechanism knob.
5. **Fallback ablation**: muon ascent with fallback ∈ {skip, adam} under balanced norm —
   does perturbing embeddings matter at all?

## Evaluation methods (how to tell what these methods actually do)

Beyond val loss/ppl at matched steps (now correctly at clean w):

- **`eval_sam_gap`** (already logged): loss(w̃) − loss(w) over training. Sharpness proxy along
  the perturbation direction, per run, for free.
- **Weight-noise robustness** (probably the single most decisive final-checkpoint eval): add
  Gaussian noise ε~N(0,σ²) to all weights, plot ppl degradation vs σ for SAM runs vs baselines,
  ~5 seeds per σ. Direct, cheap, model-agnostic flatness measurement. Worth a small
  `eval_flatness.py` script.
- **Directional landscape slices**: 1D loss profiles w + t·d for d ∈ {update dir, ascent dir,
  random}; compare curvature around the found minima.
- **Hessian spectrum**: top-k eigenvalues (power iteration / Lanczos via HVP) and trace
  (Hutchinson). Feasible at 250M on one GPU with a few hundred batches.
- **Generalization gap**: train-vs-val loss gap at matched train loss (SAM's claim is a better
  gap, not a better train loss).
- **Held-out-distribution ppl**: wikitext-103, a Pile slice — flat minima are claimed to
  transfer better under distribution shift.
- **lr-robustness**: sweep lr ±2× around the tuned value at fixed ρ; flatness-inducing methods
  should widen the good-lr basin (also directly useful).
- **Weight-averaging interplay**: EMA/SWA of checkpoints should help flat-minimum runs more;
  cheap to test post-hoc from saved checkpoints (requires periodic checkpointing next sweep).
- **Batch-size sensitivity**: SAM effects often grow with batch size (gradient noise itself
  regularizes sharpness at small batch); one large-batch replicate of the headline comparison.

## Open ideas

- **Disagreement perturbation**: perturb along (adam_dir − muon_dir) — the component one
  preconditioner endorses and the other suppresses. Nearly free; probes curvature exactly where
  optimizer choice matters. Natural `ascent="adam_minus_muon"` extension.
- **Per-family sign**: SAM-ascent on matrices, MSAM-lookahead on embeddings (or vice versa).
- **Resurrect weight-decay perturbation** as `ascent="weights"` (direction −w) inside HybridSAM
  if ever wanted — it's a one-branch addition, not a separate optimizer.
- ρ schedule beyond start-step gating is deprioritized: MSAM found ρ∝lr coupling hurts.
