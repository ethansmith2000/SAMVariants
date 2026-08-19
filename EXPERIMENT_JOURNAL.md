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

## 2026-08-17 — harness validated end-to-end (local A/A anchors)

Ran the full train_gpt.py pipeline locally on the Vast box (8×RTX 5090, shared with other
jobs — use `launch_local.sh`, which picks GPUs; `GPUS="..."` restricts the set): wikitext-2,
512-dim/6-layer model, bs 8, 200 steps, evals every 50.

- **A/A anchor PASSED**: `hybrid_sam` muon→muon ρ=0 matches the Muon baseline to ~1e-4
  eval loss across all evals (GPU nondeterminism noise); `sam_gap` exactly 0.0 at ρ=0.
- **ρ=0.5 smoke**: perturbation/stats/gap machinery works. `sam_gap` came out small and
  *negative* (−0.002 → −0.0003 nats over training): the MSAM-sign perturbation locally
  lowers loss early on. Also note the gap magnitude ⇒ balanced-norm ρ=0.5 is a gentle
  perturbation; the sweep's ρ range likely needs to extend higher (watch
  `optim/perturb_norm` vs update norms in the first real runs).
- `eval_flatness.py` validated on the produced checkpoints (indistinguishable at 200
  steps, as expected).
- Gotcha for local runs: dataset name must be `Salesforce/wikitext` (new `datasets`
  rejects un-namespaced ids); set `TMPDIR` (train_gpt defaults it to a cluster path).

## 2026-08-17 — OWT pilot launched (7 runs, 25k steps)

Tokenized OpenWebText into the shared cache `/workspace/data/tokenized/openwebtext_gpt2_1024`
(8.37M train / 443k val blocks of 1024; `prep_dataset.py`, parallel save). Pilot on the local
box (wandb `sam-variants-llm`, configs `sweep_configs/pilot-*.json`, `launch_local.sh` over
GPUs 0,1,3,4,6): muon, muon-nesterov, hybrid muon→muon ρ ∈ {0, +0.3, +1.0, +3.0, −1.0};
1024×12 geglu, bs 32, constant lr 4e-4 / muon 6e-3, balanced norm, start_step 100,
`compile_mode: default` (reduce-overhead's cudagraph pools cost ~4GB and OOM'd shared GPUs).

Step-1000 sanity: at-scale A/A PASSED (muon 4.38852 vs hybrid ρ=0 4.38821, sam_gap 0.0);
ρ=1 shows sam_gap +0.0005 and eval 4.3661. Scale note: balanced ρ=1 ⇒ total perturbation
norm ≈ half an optimizer step (lr·‖muon dirs‖ ≈ 1.8 globally), so {0.3, 1, 3} spans
~0.15–1.5 steps of lookahead distance.

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

## 2026-08-18 — box migration, slim data, gpu-claim integration

Old box died mid-pilot; its auto-commit had pushed everything to GitHub, so no code loss. New
box (8×5090, shared with the other projects) rebuilt: OWT re-tokenized into the **slim format**
(int32 input_ids only — attention_mask was all-ones, labels duplicated input_ids; 34GB vs 110GB,
stays in page cache, kills the random-read IO stalls that throttled the old pilot to 0.16 it/s).
Train loop derives labels and casts to long.

Launch lessons, learned the hard way: (1) index-based CUDA_VISIBLE_DEVICES is unsafe (CUDA
FASTEST_FIRST ordering ≠ nvidia-smi PCI order) — pin by UUID with CUDA_DEVICE_ORDER=PCI_BUS_ID;
(2) this box's projects coordinate GPUs via the shared `gpu-claim` protocol
(/workspace/GPU_QUEUEING.md) — the pilot service is now a thin queue of
`gpu-claim run --owner samvariants --job <cfg> --wait` calls, one per config, runs starting
whenever a GPU frees; (3) never kill `train_gpt.py` by name — other projects use the same
filename (ComboAdam's trainer was collateral damage twice; it recovered).

Pilot restarted from scratch under the queue (~2 it/s per run when placed → ~3.5h/run).

## 2026-08-19 — pilot results: lookahead sign wins monotonically; flatness story inverted

All 7 runs completed 25k steps (bs 32×1024, constant lr). Final eval (clean w, 26×32 val batches):

| run | eval loss | ppl | final sam_gap |
|---|---|---|---|
| hybrid ρ=+3.0 | **3.3750** | **29.23** | +0.0019 |
| muon-nesterov | 3.3774 | 29.30 | — |
| hybrid ρ=+1.0 | 3.3800 | 29.37 | +0.0004 |
| hybrid ρ=+0.3 | 3.3820 | 29.43 | +0.0001 |
| muon | 3.3824 | 29.44 | — |
| hybrid ρ=0 (A/A) | 3.3831 | 29.46 | 0.0 |
| hybrid ρ=−1.0 | 3.3884 | 29.62 | −0.0005 |

**Findings** (single seed each; margins are small but the *orderings* are clean):
1. Strictly monotone in ρ: +3 > +1 > +0.3 > 0 ≈ muon > −1. The sign ablation is decisive:
   MSAM-sign lookahead helps, classic SAM-ascent (ρ<0) hurts.
2. ρ=3 beats the nesterov control by 0.0024 nats; nesterov beats muon by 0.0050. So ~2/3 of
   the gain is generic lookahead, ~1/3 is specific to the hybrid perturbation — and best-ρ is
   at the top of the swept range.
3. A/A anchor held over 25k steps (ρ=0 vs muon: 7e-4, noise level).
4. sam_gap is positive for ρ>0, negative for ρ<0, scaling with ρ — consistent with the
   overshoot interpretation throughout training.

**Flatness probe inverts the naive story** (eval_flatness, relative Gaussian weight noise,
3 seeds × 20×16 sequential val batches): degradation at σ=0.1 orders almost exactly
*inversely* to eval performance — ρ=−1 flattest (+0.070), then ρ=0 (+0.079), muon (+0.081),
ρ=0.3 (+0.086), ρ=1 (+0.096), ρ=3 (+0.104), nesterov sharpest (+0.109). The winners live in
*sharper* minima by this measure. Mechanism looks optimization-dynamical (extragradient-like)
rather than flat-minima-geometric. Caveats: small correlated eval subset (its σ=0 ordering
disagrees with the training eval — trust per-model deltas, not absolutes), one seed per config.

**Next sweep**: extend ρ to {5.6, 10} (win is at the range edge); 3 seeds at {muon, nesterov,
ρ=3, best-new-ρ} for error bars; ascent_beta1 sweep at best ρ (staleness knob); adam→muon arm;
flatness with a larger shuffled eval subset + directional (update-dir/ascent-dir) sharpness
slices, since isotropic noise may be the wrong probe. Dataset note: shared cache was rebuilt as
`/workspace/data/tokenized/openwebtext_gpt2_bs1024` (slim schema, same script) — use that path.

## 2026-08-19 — sweep2 (partial) + directional landscape probe

**Relative ρ** (`perturbation_scale="relative"`, ‖ε_p‖ = ρ·EMA‖step_p‖) implemented; pilot winner
maps to ρ_rel≈1.2. Sweep2 (12 runs, cross-optimizer + peak hunt + negative dose-response) queued
via gpu-claim; 5 done at time of writing:

| ascent→descent, ρ_rel | ppl | vs refs |
|---|---|---|
| **adam→muon, 4** | **29.11** | ≈1.6k steps (6.4%) ahead of muon; 0.9k ahead of nesterov |
| muon→muon, 4 | 29.20 | peak is ≥4, rel8 pending |
| muon→muon, 2 | 29.27 | |
| muon→adam, 1 | 29.49 | worse with ρ: 4 → 29.56 (adamw baseline pending) |

Cross-optimizer asymmetry: adam-ascent helps the muon descender (new best); muon-ascent hurts
the adam descender, monotonically in ρ.

**Directional probe** (`eval_directional.py`: loss slices ±8 step-units along grad / muon-dir /
sign(g) / random, fresh val gradient): (1) random directions dead flat (±5e-4) while gradient-
family directions move 0.1–10 nats — the landscape is entirely trajectory-anisotropic, isotropic
flatness probes measure ~nothing; (2) "sharper but better" holds along mechanism directions —
directional curvature orders ρ=−1 (1.40) < muon (1.47) < nesterov (1.69) < mm-rel4 (2.02) <
am-rel4 (2.41); (3) **fingerprint**: am-rel4's residual descent slope along the adam/sign
direction is ~half everyone else's (0.13 vs 0.17–0.35) — the adam-direction lookahead harvested
adam-geometry descent that pure-muon training leaves unmined. Predicts the asymmetry: Adam
already covers spectral directions decently, so muon→adam has less to harvest (matches results).
Caveats: one seed, one 64-seq gradient batch, sign(g) is a crude adam proxy.
