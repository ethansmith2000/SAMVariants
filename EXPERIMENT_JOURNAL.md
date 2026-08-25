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

**Our conventions** (`hybrid_sam.py`) — terminology, stated precisely:

Each step: **perturb w → compute the gradient at w̃ → revert → descend from w.**
The *sign of ρ* selects which of two algorithms that is:

| | ρ > 0 — **LOOKAHEAD** | ρ < 0 — **ASCENT** |
|---|---|---|
| ε points | *forward* along the descent direction | *uphill* along the gradient direction |
| measured | cos(ε, descent step) ≈ +0.9 | ≈ −0.9 |
| family | Nesterov / extragradient | classic SAM |

MSAM calls its ρ>0 perturbation "ascent" because the *loss* at w̃ is higher (momentum overshoots
the line-minimum), not because the direction is uphill. **We reserve "ascent" for ρ<0 and use
"lookahead" for ρ>0** — earlier journal entries used MSAM's looser wording.

`ascent=` / `descent=` name **geometries, not directions** (clearer aliases: `perturb_with=`,
`update_with=`). `ascent="adam", descent="muon", rho=4` reads: *look 4 Muon-steps ahead along the
Adam direction, then take a Muon step.*

**What ρ measures** — differs from MSAM, deliberately:
- ours, `perturbation_scale="relative"` (every sweep since the pilot): ‖ε_p‖ = |ρ| · EMA(‖actual
  update applied to p‖), **per parameter**. ρ is dimensionless: multiples of that parameter's own
  recent step length.
- MSAM: ε = −ρ·v/‖v‖ with **one global** Frobenius norm over all parameters concatenated, so their
  ρ is an absolute distance in weight space (hence their 1.7–5.5). Our `"absolute"` mode is the
  MSAM-like setting; the pilot used it.

Other: `perturbation_norm="balanced"` (absolute mode only) gives each param ‖ε_p‖ = ρ·√(numel_p/
total); raw `"global"` reproduces MSAM but is only safe when every param uses the same direction
family (see the 2026-08-16 finding). Weights live at w̃ between steps: eval inside
`optimizer.unperturbed()`, final save after `optimizer.remove_perturbation()`, and
`eval_sam_gap` = loss(w̃) − loss(w) is logged each validation.

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

## 2026-08-20 — sweep2 complete; regime critique; curvature disentangled

**Full sweep2** (12 runs, relative ρ; refs: muon 3.3824, nesterov 3.3774, AdamW 3.4129):

| ascent→descent, ρ_rel | eval loss |
|---|---|
| **adam→muon, 4** | **3.3711** (best; ≈1.5k steps / 6% ahead of muon, ≈0.9k ahead of nesterov) |
| muon→muon, 8 / 4 | 3.3738 / 3.3741 (broad plateau — peak ≥4) |
| muon→muon, 2 / 1 | 3.3766 / 3.3767 |
| adam→muon, 1 | 3.3773 |
| momentum→muon, 1 (MSAM-proper) | 3.3784 |
| muon→adam, 1 / 4 | 3.3839 / 3.3865 (worse with ρ) |
| muon→muon, −0.25 / −1 | 3.3866 / 3.3983 (ascent = monotone tax) |

Headline: **cross-optimizer perturbation beats MSAM-style momentum perturbation** (3.3711 vs
3.3784, ≈1k steps) and beats the nesterov control. Asymmetric: adam-ascent helps a muon
descender; muon-ascent hurts an adam descender.

**Confound in our own baseline** (queued as sweep3): HybridSAM's adam-descent takes
`beta1=muon_beta1=0.95` while the `mode=adamw` baseline uses `beta1=0.9`, so muon→adam vs AdamW
mixes ρ with a β₁ change. Proper A/A (muon→adam at ρ=0) + MSAM-proper at ρ=4 now running.

**Regime critique (Ethan).** SAM's flat-minima premise was validated on classifiers trained to
convergence; we run 0.82B tokens against an 8.6B-token corpus — <10% of one epoch, every sample
fresh, so there is no generalization gap at all and eval loss measures pure optimization progress.
SAM's mechanism has nothing to act on, and its cost is pure tax (matches the negative-ρ result).
Stronger: nothing has *converged*, so "flat vs sharp minima" is not merely inert but
category-confused — these are points mid-trajectory, not minima. Reframe the project as an
optimizer result (cross-geometry lookahead), and test the sharpness claim only in a regime where
generalization binds (multi-epoch on a subset, or downstream/OOD eval), where the sharp prediction
is that **negative ρ should flip from worst to plausibly best**.

**Curvature disentangled** (`eval_directional.py` over whole trajectories; step_* checkpoints of
SAM runs sit at w̃ — measured bias from step_25000 vs the clean final save: loss +0.020,
curvature −0.113 — so corrections are applied below):

1. *Most of "winners are sharper" was a progress artifact*, as suspected: curvature along the
   **baseline's own** trajectory triples (0.42 → 1.47) with no intervention, at ~10.9 curvature
   units per nat. Earlier framing retracted.
2. *A residual survives at matched loss*: winner's w̃-corrected 20k checkpoint (loss 3.846,
   curv 2.03) vs baseline interpolated to the same loss (curv 1.41) → **+44% curvature at equal
   loss**. Real, but smaller than the naive cross-run gap suggested.
3. *The adam-direction slope depletion is progress-invariant* — the cleanest mechanistic
   signature we have. Residual descent slope along sign(g) is flat across training for both runs
   yet ~1.9× apart: baseline 0.224–0.276 at every checkpoint, winner 0.127–0.137 at every
   checkpoint, from 10k onward. Perturbing along the adam direction genuinely harvests
   adam-geometry descent that pure-muon training leaves unmined, and it is not a byproduct of
   being further along.

Probe caveat: 64-sequence gradient/eval batches — base losses are too noisy to rank runs (ρ=−1
scores better than muon on them), so only paired/matched comparisons above are trusted.

## 2026-08-21 — sweep3 controls: two-mode framing, and a retraction

**Framing (Ethan).** The sign of ρ selects between two genuinely different algorithms:

- **Mode A (ρ<0)** — ascend in optim₁'s geometry → grad → undo → descend with optim₂. SAM-like.
- **Mode B (ρ>0)** — extra *forward* step in optim₁'s geometry → grad → undo → descend with
  optim₂. Nesterov/extragradient-like, in a possibly different geometry.

**Retraction.** Earlier claim "muon-ascent hurts an adam descender, monotonically in ρ" was an
artifact of the β₁-mismatched AdamW baseline. Proper A/A (`sweep3-ma-rel0`, β₁=0.95, ρ=0) =
3.3945; β₁ 0.9→0.95 alone accounts for 0.018 of the 0.029 gap. Against its *correct* baseline
the adam-descent arm **improves**: ρ=1 → 3.3839 (−0.0106), ρ=4 → 3.3865 (−0.0080). So Mode B
helps every descent geometry tested; what differs is the optimal ρ (muon-descent peaks ≥4,
adam-descent peaks near 1).

**Mode B summary, each against its own proper baseline:**

| ascent→descent | baseline | best | Δ |
|---|---|---|---|
| adam→muon | muon 3.3824 | 3.3711 (ρ=4) | −0.0113 |
| muon→muon | muon 3.3824 | 3.3738 (ρ=8) | −0.0086 |
| muon→adam | adam-β₁.95 3.3945 | 3.3839 (ρ=1) | −0.0106 |
| momentum→muon (MSAM) | muon 3.3824 | 3.3784 (ρ=1) | −0.0040 |
| nesterov control | muon 3.3824 | 3.3774 | −0.0050 |

**New: preconditioning determines how far you can extrapolate.** MSAM's raw-momentum ascent
*collapses* at large ρ (`sweep3-mom-m-rel4` = 3.3945, far worse than its ρ=1 3.3784 and worse
than baseline muon), while preconditioned ascent directions (adam, muon) keep improving to ρ=4–8.
Plausible reading: the raw momentum direction has wildly heterogeneous per-coordinate scale, so
extrapolating several steps along it is destructive; whitened/orthogonalized directions stay
well-conditioned far from the current iterate. This is a mechanism-level reason cross-optimizer
perturbation beats MSAM that has nothing to do with sharpness.

**Gap, now obvious under the two-mode framing:** Mode A has only ever been run in the *matched*
cell (muon-ascent → muon-descent). Mode A × cross-geometry is untested — and under SAM theory
that is the interesting one, since steepest ascent is norm-relative (Euclidean → gradient,
spectral → NS-orthogonalized, ~L∞ → sign-like), i.e. "which norm ball should SAM use for a
Muon-trained transformer" (cf. ASAM). Also still untested: true SAM with a *fresh* gradient at w
(2× cost) — our Mode A only approximates ascent using stale momentum buffers.

## 2026-08-21 — norm-matching caveat quantified; sweep4 (balanced grid)

**Ethan's caution on the MSAM-collapse claim is correct and measurable.** ρ matches the *L2 norm*
of the perturbation to the descent-step norm, but families distribute that budget completely
differently (measured on a trained checkpoint, per 2D weight matrix):

| direction family | max/RMS | participation ratio (1 = uniform) |
|---|---|---|
| raw momentum | 21.1 | 0.07 |
| adam (whitened) | 1.0 | 0.995 |
| muon (orthogonalized) | 5.5 | 0.31 |

At matched L2 norm the momentum perturbation concentrates ~all displacement in <10% of
coordinates, moving those ~21× RMS, while the adam direction spreads uniformly. So nominal ρ is
**not comparable across ascent families**, and "raw momentum collapses at ρ=4" is at least partly
"ρ=4 is a much larger effective lookahead for momentum". Retract the conditioning explanation as
stated; the defensible comparison is **peak-to-peak** (each family at its own optimal ρ), and
momentum's peak is undersampled (only ρ ∈ {1, 4} run).

**Design critique (Ethan):** the sweep varied geometry pair and ρ together, unbalanced, and the
low-ρ range Mode B most plausibly wants (0.25–2) was barely sampled — note `muon→adam` peaked at
ρ=1, the *lowest* value tested, the same edge-of-grid problem flagged earlier for muon-descent.
Also `adam→adam` (the matched cell for adam descent) and `momentum→adam` (MSAM's own Adam
variant!) had never been run.

**sweep4** (15 runs, queued): completes a balanced ascent {momentum, adam, muon} × descent
{muon, adam} × ρ ∈ {0.5, 1, 2} grid, plus ρ=0.25 for the two most promising cells, with existing
ρ ∈ {4, 8} points as upper-range context and `sweep3-ma-rel0` (3.3945) as the adam-descent ρ=0
anchor. This makes every family comparable peak-to-peak instead of at an arbitrary common ρ.

## 2026-08-21 — disk blowout: checkpoint policy tightened

Checkpoints filled ~500GB and blocked VSCode SSH; Ethan deleted `model-output`. No *results*
were lost (eval numbers live in the run logs and wandb) — only checkpoints, and only the
directional/flatness probes depend on those.

Arithmetic worth remembering: the 255M-param model is ~1GB fp32, and an accelerate `save_state`
adds optimizer state — `exp_avg` + `exp_avg_sq` + **HybridSAM's cached `perturb` tensor** — so
each step_* checkpoint is ~3–4GB, and one 15-run sweep at every-5k checkpointing is ~250GB. Our
own perturbation cache contributes ~1GB of that per checkpoint (the price of exact removal).

Policy now:
- `train_gpt.py` prunes to `keep_last_n_checkpoints` (default 1) after every periodic save, and
  drops resume checkpoints entirely at the end of training (`discard_checkpoints_at_end`, default
  true) since the final unperturbed model is saved separately.
- `checkpointing_steps` 5000 → 10000 in all sweep configs.
- `prune_checkpoints.sh` reaps for already-running jobs (which hold the pre-patch code); safe any
  time, since auto-resume only reads the newest checkpoint. Running in loop mode during sweep4.
- Steady state per run: ~1GB final model (+ ≤4GB transient while training) instead of ~20GB.

If trajectory probes (`eval_directional.py` over step_* checkpoints) are wanted for a specific
run, set `keep_last_n_checkpoints` high *for that run only* — don't re-enable it sweep-wide.

Note: `/workspace/.hf_home` holds 182GB, mostly the raw OWT download that the slim tokenized
cache (34GB) supersedes. Reclaimable if needed, at the cost of a ~30min re-download.

## 2026-08-21 — sweep4 partial: cross-geometry wins on BOTH descent sides

`collect_results.py` added — scans logs + configs and prints the grid (marks in-flight runs, and
segregates absolute-ρ pilot runs from relative-ρ sweeps, which are not comparable).

descent = **muon** (ρ=0 baseline 3.3824; nesterov control 3.3774):

| ascent \ ρ | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| momentum | – | run | 3.3784 | 3.3821 | 3.3945 | – |
| adam | 3.3771 | run | 3.3773 | run | **3.3712** | – |
| muon | – | 3.3790 | 3.3767 | 3.3766 | 3.3741 | 3.3738 |

descent = **adam** (ρ=0 baseline 3.3945, β₁=0.95):

| ascent \ ρ | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| momentum | – | **3.3829** | 3.3861 | run | – |
| adam | – | 3.3902 | run | run | – |
| muon | 3.3964 | – | 3.3839 | run | 3.3865 |

**The thesis now holds symmetrically: cross-geometry beats matched geometry on both sides.**
- descent=muon: best cross (adam→muon, 3.3712) beats best matched (muon→muon, 3.3738).
- descent=adam: best cross so far (momentum→adam 3.3829; muon→adam 3.3839) beats matched
  (adam→adam 3.3902).

Other reads (provisional, 8 cells still running):
- **MSAM-on-Adam (momentum→adam) is strong at low ρ** — 3.3829 at ρ=0.5, the best adam-descent
  number yet, −0.0116 vs its baseline. Note this is the paper's own Adam variant, run properly
  for the first time.
- Each ascent family has a distinct optimal ρ, consistent with the concentration measurements:
  momentum peaks lowest (≤1 for muon-descent, 0.5 for adam-descent) and degrades fast; muon
  peaks broad and high (4–8); adam is flat 0.25–1 then improves at 4. Ethan's [0.25, 2] intuition
  holds for the *concentrated* (momentum) directions; the whitened/orthogonalized ones tolerate —
  and prefer — much larger lookahead.
- `muon→adam` at ρ=0.25 is *worse* than baseline (3.3964, +0.0019) while ρ=1 is best for that
  cell: too small a lookahead is not merely weak but slightly harmful there.

## 2026-08-22 — 38-hour silent stall (post-mortem) + watchdog

Six sweep4 runs **deadlocked at 2026-08-21 06:22** and sat alive-but-frozen for 38 hours,
holding ~30GB of GPU each across six GPUs. Diagnosis: trainer blocked in `futex_wait_queue_me`
with all 8 dataloader workers in `do_poll`, zero log output, CPU time frozen. Timing coincides
exactly with the disk hitting 100% — a write blocking under ENOSPC while holding a lock is the
most plausible trigger. `model-output/sweep4_owt` had been deleted, so there was nothing to
resume from and the affected cells restart from scratch.

Cost: ~38h × 6 GPUs, and it blocked other projects' jobs from claiming those GPUs. **Nothing in
our tooling noticed** — the supervisor service showed RUNNING, gpu-claim showed the jobs HELD by
live PIDs, and progress only looked wrong when step counts were compared across days.

Fixes:
- `watchdog.sh` — kills our trainers whose log has not advanced in 25 min (safe: runs auto-resume;
  matches strictly on `/proc/<pid>/cwd` so other projects' `train_gpt.py` is never touched).
- `supervisor/sam_sweep4.sh` — skips configs whose log already contains "Saving model to", and
  gives each config up to 3 attempts, so a killed/stalled run retries instead of being lost.
- Root trigger (unbounded checkpoints filling the disk) already fixed 2026-08-21.

Lesson for the journal: "process alive + GPU claimed" is not progress. Compare step counts
against wall-clock, or watch log mtimes.

## 2026-08-22 — corrections and scope decisions (Ethan)

**Retraction: "the flatness story is dead" was overstated.** What the evidence supports is
narrower: *lookahead's* gains are not explained by flatness — and never should have been expected
to be, since ρ>0 is the opposite manoeuvre from SAM. The *ascent* branch's flatness claim is
**untested in an applicable regime**, not disproven: our runs are ~10% of one epoch, nowhere near
convergence, with no generalization gap for a flat-minimum effect to show up in. Ascent may well
need much longer training before it can pay off, and the ρ<0 numbers here should be read as
"no benefit at 25k steps", not "no benefit".

**Scope: drop momentum as an ascent geometry.** Perturbing along the raw momentum direction is
MSAM's own choice, not the cross-optimizer question this project exists to answer. Existing
results are kept (gm/ga cells in the grid), but no further compute goes there. Enforced by a
blocklist in `watchdog.sh` and removal from the sweep4 queue.

**Deadlock vs. queue — how to tell them apart** (both occurred, and they look identical in
`supervisorctl status`):
- *Deadlocked*: CPU time accumulated then froze, `wchan = futex_wait_queue_me`, GPU memory held
  at 0% utilization, log mtime static for hours. Kill it.
- *Queued*: no `train_gpt.py` process at all — only a `gpu-claim.py run --wait` waiter, no GPU
  memory held. Its log is stale simply because the run has not started. Leave it.

**ρ optimum splits by DESCENT geometry, not ascent geometry** — the clearest pattern in the grid:

| update with | best ρ | reading under Ethan's integrator framing |
|---|---|---|
| adam | 0.5–1 | ρ=0.5 ≈ RK2 midpoint, ρ=1 ≈ Heun/trapezoid predictor — the *classical* regime |
| muon | 4–8 | far past the endpoint; not a standard integrator at all |

Caveat before over-reading ρ=4: the perturbation direction is built from the momentum EMA
(β₁=0.95, horizon ~20 steps), so it is a *smoothed* direction. Displacing 4 step-norms along a
smoothed direction is not the same as being 4 actual steps ahead on a curving trajectory — the
true along-path displacement is smaller. Diagnostic to settle it: measure
cos(perturbation direction, actual displacement over the next k steps) and the ratio
‖ε‖ / ‖w_{t+k} − w_t‖, which converts ρ into "effective steps of lookahead".

## 2026-08-23 — lookahead diagnostic: rho=4 really is 4 steps, and my explanation was wrong

`diag_lookahead.py` (400 steps, real 1024×12 model, 4 watched matrices). S(k) = ‖Σuᵢ‖/Σ‖uᵢ‖ is
trajectory straightness (1 = straight line); ‖u‖/‖W‖ is per-step displacement relative to weight
norm.

| descent | cos(u_t,u_t₊₁) | S(2) | S(4) | S(8) | ρ=4 overshoot | cos(adam,muon) | ‖u‖/‖W‖ | ρ=4 → %‖W‖ |
|---|---|---|---|---|---|---|---|---|
| muon | 0.720 | 0.927 | 0.875 | 0.823 | 1.14× | 0.557 | 0.485% | 1.94% |
| adam | 0.971 | 0.993 | 0.981 | 0.945 | 1.02× | 0.424 | 0.193% | 0.77% |

1. **ρ=4 ≈ 4 real steps ahead.** Trajectories are nearly straight over 4–8 steps, so displacing
   4 update-lengths overshoots the true 4-step position by only 1.14× (muon) / 1.02× (adam). The
   curvature caveat I raised is resolved — Ethan's reading ("it's 4× the update, full stop") was
   right, and ρ is a faithful "steps of lookahead" unit.
2. **My straightness explanation for the descent-side ρ split is refuted, and backwards.** Adam's
   path is *straighter* (0.971 vs 0.720 one-step autocorrelation; S(8) 0.945 vs 0.823) yet Adam
   prefers ρ≈0.5–1 while Muon wants 4–8. Straightness does not explain the split.
3. **Nor is it a units artifact.** Per-step displacement differs only 2.5× (0.485% vs 0.193% of
   ‖W‖), while the ρ optima differ 4–8×; in %‖W‖ the optima are ~1.9–3.9% (muon) vs ~0.1–0.2%
   (adam), roughly 20× apart. The adam-descent arm genuinely wants a much smaller probe.
   **The descent-side split remains unexplained.**
4. **The winning config is mostly a sideways probe, not extrapolation.** cos(adam_dir, muon_dir)
   ≈ 0.56, so adam→muon at ρ=4 decomposes into ≈2.2 update-lengths *along* the descent direction
   and ≈3.3 *perpendicular* to it. That fits the progress-invariant slope-depletion signature
   (2026-08-19) better than any lookahead story: the gain looks like sampling gradients in
   directions Muon's own geometry never visits.
5. Side note: adam-descent's 0.971 update autocorrelation is very high, consistent with β₁=0.95
   (we use muon_beta1 for all hybrid runs) being over-smoothed relative to Adam's usual 0.9.

**sweep4 complete** (12/15 saved; the 3 unsaved are momentum-ascent cells we dropped, whose eval
numbers are already recorded). **sweep5 launched**: seed replicates (seeds 102/103) of
adam→muon ρ=4, muon baseline, and muon→muon ρ=4 — the n=1 problem is the main validity gap.
Note: unify `tokenized_dataset_path` when deriving configs from older sweeps — the pilot configs
still pointed at the deleted `_slim` cache, which would have silently re-tokenized *and* broken
comparability.

## 2026-08-23 — sweep6: is the mechanism lookahead, or a sideways probe?

**Correction to the sweep4 read.** I previously described a "descent-side split"
(adam-descent optimal at rho~0.5-1, muon-descent at 4-8) and called it
unexplained. It was an artifact of an under-swept column: adam->adam goes
0.5=3.3902, 1.0=3.3845, **2.0=3.3720** — still improving at the edge of its
range, and the largest single delta in the table (-0.0225 vs the adam
baseline 3.3945). Both descent geometries want large rho. There is likely no
split to explain. Two cells now sit at a swept-range edge (same mistake as the
pilot): adam->muon@4 (global best, 3.3712) with no rho=8, and adam->adam@2 with
no rho=4.

**The hypothesis under test.** diag_lookahead gave cos(adam_dir, muon_dir)=0.56,
so adam->muon@rho=4 decomposes into ~2.2 update-lengths forward and ~3.3
sideways. Matched muon->muon@4 is *pure* forward (4.0 lengths) and is worse
(3.3741 vs 3.3712). So the gain may come from sampling the gradient in
directions the descent geometry never visits, not from lookahead.

New optimizer options (default off; all 19 tests pass incl. the bit-exact
rho=0 == Muon/AdamW anchors):
- `ascent_orthogonalize` — project the ascent direction orthogonal to the
  descent direction: pure sideways, forward component removed. Reuses the
  descent direction cached during the step, so no second Newton-Schulz pass.
- `ascent="random"` — isotropic control at the same norm.
- `perturb_muon_eligible_only` — coverage-matched control. On non-muon-eligible
  params (embeddings, 1D) muon-descent falls back to adam, so an adam ascent
  direction is parallel there and the orthogonal probe drops them. Without this
  control, perp-vs-baseline would confound "forward component removed" with
  "embeddings no longer perturbed". A test asserts the two flags select exactly
  the same param set.

Queued (25k steps, seed 0, all else identical to sweep2-am-rel4):

| run | tests |
|---|---|
| `sweep6-am-perp4` | adam⊥muon -> muon, rho=+4: pure sideways |
| `sweep6-am-rel4-elig` | coverage-matched control for the above |
| `sweep6-am-perpn4` | same, rho=-4: a pure sideways probe should be ~sign-invariant |
| `sweep6-rand-m-rel4` | random -> muon, rho=4: does *any* off-trajectory probe help? |
| `sweep6-am-rel8` | closes the range edge on the global best |
| `sweep6-aa-rel4` | closes the range edge on adam-descent |
| `sweep6-am-reln1` | ascent quadrant is empty for cross-geometry |

Predictions worth recording before the numbers land: if the mechanism is
sideways probing, perp4 ~= am-rel4-elig and perpn4 ~= perp4; if it is genuine
lookahead, perp4 falls back toward the elig-baseline's rho=0. If rand-m-rel4
also helps, the effect is gradient smoothing rather than anything about Adam's
geometry — that would be the deflating outcome and is the reason the control
is in the batch.

**Still untested after sweep6:** true SAM with a fresh gradient (2x cost) as a
reference point; `ascent_beta1` staleness; lr decay; and the long-run regime.
25k steps is 0.82B tokens of an 8.6B-token corpus (<10% of one epoch) with no
generalization gap, so the flat-minima axis SAM targets is not merely inert
here but ill-defined — the ascent-side claim cannot be settled at this length.

## 2026-08-23 — long-run arm (100k steps, cosine decay)

Every result so far is from 25k steps = 0.82B tokens of an 8.6B-token corpus
(<10% of one epoch), constant LR, never converged, no generalization gap. Two
claims cannot be evaluated in that regime at all: whether the lookahead gain
persists with more tokens, and whether ascent (rho<0) pays off near convergence
— the flat-minima mechanism SAM actually targets. Three 100k-step runs
(~14h each) address it:

| run | config | question |
|---|---|---|
| `long-muon` | Muon, cosine to 0 | baseline |
| `long-am-rel4` | adam->muon, relative rho=+4 | does the best cell's gain survive 4x tokens + decay? |
| `long-mm-absn2` | muon->muon, **absolute** rho=-2.0 | does ascent pay off near convergence? |

Two deliberate design choices worth recording:

1. **Cosine decay, not constant LR.** Constant LR never converges, so the
   flat-minima axis stays ill-defined no matter how long we run. Decay is what
   makes the ascent question answerable. Cost: the long runs differ from the
   25k grid in length *and* schedule, so they are internally controlled
   (all three share the schedule) but not directly comparable cell-by-cell to
   the grid.
2. **The ascent arm uses absolute scale; the lookahead arm stays relative.**
   Under `relative`, ||eps|| tracks the update norm, which decays with the LR —
   so a relative-rho ascent run would shrink its perturbation radius toward zero
   exactly as convergence arrives, i.e. it would switch the treatment off at the
   only moment the hypothesis predicts an effect. Absolute holds the radius
   fixed. rho=-2.0 is calibrated, not guessed: the 25k `mm-reln1` run logged a
   total perturbation norm of 1.95, which also sits inside MSAM's published
   1.7-5.5 band. Relative stays correct for the lookahead arm, where "always
   look 4 of my own steps ahead" is the scale-free semantics we tuned.

`num_validation_batches` raised 25 -> 100: the effects we are chasing are ~0.01
nats and the grid's 25-batch estimate is too noisy to resolve that at n=1.

Also installed `sam_watchdog` as a supervisor service — it was written after the
38h stall but was not actually running. STALL_MIN raised 25 -> 45min, because
the long runs validate every 2500 steps (~21min) and 25min sits too close to a
normal quiet gap to be a safe kill threshold.

## 2026-08-25 — the watchdog killed 8 of 10 queued runs (my error)

Of the 10 jobs queued on 08-23, **2 completed and 8 were killed before they ever
started**. Cause: the `sam_watchdog` service I installed that same day.

`watchdog.sh` finds trainers with `pgrep -f "train_gpt.py --override"`. A
gpu-claim **waiter** carries the entire trailing command in its own argv, so
that pattern matches a job that is merely sitting in the queue holding no GPU.
A waiter never writes to its log, so the log's mtime stays at creation time and
the job looks "stalled" the instant STALL_MIN (45min) elapses — and gets killed.
Each config burned its 3 retries the same way, which is why 8 logs are exactly
0 bytes and both queue services exited early. The watchdog was written to
prevent wasted GPU-hours after the 38h deadlock; running it unguarded cost more
than the failure it was written for.

Two guards added, and verified against live processes on the box (a gpu-claim
wrapper and a real trainer are now correctly separated):
1. skip any process whose argv contains `gpu-claim` — only the real trainer
   child is ever a target;
2. skip any log of size 0 — an empty log means "not started yet", not "stalled".
Kills now also append to `slurm_logs/watchdog.log`; the previous `echo`s went to
supervisor's /dev/stdout and were unrecoverable exactly when needed.

### The two results that did land

**`long-muon` (100k steps, cosine to 0) — eval 2.9813.** Smooth descent
3.94 -> 2.99, no instability. For reference the 25k/constant-LR baseline was
3.3824. 100k steps = 3.3B tokens for this ~200M-param model, i.e. ~0.8x
Chinchilla-optimal, versus 0.82B (~0.2x) for the whole 25k grid. This is a
materially different regime, which is the point.

**`sweep6-am-perpn4` (adam⊥muon -> muon, rho=-4) — eval 3.6595.** Healthy
monotone run (5.67 -> 3.66), just far worse than the 3.3824 baseline (+0.277).
Its perturbation norm was 7.77 vs 1.95 for the 25k mm-reln1 run: a large,
purely off-trajectory displacement.

This is the first evidence against the **strong** form of the sideways-probe
hypothesis. A perturbation orthogonal to the descent direction is roughly
sign-symmetric by construction, so rho=-4 should stand in for |rho|=4 — and
pure sideways at that magnitude is strongly harmful, not beneficial. But the
picture is not simply "sideways bad": adam->muon@4 (2.2 forward + 3.3 sideways)
= 3.3712 still beats muon->muon@4 (4.0 forward, 0 sideways) = 3.3741. So a
moderate sideways admixture helps while pure sideways is destructive — a
non-monotonicity that one datapoint cannot resolve. `sweep6-am-perp4` (+4) and
`sweep6-am-rel4-elig` (coverage control) are requeued and settle it.

### Changes to the long-run design (Ethan, 2026-08-25)

- **Ascent arm switched from absolute back to relative scale.** Ethan's
  objection is correct and my earlier reasoning was backwards: running the
  ascent arm on `absolute` while the lookahead arm ran `relative` confounds
  *sign* with *scale semantics*, which is the single comparison the arm exists
  to make. Consistency wins; `long-mm-absn2` is replaced by `long-mm-reln1`
  (relative rho=-1, matching the 25k `mm-reln1` datapoint 3.3983 for a clean
  short-vs-long comparison). If that arm shows nothing, an absolute-rho variant
  is the follow-up that distinguishes "ascent doesn't work" from "the radius
  decayed away with the LR" — as a deliberate ablation, not a mixed default.
- **Linear decay is the default going forward** (Ethan's preference). This
  study keeps cosine, because `long-muon` is already complete and re-running a
  13h baseline to change decay shape buys nothing: the comparison that matters
  is internal to the study and all three arms share the schedule.
- Ethan's point on run length is the load-bearing one: decay only manufactures
  convergence if the run is long enough that the schedule has actually come
  down. A 5-10k-step slice of a 100k schedule still sits near peak LR, so short
  runs cannot fake convergence by decaying faster.

## 2026-08-25 (pm) — sweep6 results: the mechanism decomposes into thirds

Watchdog fix held; 5/7 cells complete (`am-rel8`, `aa-rel4` still running).

| run | eval | vs muon 3.3824 |
|---|---|---|
| `am-rel4` (all params) | **3.3712** | -0.0112 |
| `am-rel4-elig` (muon-eligible only) | 3.3761 | -0.0063 |
| `am-perp4` (pure sideways, rho=+4) | 3.3793 | -0.0031 |
| `rand-m-rel4` (random probe, rho=4) | 3.3838 | **+0.0014** |
| `am-perpn4` (pure sideways, rho=-4) | 3.6595 | +0.277 |
| `am-reln1` (cross-geometry ascent, rho=-1) | 3.4735 | +0.091 |

**The random control does nothing** (3.3838 vs baseline 3.3824). An isotropic
off-trajectory probe at the same norm (8.10 vs 7.79) buys exactly zero. The
effect is specific to Adam's geometry: not gradient smoothing, not noise
regularization, not "any perturbation of this size". This was the outcome that
would have deflated the whole project, and it is ruled out.

**Neither hypothesis was right; the gain splits into ~equal thirds** of am-rel4's
0.0112 over baseline:
- **~0.0049** from perturbing the *non*-muon-eligible params (embeddings, 1D).
  Those are Adam-descended, so this is adam->adam lookahead — the single largest
  contributor, and consistent with `aa-rel2` being the strongest cell in the
  grid (-0.0225 vs the adam baseline 3.3945). I had treated this coverage
  difference purely as a confound to control away; it is actually where most of
  the effect lives.
- **~0.0031** sideways component (perp4 3.3793 vs its coverage-matched control
  elig 3.3761).
- **~0.0032** forward/lookahead component.

**Retraction: "an orthogonal perturbation is sign-symmetric by construction".**
I used that this morning to argue `am-perpn4` (rho=-4) could stand in for
|rho|=4. It is false, and the gap between `am-perp4` (3.3793) and `am-perpn4`
(3.6595) is the measurement of how false. The direction is orthogonal to the
*descent direction*, not to the *gradient*: Muon's orthogonalized step is not
the gradient, so the projected Adam direction keeps a large gradient component.
rho>0 still probes partly downhill and rho<0 partly uphill. `am-perpn4` was
measuring ascent, not sideways — which is why the earlier journal entry's
"first evidence against the sideways-probe hypothesis" was reading the wrong
variable. The actual sideways datapoint, perp4, is mildly *beneficial*.

`am-reln1` = 3.4735 also makes cross-geometry ascent **worse** than matched
ascent (`mm-reln1` 3.3983). The ascent side degrades further the more the
geometries are allowed to differ — the mirror image of the lookahead side,
where crossing geometries helps.

## 2026-08-25 (eve) — back to basics: the clean 2-axis grid

Per Ethan: the perpendicular/orthogonalized perturbation is removed from
`hybrid_sam.py` and `train_gpt.py` (it was a curveball, and it produced one
wrong inference — see the retraction above). `ascent="random"` and
`perturb_muon_eligible_only` are kept as inert defaults so the two completed
control results stay reproducible; both are one-line opt-ins, not part of the
design. 16 tests pass, rho=0 == Muon/AdamW anchors still bit-exact.

The design is two axes: **{adam,muon} perturb x {adam,muon} update** and
**signed relative rho**.

### Seed replicates: the grid is trustworthy at n=1

| config | n | mean | sd |
|---|---|---|---|
| muon baseline | 3 | 3.38241 | **0.00033** |
| adam->muon rho=4 | 3 | 3.36956 | 0.00150 |
| muon->muon rho=4 | 2 | 3.37383 | 0.00026 |

Baseline seed sd is 0.0003; the effects are 0.005-0.034. Single-seed cells are
fine for anything above ~0.003. This closes the n=1 validity gap that has been
the top open issue since the pilot.

### The grid (eval loss; rho=0 baselines: adam 3.3945, muon 3.38241)

| perturb->update | -2 | -1 | -0.5 | +0.5 | +1 | +2 | +4 | +8 |
|---|---|---|---|---|---|---|---|---|
| adam->adam | q | q | q | 3.3902 | 3.3845 | 3.3720 | **3.3606** | q |
| adam->muon | q | 3.4735 | q | 3.3767 | 3.3773 | 3.3737 | 3.3696* | 3.3702 |
| muon->adam | q | q | q | 3.3986 | 3.3839 | 3.3902 | - | - |
| muon->muon | q | 3.3983 | q | 3.3790 | 3.3767 | 3.3766 | 3.3738* | 3.3738 |

*seed-mean. `q` = queued (sweep7, 10 cells, all rho<0).

### The headline changed: matched adam->adam lookahead now wins

`aa-rel4` = **3.3606** is the best cell in the project, beating the previous
best (adam->muon@4, 3.3696) and every muon-descent cell. Verified: monotone
4.64 -> 3.36, correct config, perturb_norm 3.73.

Two consequences:
1. **The cross-optimizer story is weaker than it looked.** The single strongest
   effect is *matched-geometry* lookahead on Adam (-0.0339 vs the adam
   baseline), i.e. essentially an aggressive Nesterov/extragradient effect, not
   a hybrid-geometry effect. Cross-geometry still helps on the muon-descent
   side, but it is no longer the headline.
2. **rho in [0.5, 2] is the wrong search range.** adam->adam is monotone
   improving across 0.5 -> 1 -> 2 -> 4 (3.3902, 3.3845, 3.3720, 3.3606) and has
   not turned over. adam->muon saturates around 4-8 (3.3696, 3.3702). Every
   optimum found so far sits at rho >= 4, i.e. *outside* the proposed typical
   range. `sweep7-aa-rel8` is queued to find where adam->adam turns over.

Ascent (rho<0) remains uniformly bad and worsens with |rho| and with geometry
mismatch: mm -0.25=+0.004, mm -1=+0.016, am -1=+0.091 vs their baselines. The
10 queued cells complete the negative half of the map rather than probing a
live hypothesis.

## 2026-08-25 (night) — LONG RUNS REVERSE THE RESULT

| run | 100k steps, cosine to 0 | vs baseline |
|---|---|---|
| `long-muon` (baseline) | **2.98133** | — |
| `long-am-rel4` (adam->muon rho=+4) | 2.98724 | **+0.0059 (worse)** |
| `long-mm-reln1` (muon->muon rho=-1) | 2.98770 | +0.0064 (worse) |

At 25k steps with constant LR, adam->muon@4 beat the Muon baseline by **-0.0129**
(3-seed means, baseline seed sd 0.00033). At 100k steps with cosine decay, the
same config **loses by +0.0059**. The sign of the effect flips.

The lookahead gain measured across the entire grid so far is, on this evidence,
a short-horizon / constant-LR artifact. It does not survive 4x the tokens with a
decaying LR — which is the regime any real claim has to hold in.

What this does and does not establish:
- It does **not** show lookahead can never help at longer horizons. rho=4 was
  tuned at 25k/constant; the honest statement is that the *tuned configuration
  does not transfer*, and an optimum re-tuned at 100k/cosine is untested.
- The long runs are **n=1**. Seed sd is known only at 25k (0.00033 for the
  baseline). The +0.0059 gap is ~18x that, so it is probably real, but long-run
  seed noise is unmeasured. Seeds at 100k are the obvious next spend.
- The relative-rho/decay interaction is a genuine confound here: under cosine
  decay a relative rho shrinks the perturbation with the LR, so the treatment
  fades late in training. That predicts convergence *to* baseline, though, not
  a loss to it.
- The ascent arm (`long-mm-reln1`, +0.0064) is no better than lookahead at long
  horizon. The flat-minima hypothesis gets no support even in the regime that
  was supposed to favour it. That was the main reason for running long, and the
  answer is negative.

### Prior art: the adam->adam arm is largely known territory

Ethan's instinct was right. Evaluating the gradient at a point extrapolated
along the update direction and applying the step from the original point is:
- **Nesterov/NAG in its "true" form**, and for Adam specifically **NAdam**
  (Dozat 2016) and **Adan** (arXiv 2208.06677) — both reformulate the
  extrapolation into momentum algebra to avoid the extra evaluation, with the
  extrapolation length fixed by beta1 (~one update).
- **Extragradient** (Korpelevich 1976) and **ExtraAdam / Optimistic Adam**
  (Gidel et al., arXiv 1802.10551; Daskalakis et al. 2018) from the
  variational-inequality / GAN literature. ExtraAdam already carries a
  *separately tunable* extrapolation step size, which is close to our rho.
  That literature also reports that forcing extrapolation = update step size
  hurts, i.e. our "rho > 1 is better" finding has precedent.
- Note **Lookahead (Zhang et al. 2019)** is a *different* algorithm despite the
  name: slow/fast weight interpolation, no extrapolated gradient evaluation.

So `adam->adam` with rho>0 is best described as NAG-form Nesterov-Adam with a
tunable, larger-than-standard extrapolation — not a new algorithm. Queued
`sweep7-nadam` (torch NAdam, beta1=0.95 to match, decoupled wd) to measure how
much of aa-rel4's -0.0339 is simply NAdam. On the muon side the analogous
control (muon-nesterov 3.3774 vs muon 3.38241) recovers ~58% of mm-rel4's gain,
so a large fraction of the adam side is likely prior art too.

### Incident: duplicate concurrent runs from `supervisorctl restart`

Restarting `sam_sweep7b` to append the NAdam job killed only the wrapper
script, leaving its gpu-claim wrapper and trainer orphaned and still holding a
GPU; the new service then started a **second** `sweep7-aa-rel8` that trained
concurrently into the same log and checkpoint directory (GPU 4 and GPU 1, ~1.8h
overlap). Killed the orphan by PID and kept the supervised instance, which
resumed from the step-10000 checkpoint. `sweep7-aa-rel8.log` contains
interleaved output from both instances up to 23:50 and should not be trusted
for anything but the final eval line.

Fix: `stopasgroup=true` / `killasgroup=true` added to every `supervisor/*.conf`
in the repo. NOT yet applied to `/etc/supervisor/conf.d` — `supervisorctl
update` restarts changed programs, which would kill sweep7's six running jobs.
Apply at the next natural gap. Two trainers (`sweep7-aa-rel8`, `sweep7-nadam`)
are currently running unsupervised as a result; the watchdog still covers stalls.
