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
