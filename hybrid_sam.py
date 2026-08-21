"""HybridSAM: SAM-style optimizer with configurable ascent/descent directions.

Follows the Momentum-SAM (MSAM, arXiv:2401.12033) convention:

    w̃ = w - rho * d / ||d||        (perturbation, applied AFTER each step)

where d is an "ascent direction" built from optimizer buffers (momentum EMA,
Adam direction, or Muon-orthogonalized momentum) and points *uphill* (the
buffers accumulate gradients). So for rho > 0 the perturbation -rho*d̂ moves
FORWARD along the descent direction — Nesterov-style lookahead, confirmed by
cos(eps, descent step) ~ +0.9 in tests. MSAM labels this "ascent" because the
loss at w̃ is higher (past ~1-2 steps you overshoot the line-minimum), not
because the direction is uphill. Passing a negative rho perturbs along +d,
i.e. the true gradient/uphill direction: classic SAM-style ascent.

Invariant: between steps, param.data holds the perturbed weights w̃, so the
gradient PyTorch computes at forward time is grad L(w̃). step() removes the old
perturbation (exactly, from a cached buffer), descends from the clean w, then
applies a fresh perturbation for the next forward pass.

Evaluation / saving: weights are perturbed between steps, so use

    with optimizer.unperturbed():
        evaluate(model)

for eval, and optimizer.remove_perturbation() once before the final save.

Shared state: exp_avg / exp_avg_sq are updated from the perturbed-point
gradient and shared by ascent and descent (as in MSAM). If ascent_beta1 is set
(and differs from beta1), a separate EMA with that beta feeds the ascent
direction instead — the ascent buffer's lag behind the iterate is what makes
the lookahead an ascent, so this knob directly controls perturbation "staleness".
"""

from contextlib import contextmanager
from typing import Literal, Optional

import torch

try:
    from .utils import zeropower_via_newtonschulz5
except ImportError:
    from utils import zeropower_via_newtonschulz5


def _adam_direction(exp_avg, exp_avg_sq, step, beta1, beta2, eps):
    """Adam's update direction (unscaled by lr), bias-corrected."""
    m_hat = exp_avg / (1 - beta1 ** step)
    v_hat = exp_avg_sq / (1 - beta2 ** step)
    return m_hat / (v_hat.sqrt() + eps)


def _muon_direction(buf, grad, beta1, ns_steps, nesterov, rescale=True):
    """Muon's update direction (unscaled by lr)."""
    update = grad.lerp(buf, beta1) if nesterov else buf
    og_shape = update.shape
    if update.ndim != 2:
        update = update.reshape(update.size(0), -1)
    g = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if rescale:
        g *= max(1, g.size(0) / g.size(1)) ** 0.5
    return g.reshape(og_shape).type_as(buf)


def _is_muon_eligible(param, group):
    max_dim = group["muon_max_dim"]
    return param.ndim == 2 and (max_dim is None or max(param.shape) <= max_dim)


class HybridSAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        muon_lr=None,
        rho=1.0,                          # perturbation magnitude; >0 = MSAM lookahead, <0 = SAM ascent
        ascent: Literal["momentum", "muon", "adam"] = "muon",
        descent: Literal["momentum", "muon", "adam"] = "adam",
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        nesterov=True,
        ascent_beta1: Optional[float] = None,   # separate EMA beta for the ascent buffer
        perturbation_start_step=0,              # rho=0 until this step (MSAM does this for warmup)
        normalize_perturbation=True,
        perturbation_norm: Literal["per_param", "global", "balanced"] = "balanced",
        perturbation_scale: Literal["absolute", "relative"] = "absolute",
        step_norm_beta=0.9,                     # EMA beta for per-param update-norm tracking
        muon_max_dim=16384,
        muon_fallback_ascent: Literal["skip", "momentum", "adam"] = "skip",
        track_stats=False,
        perturb_with=None,   # clearer alias for `ascent` (a geometry, not a direction)
        update_with=None,    # clearer alias for `descent`
    ):
        # `ascent`/`descent` name which optimizer's *geometry* builds the
        # perturbation and which performs the update. They say nothing about
        # direction: the sign of rho decides that (rho>0 lookahead / forward,
        # rho<0 ascent / uphill).
        ascent = perturb_with if perturb_with is not None else ascent
        descent = update_with if update_with is not None else descent
        # perturbation_scale="relative": ||eps_p|| = |rho| * EMA(||descent step of p||),
        # i.e. rho is dimensionless — "how many of my own update steps of lookahead".
        # Comparable across ascent/descent families, optimizers, and model scales
        # (an absolute budget means a very different relative nudge for a param
        # that moves 0.2/step than one that moves 0.002/step). Note this couples
        # perturbation size to lr; under constant lr it is a pure
        # reparameterization of "absolute", under lr decay it is a distinct
        # (MSAM-discouraged) choice — treat that as an ablation, not a default.
        # In relative mode perturbation_norm is ignored.
        muon_lr_mult = 1.0 if muon_lr is None else muon_lr / lr
        defaults = dict(
            lr=lr, muon_lr_mult=muon_lr_mult, rho=rho, ascent=ascent, descent=descent,
            beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, ns_steps=ns_steps,
            nesterov=nesterov, ascent_beta1=ascent_beta1,
            perturbation_start_step=perturbation_start_step,
            normalize_perturbation=normalize_perturbation,
            perturbation_norm=perturbation_norm,
            perturbation_scale=perturbation_scale,
            step_norm_beta=step_norm_beta,
            muon_max_dim=muon_max_dim,
            muon_fallback_ascent=muon_fallback_ascent,
        )
        super().__init__(params, defaults)
        self.track_stats = track_stats
        self.last_stats = {}

    # ------------------------------------------------------------------ state

    def _needs_second_moment(self, param, group):
        """Only allocate/update exp_avg_sq when some path actually uses Adam."""
        eligible = _is_muon_eligible(param, group)
        descent_adam = group["descent"] == "adam" or (group["descent"] == "muon" and not eligible)
        ascent_adam = group["ascent"] == "adam" or (
            group["ascent"] == "muon" and not eligible and group["muon_fallback_ascent"] == "adam"
        )
        return descent_adam or ascent_adam

    def _uses_ascent_buffer(self, group):
        ab = group["ascent_beta1"]
        return ab is not None and ab != group["beta1"]

    def _init_state(self, param, grad, state, group):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        if self._needs_second_moment(param, group):
            state["exp_avg_sq"] = torch.zeros_like(grad)
        if self._uses_ascent_buffer(group):
            state["exp_avg_ascent"] = torch.zeros_like(grad)

    # ------------------------------------------------------------- directions

    def _ascent_buffer(self, state, group):
        return state["exp_avg_ascent"] if self._uses_ascent_buffer(group) else state["exp_avg"]

    def _ascent_beta1(self, group):
        return group["ascent_beta1"] if self._uses_ascent_buffer(group) else group["beta1"]

    def _ascent_direction(self, param, state, group, mode=None):
        """Direction for the perturbation. Returns (direction, family) or (None, None)."""
        mode = mode or group["ascent"]
        buf = self._ascent_buffer(state, group)
        if mode == "momentum":
            return buf, "momentum"
        if mode == "adam":
            return _adam_direction(
                buf, state["exp_avg_sq"], state["step"],
                self._ascent_beta1(group), group["beta2"], group["eps"],
            ), "adam"
        if mode == "muon":
            if not _is_muon_eligible(param, group):
                fallback = group["muon_fallback_ascent"]
                if fallback == "skip":
                    return None, None
                return self._ascent_direction(param, state, group, mode=fallback)
            # rescale=False: per-param/balanced norms make it a no-op, and under
            # a raw global norm it would skew budget toward wide matrices.
            return _muon_direction(
                buf, buf, group["beta1"], group["ns_steps"], nesterov=False, rescale=False,
            ), "muon"
        raise ValueError(f"Unsupported ascent mode: {mode}")

    def _descent_direction(self, param, state, grad, group):
        if group["descent"] == "momentum":
            return state["exp_avg"]
        if group["descent"] == "muon" and _is_muon_eligible(param, group):
            return _muon_direction(
                state["exp_avg"], grad, group["beta1"], group["ns_steps"], group["nesterov"],
            )
        if group["descent"] in {"adam", "muon"}:
            return _adam_direction(
                state["exp_avg"], state["exp_avg_sq"], state["step"],
                group["beta1"], group["beta2"], group["eps"],
            )
        raise ValueError(f"Unsupported descent mode: {group['descent']}")

    def _descent_lr(self, param, state, group):
        if group["descent"] == "muon" and _is_muon_eligible(param, group):
            return group["lr"] * group["muon_lr_mult"]
        return group["lr"]

    # ---------------------------------------------------------- perturbation

    def _iter_params(self):
        for group in self.param_groups:
            for param in group["params"]:
                yield group, param

    def remove_perturbation(self):
        """Permanently move params back to clean w (exact: uses cached tensors).

        Covers every param with a stored perturbation, whether or not it has a
        grad this step, so intermittently-frozen params cannot leak.
        """
        for _, param in self._iter_params():
            state = self.state.get(param)
            if state:
                perturb = state.pop("perturb", None)
                if perturb is not None:
                    param.data.sub_(perturb)

    @contextmanager
    def unperturbed(self):
        """Temporarily evaluate/save at clean w; restores w̃ bit-exactly on exit."""
        stashed = []
        for _, param in self._iter_params():
            state = self.state.get(param)
            if state and state.get("perturb") is not None:
                param.data.sub_(state["perturb"])
                stashed.append((param, state["perturb"]))
        try:
            yield
        finally:
            for param, perturb in stashed:
                param.data.add_(perturb)

    def _apply_new_perturbations(self, items):
        """Compute ascent directions, normalize, apply, and cache the applied ε."""
        stats_enabled = self.track_stats
        pending = []
        for item in items:
            group, param, state = item["group"], item["param"], item["state"]
            if group["rho"] == 0 or state["step"] <= group["perturbation_start_step"]:
                continue
            direction, family = self._ascent_direction(param, state, group)
            if direction is None:
                continue
            pending.append(
                {"group": group, "param": param, "state": state,
                 "grad": item["grad"], "direction": direction, "family": family}
            )
        if not pending:
            self.last_stats = {}
            return

        norm_mode = pending[0]["group"]["perturbation_norm"]
        normalize = pending[0]["group"]["normalize_perturbation"]
        eps = pending[0]["group"]["eps"]

        if normalize and norm_mode == "global":
            global_norm = torch.stack(
                [p["direction"].norm() for p in pending]
            ).norm().clamp_min(eps)
        elif normalize and norm_mode == "balanced":
            total_numel = sum(p["direction"].numel() for p in pending)

        stats = {"perturb_sq_total": 0.0, "dot_pg": 0.0, "grad_sq": 0.0}
        family_sq = {}
        for p in pending:
            group, direction = p["group"], p["direction"]
            if group["perturbation_scale"] == "relative":
                # rho in units of this param's own (EMA'd) update-step norm
                scale = (
                    -group["rho"] * p["state"]["step_norm_ema"]
                    / direction.norm().clamp_min(eps)
                )
            elif not normalize:
                scale = -group["rho"]
            elif norm_mode == "global":
                scale = -group["rho"] / global_norm
            elif norm_mode == "per_param":
                scale = -group["rho"] / direction.norm().clamp_min(eps)
            elif norm_mode == "balanced":
                # each param gets ||eps_p|| = rho * sqrt(numel_p / total_numel):
                # equal per-element RMS everywhere, total norm = rho, and
                # heterogeneous direction families (muon vs adam) can't starve
                # each other the way they do under a raw global norm.
                frac = (direction.numel() / total_numel) ** 0.5
                scale = -group["rho"] * frac / direction.norm().clamp_min(eps)
            else:
                raise ValueError(f"Unsupported perturbation_norm: {norm_mode}")

            perturb = direction * scale
            p["param"].data.add_(perturb)
            p["state"]["perturb"] = perturb

            if stats_enabled:
                psq = perturb.pow(2).sum()
                stats["perturb_sq_total"] += psq
                stats["dot_pg"] += (perturb * p["grad"]).sum()
                stats["grad_sq"] += p["grad"].pow(2).sum()
                family_sq[p["family"]] = family_sq.get(p["family"], 0.0) + psq

        if stats_enabled:
            total = stats["perturb_sq_total"] ** 0.5
            grad_norm = stats["grad_sq"] ** 0.5
            self.last_stats = {
                "perturb_norm": float(total),
                "cos_perturb_grad": float(stats["dot_pg"] / (total * grad_norm + 1e-12)),
                **{f"perturb_norm_{k}": float(v ** 0.5) for k, v in family_sq.items()},
            }

    # ----------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Remove previous perturbation: param.data = w̃ -> w. Exact (cached),
        #    and independent of which params happen to have grads this step.
        self.remove_perturbation()

        items = []
        for group, param in self._iter_params():
            grad = param.grad
            if grad is None:
                continue
            state = self.state[param]
            if "step" not in state:
                self._init_state(param, grad, state, group)

            # Buffers are updated from the perturbed-point gradient grad L(w̃)
            state["step"] += 1
            state["exp_avg"].lerp_(grad, 1 - group["beta1"])
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(group["beta2"]).addcmul_(grad, grad, value=1 - group["beta2"])
            if "exp_avg_ascent" in state:
                state["exp_avg_ascent"].lerp_(grad, 1 - group["ascent_beta1"])

            # 2. Descent from clean w + decoupled weight decay
            descent_lr = self._descent_lr(param, state, group)
            descent_dir = self._descent_direction(param, state, grad, group)
            param.data.add_(descent_dir, alpha=-descent_lr)
            if group["weight_decay"] != 0:
                param.data.mul_(1 - descent_lr * group["weight_decay"])

            # Track ||actual update|| per param for relative perturbation scaling
            if group["perturbation_scale"] == "relative":
                step_norm = (descent_lr * descent_dir.norm()).detach()
                if "step_norm_ema" not in state:
                    state["step_norm_ema"] = step_norm
                else:
                    state["step_norm_ema"] = state["step_norm_ema"].lerp(
                        step_norm, 1 - group["step_norm_beta"]
                    )

            items.append({"group": group, "param": param, "state": state, "grad": grad})

        # 3. Apply new perturbation: param.data = w_new -> w̃_new for next forward
        self._apply_new_perturbations(items)

        return loss
