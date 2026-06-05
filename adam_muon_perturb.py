from typing import Literal
import torch
from utils import zeropower_via_newtonschulz5


def _adam_direction(exp_avg, exp_avg_sq, step, beta1, beta2, eps, bias_correct=True):
    """Compute Adam's update direction (unscaled by lr)."""
    if bias_correct:
        m_hat = exp_avg / (1 - beta1 ** step)
        v_hat = exp_avg_sq / (1 - beta2 ** step)
    else:
        m_hat, v_hat = exp_avg, exp_avg_sq
    return m_hat / (v_hat.sqrt() + eps)


def _muon_direction(exp_avg, grad, beta1, ns_steps, nesterov, rescale=True):
    """Compute Muon's update direction (unscaled by lr)."""
    update = grad.lerp(exp_avg, beta1) if nesterov else exp_avg
    og_shape = update.shape
    if update.ndim != 2:
        update = update.view(update.size(0), -1)
    update = update.clone()
    g = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if rescale:
        g *= max(1, g.size(0) / g.size(1)) ** 0.5
    return g.view(og_shape).type_as(exp_avg)


def _is_muon_eligible(param, group):
    max_dim = group["muon_max_dim"]
    return param.ndim == 2 and (max_dim is None or max(param.shape) <= max_dim)


class HybridSAM(torch.optim.Optimizer):
    """
    SAM-style optimizer where ascent and descent directions are configurable.

    Maintains the perturbed-point invariant: param.data holds w̃ = w + ε at
    forward time, so the gradient PyTorch computes is ∇L(w̃). The step()
    method removes the old perturbation, applies descent from w, then applies
    a fresh perturbation for the next forward pass.

    Shared momentum: exp_avg and exp_avg_sq are updated from the perturbed
    gradient and used by both the ascent (perturbation) and descent paths.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        muon_lr=None,
        rho=1.0,                          # perturbation magnitude (unit-norm scaled)
        ascent: Literal["momentum", "muon", "adam"] = "muon",
        descent: Literal["momentum", "muon", "adam"] = "adam",
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        nesterov=True,
        normalize_perturbation=True,      # True = MSAM-style L2 normalize
        perturbation_norm: Literal["per_param", "global"] = "per_param",
        muon_max_dim=16384,
        muon_fallback_ascent: Literal["skip", "momentum", "adam"] = "skip",
    ):
        muon_lr_mult = 1.0 if muon_lr is None else muon_lr / lr
        defaults = dict(
            lr=lr, muon_lr_mult=muon_lr_mult, rho=rho, ascent=ascent, descent=descent,
            beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, ns_steps=ns_steps,
            nesterov=nesterov, normalize_perturbation=normalize_perturbation,
            perturbation_norm=perturbation_norm,
            muon_max_dim=muon_max_dim,
            muon_fallback_ascent=muon_fallback_ascent,
        )
        super().__init__(params, defaults)

    def _ascent_direction_from_buffers(self, state, group):
        """Direction for the perturbation, reproducible without current grad."""
        return self._buffer_direction(
            group["ascent"], state, group, muon_param=state["param"], for_ascent=True,
        )

    def _buffer_direction(self, mode, state, group, muon_param, for_ascent=False):
        """Direction depending only on optimizer buffers."""
        exp_avg = state["exp_avg"]
        if mode == "momentum":
            return exp_avg
        if mode == "adam":
            return _adam_direction(
                state["exp_avg"], state["exp_avg_sq"], state["step"],
                group["beta1"], group["beta2"], group["eps"],
            )
        if mode == "muon":
            if not _is_muon_eligible(muon_param, group):
                if for_ascent:
                    fallback = group["muon_fallback_ascent"]
                    if fallback == "skip":
                        return None
                    return self._buffer_direction(
                        fallback, state, group, muon_param, for_ascent=True,
                    )
                return self._buffer_direction(
                    "adam", state, group, muon_param, for_ascent=False,
                )
            return _muon_direction(
                exp_avg, exp_avg,
                group["beta1"], group["ns_steps"], nesterov=False,
                rescale=False,
            )
        raise ValueError(f"Unsupported direction mode: {mode}")

    def _descent_direction(self, state, grad, group):
        """Direction for the descent update."""
        if group["descent"] == "momentum":
            return state["exp_avg"]
        if group["descent"] == "muon" and _is_muon_eligible(state["param"], group):
            return _muon_direction(
                state["exp_avg"], grad,
                group["beta1"], group["ns_steps"], group["nesterov"],
            )
        if group["descent"] in {"adam", "muon"}:
            return _adam_direction(
                state["exp_avg"], state["exp_avg_sq"], state["step"],
                group["beta1"], group["beta2"], group["eps"],
            )
        raise ValueError(f"Unsupported descent mode: {group['descent']}")

    def _descent_lr(self, state, group):
        if group["descent"] == "muon" and _is_muon_eligible(state["param"], group):
            return group["lr"] * group["muon_lr_mult"]
        return group["lr"]

    def _apply_perturbations(self, perturbations, sign):
        """Apply ±rho-scaled perturbations with per-param or global norm."""
        if not perturbations:
            return

        if not perturbations[0]["group"]["normalize_perturbation"]:
            for item in perturbations:
                item["param"].data.add_(
                    item["direction"], alpha=sign * item["group"]["rho"],
                )
            return

        norm_mode = perturbations[0]["group"]["perturbation_norm"]
        if norm_mode == "global":
            norm_sq = None
            for item in perturbations:
                part = item["direction"].pow(2).sum()
                norm_sq = part if norm_sq is None else norm_sq + part
            norm = norm_sq.sqrt().clamp_min(perturbations[0]["group"]["eps"])
            for item in perturbations:
                item["param"].data.add_(
                    item["direction"] / norm, alpha=sign * item["group"]["rho"],
                )
        elif norm_mode == "per_param":
            for item in perturbations:
                norm = item["direction"].norm().clamp_min(item["group"]["eps"])
                item["param"].data.add_(
                    item["direction"] / norm, alpha=sign * item["group"]["rho"],
                )
        else:
            raise ValueError(f"Unsupported perturbation_norm: {norm_mode}")

    def _collect_ascent_perturbations(self, items):
        perturbations = []
        for item in items:
            direction = self._ascent_direction_from_buffers(item["state"], item["group"])
            if direction is not None:
                perturbations.append({**item, "direction": direction})
        return perturbations

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active_items = []
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]

                # Initialize state on first step
                if "step" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0
                state["param"] = param
                active_items.append(
                    {"group": group, "param": param, "grad": grad, "state": state}
                )

        # 1. Remove previous perturbation: param.data = w̃ → w
        prior_items = [item for item in active_items if item["state"]["step"] > 0]
        self._apply_perturbations(
            self._collect_ascent_perturbations(prior_items), sign=+1.0,
        )

        for item in active_items:
            group = item["group"]
            param = item["param"]
            grad = item["grad"]
            state = item["state"]
            # Gradient was computed at w̃ (the perturbed weights from last step)
            # Update momentum buffers using this perturbed-point gradient
            state["step"] += 1
            state["exp_avg"].lerp_(grad, 1 - group["beta1"])
            state["exp_avg_sq"].mul_(group["beta2"]).addcmul_(grad, grad, value=1 - group["beta2"])

            # 2. Descent from unperturbed w
            descent_dir = self._descent_direction(state, grad, group)
            descent_lr = self._descent_lr(state, group)
            param.data.add_(descent_dir, alpha=-descent_lr)

            # Decoupled weight decay (AdamW-style)
            if group["weight_decay"] != 0:
                param.data.mul_(1 - descent_lr * group["weight_decay"])

        # 3. Apply new perturbation: param.data = w_new → w̃_new for next forward
        self._apply_perturbations(
            self._collect_ascent_perturbations(active_items), sign=-1.0,
        )

        return loss