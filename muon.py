import torch

try:
    from .utils import zeropower_via_newtonschulz5
except ImportError:
    from utils import zeropower_via_newtonschulz5

# https://github.com/KellerJordan/Muon/blob/master/muon.py



class Muon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        muon_lr=None,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        exp_avg_momentum=True,
        nesterov=False,
        muon_max_dim=16384,
    ):
        muon_lr_mult = 1.0 if muon_lr is None else muon_lr / lr
        defaults = dict(
            lr=lr,
            muon_lr_mult=muon_lr_mult,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            exp_avg_momentum=exp_avg_momentum,
            nesterov=nesterov,
            muon_max_dim=muon_max_dim,
        )

        super().__init__(params, defaults)

    def _use_muon(self, param, group):
        max_dim = group["muon_max_dim"]
        return param.ndim == 2 and (max_dim is None or max(param.shape) <= max_dim)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(grad, 1 - group["beta1"])
                else:
                    state["exp_avg"].mul_(group["beta1"]).add_(grad)

                if self._use_muon(param, group):
                    update = (
                        grad.lerp(state["exp_avg"], group["beta1"])
                        if group["nesterov"]
                        else state["exp_avg"]
                    )

                    # orthogonalization
                    g = zeropower_via_newtonschulz5(update.clone(), steps=group["ns_steps"])

                    # rescaling
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    update_dir = g.type_as(param.data)
                else:
                    state["exp_avg_sq"].mul_(group["beta2"]).addcmul_(
                        grad, grad, value=1 - group["beta2"],
                    )
                    bias_correction1 = 1 - group["beta1"] ** state["step"]
                    bias_correction2 = 1 - group["beta2"] ** state["step"]
                    exp_avg = state["exp_avg"] / bias_correction1
                    exp_avg_sq = state["exp_avg_sq"] / bias_correction2
                    update_dir = exp_avg / (exp_avg_sq.sqrt() + group["eps"])

                # update and weight decay
                step_lr = (
                    group["lr"] * group["muon_lr_mult"]
                    if self._use_muon(param, group)
                    else group["lr"]
                )
                param.data.add_(update_dir, alpha=-step_lr)
                if group["weight_decay"] != 0:
                    param.data.mul_(1 - step_lr * group["weight_decay"])



