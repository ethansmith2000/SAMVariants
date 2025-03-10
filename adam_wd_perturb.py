from typing import Generator

import torch


class AdamWeightDecaySAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        perturb_lr=None,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        exp_avg_momentum=True,

    ):
        perturb_lr = perturb_lr or lr
        defaults = dict(
            lr=lr,
            perturb_lr=perturb_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            exp_avg_momentum=exp_avg_momentum,
        )

        super().__init__(params, defaults)

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

        for group_num, group in enumerate(self.param_groups):
            for i, param in enumerate(group["params"]):
                grad = param.grad
                if grad is None:
                    continue

                # remove last weight decay perturbation, 
                if "step" in self.state[param] and self.state[param]["step"] > 1:
                    param.data.div_(1 - group["lr"] * group["weight_decay"])
                ############################################################

                # do Adam update
                og_shape = grad.shape
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(grad, 1 - group["momentum"])
                else:
                    state["exp_avg"].mul_(group["momentum"]).add_(grad)

                # exp avg sq update
                state["exp_avg_sq"].mul_(group["beta2"]).add_(grad.pow(2))

                # update and weight decay
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(state["exp_avg"], denom, value=-group["lr"])
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                # Do weight decay perturbation
                param.data.mul_(1 - group["lr"] * group["weight_decay"])


