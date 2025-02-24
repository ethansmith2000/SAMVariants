from typing import Generator

import torch


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        perturb_lr=None,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        nesterov=False,
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
            nesterov=nesterov,
            exp_avg_momentum=exp_avg_momentum,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, param, closure=None):
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
                param.data.div_(1 - group["lr"] * group["weight_decay"])
                ############################################################

                # do Muon update
                og_shape = grad.shape
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(g, 1 - group["momentum"])
                    g.lerp_(state["exp_avg"], group["momentum"]) if group["nesterov"] else state["exp_avg"]
                else:
                    state["exp_avg"].mul_(group["momentum"]).add_(g)
                    g = g.add(state["exp_avg"], alpha=group["momentum"]) if group["nesterov"] else state["exp_avg"]

                # exp avg sq update
                state["exp_avg_sq"].mul_(group["beta2"]).add_(g.pow(2))

                # update and weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])
                param.data.addcdiv_(state["exp_avg"], denom, value=-group["lr"])

                ############################################################

                # Do weight decay perturbation
                param.data.mul_(1 - group["lr"] * group["weight_decay"])


