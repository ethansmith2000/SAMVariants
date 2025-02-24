from typing import Generator

import torch

# https://github.com/KellerJordan/Muon/blob/master/muon.py


# @torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


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
        ns_steps=6,
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
            ns_steps=ns_steps,
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

                # remove last Muon perturbation, 
                #TODO

                ############################################################

                # do ADAM update
                og_shape = grad.shape
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(g, 1 - group["momentum"])
                    g.lerp_(state["exp_avg"], group["momentum"]) if group["nesterov"] else state["exp_avg"]
                else:
                    state["exp_avg"].mul_(group["momentum"]).add_(g)
                    g = g.add(state["exp_avg"], alpha=group["momentum"]) if group["nesterov"] else state["exp_avg"]

                # exp avg sq update
                state["exp_avg_sq"].mul_(group["beta2"]).add_(g.pow(2))

                # adam update
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(g, denom, value=group["perturb_lr"])
                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                # Do Muon perturbation
                # TODO

                # orthogonalization
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # rescaling
                g *= max(1, g.size(0)/g.size(1))**0.5
                g = g.view(og_shape).type_as(param.data)

                # update and weight decay
                param.data.add_(g, alpha=-group["lr"])


