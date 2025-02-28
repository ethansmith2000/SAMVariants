from typing import Generator
from utils import zeropower_via_newtonschulz5
import torch

# https://github.com/KellerJordan/Muon/blob/master/muon.py



class AdamMuonSAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        perturb_lr=None,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
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
                    state["exp_avg"].lerp_(grad, 1 - group["momentum"])
                else:
                    state["exp_avg"].mul_(group["momentum"]).add_(grad)

                # exp avg sq update
                state["exp_avg_sq"].mul_(group["beta2"]).add_(grad.pow(2))

                # adam update
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(state["exp_avg"], denom, value=group["perturb_lr"])

                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                # Do Muon perturbation
                # TODO

                # orthogonalization
                g = zeropower_via_newtonschulz5(state["exp_avg"], steps=group["ns_steps"])

                # rescaling
                g *= max(1, g.size(0)/g.size(1))**0.5
                g = g.view(og_shape).type_as(param.data)

                # update and weight decay
                param.data.add_(g, alpha=-group["lr"])


