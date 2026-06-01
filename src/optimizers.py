import math
import torch
from torch.optim import Optimizer

try:
    from galore_torch import GaLoreAdamW as _GaLoreAdamW

    class GaLoreAdamW(_GaLoreAdamW):
        pass

except ImportError:
    GaLoreAdamW = None


class StaceyBase(Optimizer):
    """Local vendored base class from xinyuluo8561/Stacey.

    The upstream files use absolute sibling imports, which break inside Ray's
    packaged working directories. Keeping the implementation local removes that
    runtime dependency.
    """

    def __init__(
        self,
        params,
        lr_tau,
        lr_eta,
        lr_alpha,
        lr_beta1,
        lr_beta2,
        momentum,
        weight_decay,
        dampening,
        q,
        eps,
        debug,
    ):
        if not 0.0 <= lr_tau:
            raise ValueError(f"Invalid learning rate Tau: {lr_tau}")
        if not 0.0 <= lr_eta:
            raise ValueError(f"Invalid learning rate Eta: {lr_eta}")
        if not 0.0 <= lr_alpha:
            raise ValueError(f"Invalid learning rate Alpha: {lr_alpha}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr_tau=lr_tau,
            lr_eta=lr_eta,
            lr_alpha=lr_alpha,
            lr_beta1=lr_beta1,
            lr_beta2=lr_beta2,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            q=q,
            eps=eps,
            debug=debug,
        )

        params = list(params)
        for p in params:
            p.z = p.data.clone()
            p.m = torch.zeros_like(p.data)

        super().__init__(params, defaults)


class Stacey_pp(StaceyBase):
    """Local Stacey++ optimizer implementation.

    Vendored from https://github.com/xinyuluo8561/Stacey/blob/main/Stacey/staceypp.py
    with the sibling import removed.
    """

    def __init__(
        self,
        params,
        lr_tau=0.001,
        lr_eta=0.001,
        lr_alpha=0.001,
        lr_beta1=0.9,
        lr_beta2=0.999,
        momentum=0,
        eps=1e-8,
        weight_decay=5e-4,
        dampening=0,
        q=3,
        debug=False,
    ):
        super().__init__(
            params,
            lr_tau,
            lr_eta,
            lr_alpha,
            lr_beta1,
            lr_beta2,
            momentum,
            weight_decay,
            dampening,
            q,
            eps,
            debug,
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_track = dict(
            z_dist=0.0,
            y_dist=0.0,
            update_dist=0.0,
            grad_dist=0.0,
            momentum_dist=0.0,
            pdata_dist=0.0,
            debug=False,
        )

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr_beta1 = group["lr_beta1"]
            lr_beta2 = group["lr_beta2"]
            lr_tau = group["lr_tau"]
            lr_eta = group["lr_eta"]
            lr_alpha = group["lr_alpha"]
            q = group["q"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                c = torch.add(lr_beta1 * p.m, p.grad.data, alpha=1 - lr_beta1)

                z_tmp = torch.mul(torch.pow(torch.abs(p.z), q - 2), p.z)
                z_tmp.add_(c, alpha=-lr_alpha)
                p.z = torch.mul(torch.sign(z_tmp), torch.pow(torch.abs(z_tmp), 1 / (q - 1)))
                del z_tmp

                c_tmp = torch.mul(c, torch.pow(torch.abs(c) + eps, (2 - q) / (q - 1)))
                p.data.mul_(1 - lr_tau - lr_eta * weight_decay).add_(p.z, alpha=lr_tau).add_(
                    c_tmp, alpha=lr_eta * (lr_tau - 1)
                )

                p.m.mul_(lr_beta2).add_(p.grad, alpha=1 - lr_beta2)

                del c_tmp, c

        if params_track["debug"]:
            params_track["grad_dist"] = torch.sqrt(params_track["grad_dist"])
            params_track["pdata_dist"] = torch.sqrt(params_track["pdata_dist"])
            params_track["update_dist"] = torch.sqrt(params_track["update_dist"])
            params_track["y_dist"] = torch.sqrt(params_track["y_dist"])
            params_track["z_dist"] = torch.sqrt(params_track["z_dist"])
            params_track["momentum_dist"] = torch.sqrt(params_track["momentum_dist"])

        return loss if loss is not None else params_track


class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        final_lr=0.1,
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0,
        amsbound=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group["lr"], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = 0
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    # grad = grad.add(group['weight_decay'], p.data)
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss
