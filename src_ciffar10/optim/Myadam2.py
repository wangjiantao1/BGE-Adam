import math
import torch
from torch.optim.optimizer import Optimizer


class LogAdam(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, log_scale=0.1, entropy_weight=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= log_scale:
            raise ValueError("Invalid log scale: {}".format(log_scale))
        if not 0.0 <= entropy_weight:
            raise ValueError("Invalid entropy weight: {}".format(entropy_weight))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        log_scale=log_scale,
                        entropy_weight=entropy_weight)
        super(LogAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LogAdam does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Adjust the step size by a log scale factor
                adjusted_step_size = step_size * math.log(1 + abs(grad.mean()) * group['log_scale'])
                entropy_adjustment = 1 + group['entropy_weight'] * torch.randn_like(p.data).mean()

                # Apply the parameter update
                p.data.addcdiv_(exp_avg, denom, value=-adjusted_step_size)
                p.data.addcdiv_(exp_avg, denom, value=-step_size * entropy_adjustment)

        return loss
