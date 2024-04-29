import math
import torch
from torch.optim.optimizer import Optimizer, required


class AdvancedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, adjust_lr=True, improvement_threshold=0.995):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, adjust_lr=adjust_lr, improvement_threshold=improvement_threshold)
        super(AdvancedAdam, self).__init__(params, defaults)

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
                    raise RuntimeError('AdvancedAdam does not support sparse gradients')

                amsgrad = group['amsgrad']
                adjust_lr = group['adjust_lr']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    # For auto-adjust learning rate
                    state['prev_loss'] = None
                    state['lr'] = group['lr']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = state['lr'] if adjust_lr else group['lr']

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Auto-adjust learning rate logic
                if adjust_lr and loss is not None:
                    if state['prev_loss'] is not None and loss > state['prev_loss'] * group['improvement_threshold']:
                        state['lr'] *= 0.5
                    state['prev_loss'] = loss

        return loss
