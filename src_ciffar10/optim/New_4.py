import torch
from torch.optim.optimizer import Optimizer
import math


class IntegratedAdam(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 entropy_weight=0.01, amsgrad=False, beta1_max=0.9, beta1_min=0.5, beta2_max=0.999,
                 beta2_min=0.9, loss_threshold=1e-4, entropy_decay=0.95, entropy_boost=1.05, max_entropy_weight=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, alpha=alpha, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        entropy_weight=entropy_weight, beta1_max=beta1_max, beta1_min=beta1_min,
                        beta2_max=beta2_max, beta2_min=beta2_min,
                        loss_threshold=loss_threshold, entropy_decay=entropy_decay,
                        entropy_boost=entropy_boost, max_entropy_weight=max_entropy_weight,
                        loss_prev=None)
        super(IntegratedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(IntegratedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
                    raise RuntimeError('Adam does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all moving averages of squared gradients
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

            if group['loss_prev'] is None:
                group['loss_prev'] = loss
            else:
                if loss is not None and group['loss_prev'] is not None:
                    loss_change = group['loss_prev'] - loss
                    if loss_change < group['loss_threshold']:
                        # Loss did not decrease significantly, boost entropy weight
                        group['entropy_weight'] = min(group['entropy_weight'] * group['entropy_boost'],
                                                      group['max_entropy_weight'])
                    else:
                        # Loss decreased significantly, decay entropy weight
                        group['entropy_weight'] *= group['entropy_decay']
                    group['loss_prev'] = loss

            # Add noise with the updated entropy weight
            for p in group['params']:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.data) * group['entropy_weight']
                p.data.add_(noise)

        return loss