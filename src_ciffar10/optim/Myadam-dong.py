import math
import torch
from torch.optim.optimizer import Optimizer


class AdamEntropy(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, entropy_weight=0.01,
                 loss_threshold=1e-4, decay_factor=0.9, growth_factor=1.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= entropy_weight:
            raise ValueError("Invalid entropy weight: {}".format(entropy_weight))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        entropy_weight=entropy_weight, loss_threshold=loss_threshold,
                        decay_factor=decay_factor, growth_factor=growth_factor,
                        last_loss=float('inf'), loss_increase_counter=0)
        super(AdamEntropy, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        # Calculate loss if a closure is provided
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Check for loss improvement and adjust entropy_weight accordingly
            if loss is not None:
                if loss > group['last_loss'] - group['loss_threshold']:
                    group['loss_increase_counter'] += 1
                else:
                    group['loss_increase_counter'] = 0

                # If loss hasn't improved for 2 consecutive epochs, increase entropy_weight
                if group['loss_increase_counter'] > 1:
                    group['entropy_weight'] = min(group['entropy_weight'] * group['growth_factor'], 1.0)
                # If loss improves, decrease the entropy_weight
                else:
                    group['entropy_weight'] *= group['decay_factor']

                group['last_loss'] = loss

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

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
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Apply entropy adjustment to the step size
                step_size *= (1 + group['entropy_weight'] * torch.randn_like(p.data).mean())

                # Apply the parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss