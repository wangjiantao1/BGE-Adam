import torch
from torch.optim.optimizer import Optimizer

class AdamAdamax(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr is not 0.001 and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamAdamax, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamAdamax does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Moving max of absolute gradients (for Adamax)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_inf = state['exp_avg'], state['exp_avg_sq'], state['exp_inf']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate.
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Update the moving max of absolute gradients (Adamax update)
                exp_inf.mul_(beta1).add_(torch.max(exp_inf, torch.abs(grad)), alpha=1 - beta1)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                corrected_exp_avg = exp_avg / bias_correction1
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** state['step']
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                # Compute the max term for Adamax
                denom = exp_inf / bias_correction1

                # Update parameters
                step_size = group['lr']
                p.data.addcdiv_(corrected_exp_avg, torch.max(corrected_exp_avg_sq.sqrt(), denom) + group['eps'], value=-step_size)

        return loss
