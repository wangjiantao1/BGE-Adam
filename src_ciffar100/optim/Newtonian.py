import torch
from torch.optim.optimizer import Optimizer


class AdamQN(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamQN, self).__init__(params, defaults)

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
                    raise RuntimeError('AdamQN does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # RMSprop
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute biased-corrected first and second moment estimates
                exp_avg_hat = exp_avg / (1 - beta1 ** state['step'])
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** state['step'])

                # Compute update, here's where the quasi Newtonian idea comes in
                denom = exp_avg_sq_hat.sqrt().add_(group['eps'])
                step_size = group['lr'] * exp_avg_hat / denom

                # Attempting to mimic second-order behaviour by modulating the
                # step size with the change of gradient (not exactly quasi-Newton, but inspired)
                if 'prev_grad' in state:
                    grad_change = grad - state['prev_grad']
                    step_adjustment = (grad_change.sign() * 0.5 + 1).clamp_(min=0.1, max=5)
                    step_size.mul_(step_adjustment)
                state['prev_grad'] = grad.clone()

                p.data.add_(-step_size)

        return loss