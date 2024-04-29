import torch
from torch.optim.optimizer import Optimizer
import math

# Your original AdamEntropy optimizer here
class AdamEntropy(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, entropy_weight=0.01):
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
                        entropy_weight=entropy_weight)
        super(AdamEntropy, self).__init__(params, defaults)

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

                #方式一、 用于调节学习率的熵调整系数，取熵调整张量的平均值，以此作为数值
                entropy_adjustment = 1 + group['entropy_weight'] * torch.randn_like(p.data).mean()

                # 应用参数更新，这里步长乘以一个标量的熵调整系数
                p.data.addcdiv_(exp_avg, denom, value=-step_size * entropy_adjustment)

        return loss
# ... (all code of AdamEntropy here) ...

# Now, define the SWA version of AdamEntropy
class SWAAdamEntropy(Optimizer):
    def __init__(self, params, lr=1e-3, swa_start=10, swa_freq=5, swa_lr=None, **kwargs):
        if swa_lr is None:
            swa_lr = lr
        if not 0.0 < swa_lr:
            raise ValueError("Invalid SWA learning rate: {:.2f}".format(swa_lr))

        self.optimizer = AdamEntropy(params, lr=lr, **kwargs)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self._optimizer_step_pre_hooks = []
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.defaults = self.optimizer.defaults

        # Initialize the SWA state
        self.swa_state = {'swa_params': [torch.zeros_like(p.data) for p in params],
                          'update_count': 0}

    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        # Update the SWA parameters if needed
        current_step = self.optimizer.state[next(iter(self.optimizer.param_groups))['params'][0]]['step']
        if current_step >= self.swa_start and current_step % self.swa_freq == 0:
            self.update_swa()

        return loss

    def update_swa(self):
        # Increment SWA update count
        self.swa_state['update_count'] += 1

        # Calculate the average of the SWA parameters
        for p_swa, p in zip(self.swa_state['swa_params'], self.optimizer.param_groups[0]['params']):
            if self.swa_state['update_count'] == 1:
                p_swa.data.copy_(p.data)
            else:
                # Update SWA parameters towards the current parameters
                p_swa.data.add_((p.data - p_swa.data) / float(self.swa_state['update_count']))

    def swap_swa_sgd(self):
        ''' Swap the current values of the parameters with the SWA running averages '''
        for p_swa, p in zip(self.swa_state['swa_params'], self.optimizer.param_groups[0]['params']):
            p.data, p_swa.data = p_swa.data, p.data

    def finalize(self):
        """Call this function at the end of training to set each parameter to its SWA value"""
        self.swap_swa_sgd()

# Usage example:
# optimizer = SWAAdamEntropy(model.parameters(), lr=0.001, swa_start=10, swa_freq=5)
# for epoch in range(n_epochs):
#     for input, target in dataloader:
#         # Train your model
#         ...
#         optimizer.step()
# optimizer.finalize()