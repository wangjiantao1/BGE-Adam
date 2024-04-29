import torch
import torch.optim as optim

class CustomAdam(optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(CustomAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 自定义超参数的初始值
        self.custom_beta1 = 0.9
        self.custom_beta2 = 0.999
        self.custom_eps = 1e-8

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 获取当前的超参数值
                beta1, beta2 = group['betas']
                eps = group['eps']

                # 更新自定义超参数（这里可以根据具体创新算法进行自适应调整）
                self.custom_beta1 = self.custom_beta1 * 0.9  # 举例：简单的自适应调整

                # 对Adam的标准步骤进行修改
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                beta1_t = beta1 * (1 - self.custom_beta1)  # 自定义的beta1
                beta2_t = beta2 * (1 - self.custom_beta2)  # 自定义的beta2

                state['exp_avg'] = beta1_t * state['exp_avg'] + (1 - beta1_t) * grad
                state['exp_avg_sq'] = beta2_t * state['exp_avg_sq'] + (1 - beta2_t) * (grad ** 2)

                denom = state['exp_avg_sq'].sqrt() + eps
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # step_size = group['lr'] * (bias_correction2.sqrt() / bias_correction1)
                step_size = group['lr'] * (bias_correction2 / bias_correction1)

                # p.data.addcdiv_(-step_size, state['exp_avg'], denom)
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss