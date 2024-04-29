import torch
import torch.nn as nn
import torch.optim as optim

class CustomAdam(optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-5, weight_decay=0, nu=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, nu=nu)
        super(CustomAdam, self).__init__(params, defaults)

        self.t = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.t += 1
        # clip_value = 1e-5  # 举例裁剪值
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # 拟牛顿法的思想，将梯度转换为对应的Hessian矩阵的逆的估计
                # 修改之处: 使用group['eps']取代self.eps
                hessian_inv_estimate = torch.sqrt(torch.tensor(group['eps'], dtype=torch.float)) + group['eps']
                scaled_grad = grad / hessian_inv_estimate

                # 系统熵相关联的额外项
                entropy_term = group['nu'] / (1.0 + torch.exp(-torch.norm(grad)))

                state = self.state[p]

                if 'moment1' not in state:
                    state['moment1'] = torch.zeros_like(p, requires_grad=False)
                    state['moment2'] = torch.zeros_like(p, requires_grad=False)

                m1, m2 = state['moment1'], state['moment2']

                m1.data = group['betas'][0] * m1 + (1 - group['betas'][0]) * scaled_grad
                m2.data = group['betas'][1] * m2 + (1 - group['betas'][1]) * scaled_grad**2

                # 参数更新
                p.data.add_(-group['lr'] * (m1 / hessian_inv_estimate + group['weight_decay'] * p.data + entropy_term))

        return loss