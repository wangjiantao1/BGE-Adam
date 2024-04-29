import torch
import torch.nn as nn
import torch.optim as optim


class CustomAdam(optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-5, weight_decay=0, nu=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, nu=nu)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # 防止除以0的稳健方法
                eps_stabilized = group['eps'] + 1e-8  # 增加一个非常小的常数以提高稳定性

                # 计算 hessian_inv_estimate 更稳健的方法, 保护除法操作
                hessian_inv_estimate = torch.sqrt(torch.tensor(eps_stabilized, dtype=torch.float)) + eps_stabilized

                # 执行梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(p, max_norm=1)

                # 对 grad 进行规模调整
                scaled_grad = grad / hessian_inv_estimate

                # 梯度的L2范数不应为0，避免除数接近0
                grad_norm = torch.norm(grad)
                if grad_norm == 0:
                    continue  # 如果梯度为0，跳过此参数的更新

                # 计算系统熵相关项，引入稳健性
                entropy_term = group['nu'] / (1.0 + torch.exp(-grad_norm + 1e-8))

                state = self.state[p]

                if 'moment1' not in state:
                    state['moment1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['moment2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m1, m2 = state['moment1'], state['moment2']

                m1.data = group['betas'][0] * m1 + (1 - group['betas'][0]) * scaled_grad
                m2.data = group['betas'][1] * m2 + (1 - group['betas'][1]) * scaled_grad ** 2

                # 参数更新
                p.data.add_(-group['lr'] * (m1 / hessian_inv_estimate + group['weight_decay'] * p.data + entropy_term))

        return loss