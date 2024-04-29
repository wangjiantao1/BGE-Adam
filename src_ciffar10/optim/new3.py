import math
import torch
from torch.optim.optimizer import Optimizer


class Nadam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("无效的学习率：{}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("无效的参数beta1：{}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("无效的参数beta2：{}".format(beta2))
        if not 0.0 <= eps:
            raise ValueError("无效的参数eps：{}".format(eps))

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """执行一个优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Nadam不支持稀疏梯度')

                state = self.state[p]

                # State初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # 更新偏差校正后的一阶和二阶矩估计
                # m.mul_(beta1).add_(1 - beta1, grad)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                # 计算Nesterov加速的梯度
                m_bar = (1 - beta1) * grad + beta1 * m_hat

                p.data.addcdiv_(-group['lr'], m_bar, (v_hat.sqrt().add_(group['eps'])))

        return loss