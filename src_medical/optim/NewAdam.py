import torch
from torch.optim.optimizer import Optimizer


class ImprovedAdam(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, alpha=0.5, weight_decay=0):
        if lr is not 0.01 and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, weight_decay=weight_decay)
        super(ImprovedAdam, self).__init__(params, defaults)

    @staticmethod
    def compute_gradient_change_rate(gradient, gradient_history):
        # 示例函数，实际实现应根据具体情况定制
        return torch.mean(gradient / (gradient_history + 1e-8))

    @staticmethod
    def dynamic_beta_adjustment(beta, gradient_change_rate):
        # 这里是动态调整beta值的示例逻辑，需要根据实际情况来设计
        return beta  # 示例代码，实际使用时应根据gradient_change_rate调整beta值

    @staticmethod
    def gradient_prediction(gradient_history, alpha):
        # 这里是梯度预测的示例逻辑，需要根据实际情况来设计
        return gradient_history  # 示例代码，实际使用时应采用更复杂的预测模型

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['gradient_history'] = torch.zeros_like(p.data)

                m, v, gradient_history = state['m'], state['v'], state['gradient_history']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 动态调整beta1和beta2
                gradient_change_rate = self.compute_gradient_change_rate(grad, gradient_history)
                beta1 = self.dynamic_beta_adjustment(beta1, gradient_change_rate)
                beta2 = self.dynamic_beta_adjustment(beta2, gradient_change_rate)

                # 更新历史梯度记录
                gradient_history = gradient_history * beta1 + grad * (1 - beta1)

                # 更新梯度的一阶和二阶矩估计
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差修正后的估计
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                # 梯度预测（这里可以替换为更复杂的模型）
                gradient_predicted = self.gradient_prediction(gradient_history, group['alpha'])

                # 参数更新
                p.data.addcdiv_(-group['lr'], group['alpha'] * m_hat + (1 - group['alpha']) * gradient_predicted,
                                v_hat.sqrt().add(group['eps']))
