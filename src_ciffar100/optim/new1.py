import torch

class CustomAdam():
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)  # 待优化参数列表
        self.lr = lr  # 学习率
        self.beta1 = beta1  # 一阶矩估计的衰减因子
        self.beta2 = beta2  # 二阶矩估计的衰减因子
        self.eps = eps  # 避免除0的小量
        self.t = 0  # 初始化时间步

        # 初始化一阶矩（m）和二阶矩（v）的向量
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        """执行单步优化（参数更新）"""
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue  # 跳过没有梯度的参数

            # 更新一阶矩和二阶矩的估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad.data ** 2)

            # 计算偏差校正后的一阶矩和二阶矩估计
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """清除所有参数的梯度"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()