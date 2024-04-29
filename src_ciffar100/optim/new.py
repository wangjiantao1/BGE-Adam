import torch

class CustomAdadelta():
    def __init__(self, params, rho=0.9, eps=1e-6):
        self.params = list(params)
        self.rho = rho
        self.eps = eps
        # 用于存储累积梯度平方和累积参数更新量平方的状态
        self.acc_grad_sqrs = [torch.zeros_like(p) for p in self.params]
        self.acc_deltas = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """执行Adadelta的参数更新"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue  # 如果某个参数没有梯度，则跳过

            grad = param.grad.data
            acc_grad_sqr = self.acc_grad_sqrs[i]
            acc_delta = self.acc_deltas[i]

            # 更新累积梯度平方
            acc_grad_sqr.mul_(self.rho).addcmul_(grad, grad, value=(1 - self.rho))

            # 计算参数更新量
            std = grad.new_tensor(self.eps).sqrt_() + acc_grad_sqr.sqrt_()
            delta = acc_delta.sqrt().div_(std).mul_(grad)

            # 更新参数
            param.data.add_(-delta)

            # 更新累积参数更新量平方
            acc_delta.mul_(self.rho).addcmul_(delta, delta, value=(1 - self.rho))

    def zero_grad(self):
        """清除所有参数的梯度，为下一轮优化准备"""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()