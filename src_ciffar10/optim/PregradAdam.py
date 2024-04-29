import torch
from torch.optim.optimizer import Optimizer
from math import sqrt


class ImprovedAdam(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 beta1_max=0.9, beta1_min=0.5, beta2_max=0.999, beta2_min=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < beta1_max <= 1.0:
            raise ValueError(f"Invalid beta1_max value: {beta1_max}")
        if not 0.0 <= beta1_min < 1.0:
            raise ValueError(f"Invalid beta1_min value: {beta1_min}")
        if not 0.0 < beta2_max <= 1.0:
            raise ValueError(f"Invalid beta2_max value: {beta2_max}")
        if not 0.0 <= beta2_min < 1.0:
            raise ValueError(f"Invalid beta2_min value: {beta2_min}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"Invalid betas value: {betas}")

        defaults = dict(lr=lr, alpha=alpha, betas=betas, eps=eps, weight_decay=weight_decay,
                        beta1_max=beta1_max, beta1_min=beta1_min,
                        beta2_max=beta2_max, beta2_min=beta2_min)
        self.gradient_prediction_model = self.GradientPredictionModel(alpha)
        super(ImprovedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ImprovedAdam, self).__setstate__(state)

    @staticmethod
    def compute_gradient_change_rate(grad, prev_grad):
        if prev_grad is None:
            return 0
        change_rate = torch.norm(grad - prev_grad) / (torch.norm(prev_grad) + 1e-8)
        return change_rate.item()

    class GradientPredictionModel:
        def __init__(self, alpha):
            self.alpha = alpha
            self.prev_grad = None

        def predict(self, grad):
            if self.prev_grad is None or self.prev_grad.size() != grad.size():
                # Ensure prev_grad has the same shape as the incoming grad.
                # Initialize to zeros to avoid interfering with the initial update.
                self.prev_grad = torch.zeros_like(grad)
            predicted_grad = self.alpha * self.prev_grad + (1 - self.alpha) * grad
            self.prev_grad = predicted_grad.detach()  # Detach to prevent history tracking
            return predicted_grad

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
                    raise RuntimeError("ImprovedAdam does not support sparse gradients")
                # Make sure to use grad's device for the calculation and storage
                state = self.state[p]  # 将 state 的赋值调整到这里

                # 这里已经有state，所以接下来可以安全地检查 prev_grad
                if len(state) == 0:  # 确保在使用这些值之前，状态已经被初始化
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(grad)  # 请确保这个初始化在此行之后

                    # 此时应该已经初始化了 state['prev_grad']，可以安全地进行检查和更新
                if 'prev_grad' not in state or state['prev_grad'].size() != grad.size() or state[
                    'prev_grad'].device != grad.device:
                    state['prev_grad'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the gradient change rate and dynamically adjust beta1 and beta2
                gradient_change_rate = self.compute_gradient_change_rate(grad, state.get('prev_grad'))
                state['prev_grad'] = grad.clone()

                # Dynamically adjust beta1 and beta2 using the computed gradient change rate
                adj_beta1 = self.compute_dynamic_beta(gradient_change_rate, group['beta1_min'], group['beta1_max'])
                adj_beta2 = self.compute_dynamic_beta(gradient_change_rate, group['beta2_min'], group['beta2_max'])

                # Bias-corrected first and second moment estimate
                m_hat = exp_avg / (1 - adj_beta1 ** state['step'])
                v_hat = exp_avg_sq / (1 - adj_beta2 ** state['step'])

                grad_pred = self.gradient_prediction_model.predict(m_hat)

                update = (group['alpha'] * m_hat + (1 - group['alpha']) * grad_pred) / (v_hat.sqrt() + group['eps'])

                # p.data.add_(-group['lr'], update)
                p.data.add_(update, alpha=-group['lr'])

        return loss

    @staticmethod
    def compute_dynamic_beta(change_rate, min_val, max_val):
        # An example of dynamic adjustment based on the change rate
        # Using a simple capped linear scale for demo purpose
        beta = min_val + (max_val - min_val) * (1 - change_rate)
        return min(max(beta, min_val), max_val)