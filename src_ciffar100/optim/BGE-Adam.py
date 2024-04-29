import torch
from torch.optim.optimizer import Optimizer
import math


class IntegratedAdam(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 entropy_weight=0.01, amsgrad=False, beta1_max=0.9, beta1_min=0.5, beta2_max=0.999,
                 beta2_min=0.9):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"Invalid betas value: {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid betas[0] value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas[1] value: {betas[1]}")
        if not 0.0 <= entropy_weight:
            raise ValueError(f"Invalid entropy_weight value: {entropy_weight}")

        defaults = dict(lr=lr, alpha=alpha, betas=betas, eps=eps, weight_decay=weight_decay,
                        entropy_weight=entropy_weight, amsgrad=amsgrad,
                        beta1_max=beta1_max, beta1_min=beta1_min,
                        beta2_max=beta2_max, beta2_min=beta2_min)
        super(IntegratedAdam, self).__init__(params, defaults)
        self.gradient_prediction_model = {}

    def __setstate__(self, state):
        super(IntegratedAdam, self).__setstate__(state)

    @staticmethod
    def compute_gradient_change_rate(grad, prev_grad):
        if prev_grad is None or prev_grad.size() != grad.size():
            return 0
        change_rate = torch.norm(grad - prev_grad) / (torch.norm(prev_grad) + 1e-8)
        return change_rate.item()

    @staticmethod
    def compute_dynamic_beta(change_rate, min_val, max_val):
        beta = min_val + (max_val - min_val) * (1 - change_rate)
        return min(max(beta, min_val), max_val)

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
                    raise RuntimeError('IntegratedAdam does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Compute beta1 and beta2 values dynamically based on gradient change rate
                prev_grad = state['exp_avg'] if 'exp_avg' in state else None
                gradient_change_rate = self.compute_gradient_change_rate(grad, prev_grad)
                beta1 = self.compute_dynamic_beta(gradient_change_rate, group['beta1_min'], group['beta1_max'])
                beta2 = self.compute_dynamic_beta(gradient_change_rate, group['beta2_min'], group['beta2_max'])

                # Update the moving averages of gradient and its square
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Prediction of next gradient (gradient_prediction_model)
                if p in self.gradient_prediction_model:
                    predicted_grad = self.gradient_prediction_model[p].predict(grad)
                else:
                    self.gradient_prediction_model[p] = self.GradientPredictionModel(group['alpha'])
                    predicted_grad = self.gradient_prediction_model[p].predict(grad)

                # Final parameter update with entropy adjustment
                entropy_adjustment = 1 + group['entropy_weight'] * torch.randn_like(p.data).mean()
                p.data.addcdiv_(predicted_grad, denom, value=-step_size * entropy_adjustment)

        return loss

    class GradientPredictionModel:
        def __init__(self, alpha):
            self.alpha = alpha
            self.prev_grad = None

        def predict(self, grad):
            if self.prev_grad is None or self.prev_grad.size() != grad.size():
                self.prev_grad = torch.zeros_like(grad)
            predicted_grad = self.alpha * self.prev_grad + (1 - self.alpha) * grad
            self.prev_grad = predicted_grad.detach()
            return predicted_grad