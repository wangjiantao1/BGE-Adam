import math
import torch
from torch.optim.optimizer import Optimizer

class AndEntropy(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 amsgrad=False, beta1_max=0.9, beta1_min=0.5, beta2_max=0.999, beta2_min=0.5,
                 entropy_weight=0.01):
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
        if not 0.0 <= entropy_weight:
            raise ValueError(f"Invalid entropy weight: {entropy_weight}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"Invalid betas value: {betas}")
        defaults = dict(lr=lr, alpha=alpha, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, beta1_max=beta1_max, beta1_min=beta1_min,
                        beta2_max=beta2_max, beta2_min=beta2_min, entropy_weight=entropy_weight)
        super(AndEntropy, self).__init__(params, defaults)
        self.gradient_prediction_model = self.init_gradient_prediction_model(params, alpha)

    def init_gradient_prediction_model(self, params, alpha):
        model = {}
        for p in params:
            model[p] = torch.zeros_like(p.data)
        return model

    def compute_gradient_change_rate(self, grad, prev_grad):
        change_rate = torch.norm(grad - prev_grad) / (torch.norm(prev_grad) + 1e-8)
        return change_rate.item()

    def compute_dynamic_beta(self, change_rate, min_val, max_val):
        beta = min_val + (max_val - min_val) * (1 - change_rate)
        return min(max(beta, min_val), max_val)

    def compute_entropy_adjustment(self, entropy_weight, tensor_shape, device):
        return 1 + entropy_weight * torch.randn(tensor_shape, device=device).mean()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            entropy_weight = group['entropy_weight']
            alpha = group['alpha']
            beta1_max = group['beta1_max']
            beta1_min = group['beta1_min']
            beta2_max = group['beta2_max']
            beta2_min = group['beta2_min']
            amsgrad = group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                gradient_change_rate = self.compute_gradient_change_rate(grad, state['prev_grad'])
                adj_beta1 = self.compute_dynamic_beta(gradient_change_rate, beta1_min, beta1_max)
                adj_beta2 = self.compute_dynamic_beta(gradient_change_rate, beta2_min, beta2_max)

                bias_correction1 = 1 - adj_beta1 ** state['step']
                bias_correction2 = 1 - adj_beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Gradient prediction
                if p in self.gradient_prediction_model:
                    grad_pred = alpha * self.gradient_prediction_model[p] + (1 - alpha) * grad
                    self.gradient_prediction_model[p] = grad_pred.detach()

                # Entropy adjustment
                entropy_adjustment = self.compute_entropy_adjustment(entropy_weight, p.data.size(), p.device)

                p.data.addcdiv_(exp_avg, denom, value=- step_size * entropy_adjustment)
                p.data.add_(grad_pred, alpha= -group['lr'] * (1 - alpha))

                # Update prev_grad for the next step
                state['prev_grad'] = grad.clone()
        return loss

