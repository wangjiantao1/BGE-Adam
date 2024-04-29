import torch
from torch.optim import Optimizer


def line_search(model, loss, p, init_step, grad):
    step = init_step
    while step > 1e-5:
        p.data.add_(-step, grad)
        loss_step = model(p)
        if torch.isnan(loss_step) or loss_step > loss:
            step /= 2.0
            p.data.add_(step, grad)
        else:
            break
    return loss_step


class lineSearchAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(lineSearchAdam, self).__init__(params, defaults)

    # def step(self, closure):
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             dp = p.grad.data
    #             dp_prev = self.state.get('prev_grad')
    #             dp_prev = dp_prev if dp_prev is not None else torch.zeros_like(dp)
    #
    #             loss_prev = closure()
    #
    #             step = group['lr'] * torch.norm(dp_prev) / (torch.norm(dp) + group['eps'])
    #
    #             # line search step to find the optimal step size
    #             loss_prev = line_search(p, loss_prev, p, step, dp)
    #             self.state['prev_grad'] = dp
    #
    #     return loss_prev

    def step(self, closure):
        loss_prev = None  # Add this line
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad.data
                dp_prev = self.state.get('prev_grad')
                dp_prev = dp_prev if dp_prev is not None else torch.zeros_like(dp)

                loss_prev = closure()

                step = group['lr'] * torch.norm(dp_prev) / (torch.norm(dp) + group['eps'])

                # line search step to find the optimal step size
                loss_prev = line_search(p, loss_prev, p, step, dp)
                self.state['prev_grad'] = dp

        return loss_prev