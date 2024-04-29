import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group['weight_decay'] != 0 else 0.0) + scale * p.grad
                p.e_w = e_w
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(p.e_w)  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise Exception('SAM doesn\'t support step(), please use first_step() and second_step().')

    def _grad_norm(self):
        shared_device = None
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                if group['weight_decay'] != 0
                for p in group['params']
                if p.grad is not None
                   and (shared_device := p if shared_device is None else shared_device).device.type == 'cuda'
            ]),
            p=2
        )
        return norm


# # Example of usage:
# model = YourModel()
# base_optimizer = torch.optim.Adam  # use Adam as the base optimizer
# optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=0.1)
#
# # Training loop:
# for inputs, outputs in data_loader:
#     def closure():
#         loss = loss_fn(model(inputs), outputs)
#         loss.backward()
#         return loss
#
#
#     loss = closure()
#     optimizer.first_step(zero_grad=True)
#
#     # Compute the gradient for the second time (after the climb step)
#     closure()
#     optimizer.second_step(zero_grad=True)