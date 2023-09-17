import torch
from torch import nn


class MicePoissonLoss(nn.Module):
    def __init__(self,
                 max_loss_ratio: float = 0.0,
                 log_input: bool = False,
                 full: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.max_loss_ratio = max_loss_ratio
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=full, eps=eps, reduction="none")

    def weighted_sum_poisson(self, inputs, targets, weights):
        loss = self.poisson(inputs, targets)
        loss *= weights.view(-1, *[1] * (len(loss.shape) - 1))
        return loss.sum()

    def forward(self, inputs, targets):
        target_tensors, mice_weights = targets
        mice_weights = mice_weights / mice_weights.sum()
        loss_value = 0.0
        for mouse_index, (input_tensor, target_tensor) in enumerate(zip(inputs, target_tensors)):
            mouse_weights = mice_weights[..., mouse_index]
            mask = mouse_weights != 0.0
            if torch.any(mask):
                masked_input = input_tensor[mask]
                masked_target = target_tensor[mask]
                loss_value += (1.0 - self.max_loss_ratio) * self.weighted_sum_poisson(
                    masked_input, masked_target, mouse_weights[mask]
                )
                if self.max_loss_ratio:
                    loss_value += (self.max_loss_ratio * mask.shape[0]) * self.weighted_sum_poisson(
                        masked_input.max(dim=-1)[0], masked_target.max(dim=-1)[0], mouse_weights[mask]
                    )
        return loss_value
