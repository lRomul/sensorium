import torch
from torch import nn


class MicePoissonLoss(nn.Module):
    def __init__(self, log_input: bool = False, full: bool = False, eps: float = 1e-8):
        super().__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=full, eps=eps, reduction="none")

    def forward(self, inputs, targets):
        target_tensors, mice_weights = targets
        mice_weights = mice_weights / mice_weights.sum()
        loss_value = 0
        for mouse_index, (input_tensor, target_tensor) in enumerate(zip(inputs, target_tensors)):
            mouse_weights = mice_weights[..., mouse_index]
            mask = mouse_weights != 0.0
            if torch.any(mask):
                loss = self.poisson(input_tensor[mask], target_tensor[mask])
                loss *= mouse_weights[mask].view(-1, *[1] * (len(loss.shape) - 1))
                loss_value += loss.sum()
        return loss_value
