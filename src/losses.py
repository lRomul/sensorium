import abc

import torch
from torch import nn


class MiceLoss(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_mouse_loss(self,
                           output: torch.Tensor,
                           target: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, outputs, targets):
        target_tensors, mice_weights = targets
        loss_value = 0
        for mouse_index, (output_tensor, target_tensor) in enumerate(zip(outputs, target_tensors)):
            mouse_weights = mice_weights[..., mouse_index]
            mask = mouse_weights != 0.0
            if torch.any(mask):
                loss = self.compute_mouse_loss(
                    output_tensor[mask],
                    target_tensor[mask],
                    mouse_weights[mask],
                )
                loss_value += loss
        return loss_value / outputs[0].shape[0]


class MicePoissonLoss(MiceLoss):
    def __init__(self, log_input: bool = False, full: bool = False, eps: float = 1e-8):
        super().__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=full, eps=eps, reduction="none")

    def compute_mouse_loss(self,
                           output: torch.Tensor,
                           target: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        loss = self.poisson(output, target)
        loss *= weights.view(-1, *[1] * (len(loss.shape) - 1))
        return loss.sum()


class MiceCorrelationLoss(MiceLoss):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def compute_mouse_loss(self,
                           output: torch.Tensor,
                           target: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        if torch.any(weights != 1.0):
            weights = weights.view(-1, *[1] * (len(output.shape) - 1))
            output = output * weights
            target = target * weights

        output = torch.transpose(output, 1, 2)
        output = output.reshape(-1, output.shape[-1])
        target = torch.transpose(target, 1, 2)
        target = target.reshape(-1, target.shape[-1])

        delta_output = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_output.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        correlations = (delta_output * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return -correlations.sum()
