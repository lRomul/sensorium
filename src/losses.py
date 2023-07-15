import torch
from torch import nn


class Log1pPoissonLoss(nn.Module):
    def __init__(self, log_input: bool = False, full: bool = False, reduction: str = "mean"):
        super().__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=full, reduction=reduction)

    def forward(self, inputs, targets):
        return self.poisson(inputs, torch.log1p(targets))
