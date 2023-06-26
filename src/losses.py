import torch
from torch import nn


class LogMSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(size_average=size_average,
                              reduce=reduce,
                              reduction=reduction)

    def forward(self, inputs, targets):
        return self.mse(inputs, torch.log(targets))
