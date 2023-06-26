from torch import nn

import timm
import argus
from argus.loss import pytorch_losses

from src.responses import Exp
from src.losses import LogMSELoss


class MouseModel(argus.Model):
    nn_module = {
        "timm": timm.create_model,
    }
    loss = {
        **pytorch_losses,
        "log_mse": LogMSELoss,
    }
    prediction_transform = {
        "identity": nn.Identity,
        "exp": Exp,
    }
