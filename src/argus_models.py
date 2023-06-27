from torch import nn

import timm
import argus
from argus.loss import pytorch_losses

from src.responses import Expm1
from src.losses import Log1pMSELoss


class MouseModel(argus.Model):
    nn_module = {
        "timm": timm.create_model,
    }
    loss = {
        **pytorch_losses,
        "log1p_mse": Log1pMSELoss,
    }
    prediction_transform = {
        "identity": nn.Identity,
        "expm1": Expm1,
    }
