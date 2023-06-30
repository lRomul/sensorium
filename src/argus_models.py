from torch import nn

import argus
from argus.loss import pytorch_losses

from src.responses import Expm1
from src.losses import Log1pMSELoss
from src.models.timm import TimmModel


class MouseModel(argus.Model):
    nn_module = {
        "timm": TimmModel,
    }
    loss = {
        **pytorch_losses,
        "log1p_mse": Log1pMSELoss,
    }
    prediction_transform = {
        "identity": nn.Identity,
        "expm1": Expm1,
    }
