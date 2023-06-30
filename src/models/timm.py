import torch.nn as nn

from timm import create_model


class TimmModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 use_stem_stride: bool = True,
                 **kwargs):
        super().__init__()
        self.model = create_model(model_name, **kwargs)
        if not use_stem_stride:
            self.remove_stem_stride(model_name)

    def remove_stem_stride(self, model_name: str):
        if "efficientnet" in model_name:
            self.model.conv_stem.stride = (1, 1)
            self.model.conv_stem.padding = (1, 1)
        else:
            raise ValueError(f"Removing stem stride is not supported for '{model_name}'")

    def forward(self, x):
        x = self.model(x)
        return x
