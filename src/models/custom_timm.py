import torch.nn as nn

from timm import create_model

from .utils import remove_stem_stride


class CustomTimmModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 use_stem_stride: bool = True,
                 **kwargs):
        super().__init__()
        self.model = create_model(model_name, **kwargs)
        if not use_stem_stride:
            remove_stem_stride(model_name, self.model)
        self.activation = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x
