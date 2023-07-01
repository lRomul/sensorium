from torch import nn as nn


def remove_stem_stride(model_name: str, model: nn.Module):
    if "efficientnet" in model_name:
        model.conv_stem.stride = (1, 1)
        model.conv_stem.padding = (1, 1)
    elif "regnet" in model_name:
        model.stem.conv.stride = (1, 1)
    else:
        raise ValueError(f"Removing stem stride is not supported for '{model_name}'")
