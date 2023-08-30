import random
from typing import Type

import torch
import torch.nn as nn
import kornia.augmentation as augm


class StackInputsAugmentations(nn.Module):
    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.size = size
        ratio = size[0] / size[1]
        self.frames_transforms = nn.Sequential(
            augm.RandomRotation(degrees=(-10, 10), p=0.5),
            augm.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(ratio - 0.1, ratio + 0.1), p=0.5),
        )

    def forward_behavior(self, behavior):
        c, t, _, _ = behavior.shape
        if random.random() < 0.5:
            behavior = behavior + torch.normal(0., 4., size=(c, 1, 1, 1))
        if random.random() < 0.5:
            behavior = behavior + torch.normal(0., 2., size=(c, t, 1, 1))
        return behavior

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:1] = self.frames_transforms(x[:1])
        y[1:] = self.forward_behavior(x[1:])
        return y


_AUGMENTATIONS_REGISTRY: dict[str, Type[nn.Module]] = dict(
    stack_inputs=StackInputsAugmentations,
)


def get_augmentations(name: str, augmentations_params: dict) -> nn.Module:
    assert name in _AUGMENTATIONS_REGISTRY
    return _AUGMENTATIONS_REGISTRY[name](**augmentations_params)
