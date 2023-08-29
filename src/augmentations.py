import torch.nn as nn
import kornia.augmentation as augm


def get_train_augmentations(size: tuple[int, int]) -> nn.Module:
    size = size[::-1]
    ratio = size[0] / size[1]
    transforms = nn.Sequential(
        augm.RandomRotation(degrees=(-10, 10), p=0.5),
        augm.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(ratio - 0.1, ratio + 0.1), p=0.5),
        augm.RandomBrightness(brightness=(0.0, 2.0), clip_output=False, p=0.5),
        augm.RandomGaussianNoise(mean=0.0, std=2.0, p=0.5),
    )
    return transforms
