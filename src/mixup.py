import torch

import numpy as np

SampleType = tuple[torch.Tensor, torch.Tensor]


class Mixup:
    def __init__(self, alpha: float = 0.4, prob: float = 1.0):
        self.alpha = alpha
        self.prob = prob

    def use(self):
        return np.random.random() < self.prob

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        frames1, target1 = sample1
        frames2, target2 = sample2
        lam = np.random.beta(self.alpha, self.alpha)
        frames = (1 - lam) * frames1 + lam * frames2
        target = (1 - lam) * target1 + lam * target2
        return frames, target
