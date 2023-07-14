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
        inputs1, target1 = sample1
        inputs2, target2 = sample2
        lam = np.random.beta(self.alpha, self.alpha)
        inputs = (1 - lam) * inputs1 + lam * inputs2
        target = (1 - lam) * target1 + lam * target2
        return inputs, target
