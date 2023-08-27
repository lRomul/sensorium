import torch

import numpy as np


class Mixup:
    def __init__(self, alpha: float = 0.4, prob: float = 1.0):
        self.alpha = alpha
        self.prob = prob

    def use(self):
        return np.random.random() < self.prob

    def sample_lam(self):
        lam = np.random.beta(self.alpha, self.alpha)
        if lam > 0.5:
            lam = 1 - lam
        return lam

    def __call__(self,
                 sample1: tuple[torch.Tensor, torch.Tensor],
                 sample2: tuple[torch.Tensor, torch.Tensor],
                 lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        inputs1, target1 = sample1
        inputs2, target2 = sample2
        if lam is None:
            lam = self.sample_lam()
        inputs = (1 - lam) * inputs1 + lam * inputs2
        target = (1 - lam) * target1 + lam * target2
        return inputs, target
