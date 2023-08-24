import numpy as np

from src.typing import MouseSample


class Mixup:
    def __init__(self, alpha: float = 0.4, prob: float = 1.0):
        self.alpha = alpha
        self.prob = prob

    def use(self):
        return np.random.random() < self.prob

    def __call__(self, sample1: MouseSample, sample2: MouseSample) -> MouseSample:
        (frames1, behavior1), target1 = sample1
        (frames2, behavior2), target2 = sample2
        lam = np.random.beta(self.alpha, self.alpha)
        frames = (1 - lam) * frames1 + lam * frames2
        behavior = (1 - lam) * behavior1 + lam * behavior2
        target = (1 - lam) * target1 + lam * target2
        return (frames, behavior), target
