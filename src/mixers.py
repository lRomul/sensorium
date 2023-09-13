import abc

import torch

import numpy as np

SampleType = tuple[torch.Tensor, torch.Tensor]


class Mixer(metaclass=abc.ABCMeta):
    def __init__(self, prob: float):
        self.prob = prob

    def use(self):
        return np.random.random() < self.prob

    @abc.abstractmethod
    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        pass


class Mixup(Mixer):
    def __init__(self, alpha: float = 0.4, prob: float = 1.0):
        super().__init__(prob)
        self.alpha = alpha

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        inputs1, target1 = sample1
        inputs2, target2 = sample2
        lam = np.random.beta(self.alpha, self.alpha)
        inputs = (1 - lam) * inputs1 + lam * inputs2
        target = (1 - lam) * target1 + lam * target2
        return inputs, target


def rand_bbox(height: int, width: int, lam: float):
    cut_rat = np.sqrt(lam)
    cut_w = (width * cut_rat).astype(int)
    cut_h = (height * cut_rat).astype(int)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


class CutMix(Mixer):
    def __init__(self, alpha: float = 1.0, prob: float = 1.0):
        super().__init__(prob)
        self.alpha = alpha

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        inputs1, target1 = sample1
        inputs2, target2 = sample2
        inputs = inputs1.clone().detach()
        lam = np.random.beta(self.alpha, self.alpha)
        h, w = inputs1.shape[-2:]
        bbx1, bby1, bbx2, bby2 = rand_bbox(h, w, lam)
        inputs[..., bbx1: bbx2, bby1: bby2] = inputs2[..., bbx1: bbx2, bby1: bby2]
        lam = (bbx2 - bbx1) * (bby2 - bby1) / (h * w)
        target = (1 - lam) * target1 + lam * target2
        return inputs, target


class RandomChoiceMixer(Mixer):
    def __init__(self, mixers: list[Mixer], choice_probs: list[float], prob: float = 1.0):
        super().__init__(prob)
        self.mixers = mixers
        self.choice_probs = choice_probs

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        mixer_index = np.random.choice(range(len(self.mixers)), p=self.choice_probs)
        mixer = self.mixers[mixer_index]
        return mixer(sample1, sample2)
