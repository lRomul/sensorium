import abc
import math
import random

import numpy as np

import torch

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


def fftfreqnd(h, w=None, z=None):
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, h, w=0, z=0):
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay, shape):
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, *shape)
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:shape[0]]
    if len(shape) == 2:
        mask = mask[:shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def binarize_mask(mask, lam, in_shape, max_soft=0.0):
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape(*in_shape)
    return mask


def sample_mask(alpha, decay_power, shape, max_soft=0.0):
    if isinstance(shape, int):
        shape = (shape,)
    lam = np.random.beta(alpha, alpha)
    mask = make_low_freq_image(decay_power, shape)
    mask = binarize_mask(mask, lam, shape, max_soft)
    return lam, mask


class FMix(Mixer):
    def __init__(self,
                 decay_power: float = 3.,
                 alpha: float = 1.,
                 size: tuple[int, ...] = (64, 64),
                 max_soft: float = 0.0,
                 prob: float = 1.0):
        super().__init__(prob)
        self.decay_power = decay_power
        self.alpha = alpha
        self.size = size
        self.max_soft = max_soft

    def __call__(self, sample1: SampleType, sample2: SampleType) -> SampleType:
        inputs1, target1 = sample1
        inputs2, target2 = sample2
        lam, mask = sample_mask(self.alpha, self.decay_power,
                                self.size, max_soft=self.max_soft)
        mask = torch.from_numpy(mask).to(dtype=inputs1.dtype, device=inputs1.device)
        mask = mask.expand_as(inputs1)
        inputs = (1 - mask) * inputs1 + mask * inputs2
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
