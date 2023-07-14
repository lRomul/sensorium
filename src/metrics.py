from typing import Union, Tuple

import numpy as np
from numpy.typing import ArrayLike

from argus.metrics import Metric


def corr(
    y1: ArrayLike, y2: ArrayLike, axis: Union[None, int, Tuple[int]] = -1, eps: int = 1e-8, **kwargs
) -> np.ndarray:
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).

    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy
                 axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due
                 to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2

    Returns: correlation array
    """

    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


class CorrelationMetric(Metric):
    name: str = "correlation"
    better: str = "max"

    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, step_output: dict):
        prediction = step_output["prediction"].cpu().numpy()
        target = step_output["target"].cpu().numpy()

        self.predictions.append(prediction)
        self.targets.append(target)

    def compute(self):
        targets = np.concatenate(self.targets, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        return corr(predictions, targets, axis=0).mean()
