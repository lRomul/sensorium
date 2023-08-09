from collections import defaultdict
from typing import Union, Tuple

import numpy as np

import torch

from argus.metrics import Metric


def corr(
    y1: np.ndarray, y2: np.ndarray, axis: Union[None, int, Tuple[int]] = -1, eps: float = 1e-8, **kwargs
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
    name: str = "corr"
    better: str = "max"

    def __init__(self):
        super().__init__()
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.weights = defaultdict(list)

    def reset(self):
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.weights = defaultdict(list)

    def update(self, step_output: dict):
        pred_tensors = step_output["prediction"]
        target_tensors, mice_weights = step_output["target"]

        for mouse_index, (pred, target) in enumerate(zip(pred_tensors, target_tensors)):
            mouse_weight = mice_weights[..., mouse_index]
            mask = mouse_weight != 0.0
            if torch.any(mask):
                pred, target = pred[mask], target[mask]

                if len(target.shape) == 3:
                    pred = torch.transpose(pred, 1, 2)
                    pred = pred.reshape(-1, pred.shape[-1])
                    target = torch.transpose(target, 1, 2)
                    target = target.reshape(-1, target.shape[-1])

                self.predictions[mouse_index].append(pred.cpu().numpy())
                self.targets[mouse_index].append(target.cpu().numpy())

    def compute(self):
        mice_corr = dict()
        for mouse_index in self.predictions:
            targets = np.concatenate(self.targets[mouse_index], axis=0)
            predictions = np.concatenate(self.predictions[mouse_index], axis=0)
            mice_corr[mouse_index] = corr(predictions, targets, axis=0).mean()
        return mice_corr

    def epoch_complete(self, state):
        with torch.no_grad():
            mice_corr = self.compute()
        name_prefix = f"{state.phase}_" if state.phase else ''
        for mouse_index, mouse_corr in mice_corr.items():
            state.metrics[name_prefix + self.name + f"_mouse_{mouse_index}"] = mouse_corr
        state.metrics[name_prefix + self.name] = np.mean(list(mice_corr.values()))
