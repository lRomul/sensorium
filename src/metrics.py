import torch
from argus.metrics import Metric
from argus.utils import AverageMeter


def corr(y1, y2, dim=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two PyTorch tensors along the specified dimension(s).
    Source: https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/measures/functions.py

    Args:
        y1:      first PyTorch tensor
        y2:      second PyTorch tensor
        dim:     dimension(s) along which the correlation is computed.
                 Any valid PyTorch dim spec works here
        eps:     offset to the standard deviation to avoid exploding
                 the correlation due to small division (default 1e-8)
        **kwargs: passed to final `numpy.mean` operation over standardized y1 * y2

    Returns: correlation tensor
    """
    y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (y1.std(dim=dim, keepdim=True) + eps)
    y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (y2.std(dim=dim, keepdim=True) + eps)
    return (y1 * y2).mean(dim=dim, **kwargs)


class CorrelationMetric(Metric):
    name: str = "correlation"
    better: str = "max"

    def __init__(self):
        self.avg_meter = AverageMeter()
        super().__init__()

    def reset(self):
        self.avg_meter.reset()

    @torch.no_grad()
    def update(self, step_output: dict):
        prediction = step_output["prediction"]
        target = step_output["target"]
        correlations = corr(prediction, target, dim=-1)
        for value in correlations.ravel():
            self.avg_meter.update(value.item())

    def compute(self):
        if self.avg_meter.count == 0:
            raise RuntimeError("Must be at least one example for computation")
        return self.avg_meter.average
