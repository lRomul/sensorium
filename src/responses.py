import abc
from typing import Type

import numpy as np

import torch
from torch import nn


class ResponsesProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        pass


class IdentityResponsesProcessor(ResponsesProcessor):
    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(responses)


_RESPONSES_PROCESSOR_REGISTRY: dict[str, Type[ResponsesProcessor]] = dict(
    identity=IdentityResponsesProcessor,
)


def get_responses_processor(name: str, processor_params: dict) -> ResponsesProcessor:
    assert name in _RESPONSES_PROCESSOR_REGISTRY
    return _RESPONSES_PROCESSOR_REGISTRY[name](**processor_params)


class Log(nn.Module):
    def forward(self, x):
        return torch.log(x)


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)
