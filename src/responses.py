import abc
from typing import Type

import numpy as np

import torch


class ResponsesProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        pass


class DefaultResponsesProcessor(ResponsesProcessor):
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames)


_RESPONSES_PROCESSOR_REGISTRY: dict[str, Type[ResponsesProcessor]] = dict(
    default=DefaultResponsesProcessor,
)


def get_responses_processor(name: str, processor_params: dict) -> ResponsesProcessor:
    assert name in _RESPONSES_PROCESSOR_REGISTRY
    return _RESPONSES_PROCESSOR_REGISTRY[name](**processor_params)
