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
        tensor_responses = torch.from_numpy(responses)
        tensor_responses = torch.log1p(tensor_responses)
        return tensor_responses


class IndexingResponsesProcessor(ResponsesProcessor):
    def __init__(self, index: int | list[int]):
        self.index = index

    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        responses = responses[..., self.index]
        tensor_responses = torch.from_numpy(responses)
        tensor_responses = torch.log1p(tensor_responses)
        return tensor_responses


class SelectLastResponsesProcessor(IndexingResponsesProcessor):
    def __init__(self):
        super().__init__(index=-1)


_RESPONSES_PROCESSOR_REGISTRY: dict[str, Type[ResponsesProcessor]] = dict(
    identity=IdentityResponsesProcessor,
    indexing=IndexingResponsesProcessor,
    last=SelectLastResponsesProcessor,
)


def get_responses_processor(name: str, processor_params: dict) -> ResponsesProcessor:
    assert name in _RESPONSES_PROCESSOR_REGISTRY
    return _RESPONSES_PROCESSOR_REGISTRY[name](**processor_params)


class Expm1(nn.Module):
    def forward(self, x):
        return torch.expm1(x)
