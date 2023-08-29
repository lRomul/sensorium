import abc
from typing import Type

import numpy as np

import torch


def responses_to_tensor(responses: np.ndarray) -> torch.Tensor:
    responses = responses.astype(np.float32)
    responses_tensor = torch.from_numpy(responses)
    responses_tensor = torch.relu(responses_tensor)
    return responses_tensor


class ResponsesProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        pass


class IdentityResponsesProcessor(ResponsesProcessor):
    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        return responses_to_tensor(responses)


class IndexingResponsesProcessor(ResponsesProcessor):
    def __init__(self, index: int | list[int]):
        self.index = index

    def __call__(self, responses: np.ndarray) -> torch.Tensor:
        responses = responses[..., self.index]
        responses_tensor = responses_to_tensor(responses)
        return responses_tensor


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
