import abc
from typing import Type

import numpy as np

import torch

from src.typing import Inputs


class InputsProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self,
                 frames: np.ndarray,
                 behavior: np.ndarray,
                 pupil_center: np.ndarray) -> Inputs:
        pass


class StackInputsProcessor(InputsProcessor):
    def __init__(self,
                 size: tuple[int, int],
                 pad_fill_value: int = 0):
        self.size = size
        self.pad_fill_value = pad_fill_value

    def __call__(self,
                 frames: np.ndarray,
                 behavior: np.ndarray,
                 pupil_center: np.ndarray) -> Inputs:
        length = frames.shape[-1]
        frames_array = np.full((1, length, self.size[1], self.size[0]), self.pad_fill_value, dtype=np.float32)
        frames = np.transpose(frames.astype(np.float32), (2, 0, 1))
        height, width = frames.shape[-2:]
        height_start = (self.size[1] - height) // 2
        width_start = (self.size[0] - width) // 2
        frames_array[0, :, height_start: height_start + height, width_start: width_start + width] = frames
        frames_tensor = torch.from_numpy(frames_array)

        behavior_array = np.concatenate([behavior, pupil_center], axis=0, dtype=np.float32)
        behavior_tensor = torch.from_numpy(behavior_array)

        return frames_tensor, behavior_tensor


_INPUTS_PROCESSOR_REGISTRY: dict[str, Type[InputsProcessor]] = dict(
    stack_inputs=StackInputsProcessor,
)


def get_inputs_processor(name: str, processor_params: dict) -> InputsProcessor:
    assert name in _INPUTS_PROCESSOR_REGISTRY
    return _INPUTS_PROCESSOR_REGISTRY[name](**processor_params)
