import abc
from typing import Type

import numpy as np

import torch


class InputsProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frames: np.ndarray, behavior: np.ndarray, pupil_center: np.ndarray) -> torch.Tensor:
        pass


class StackInputsProcessor(InputsProcessor):
    def __init__(self,
                 size: tuple[int, int],
                 pad_fill_value: int = 0):
        self.size = size
        self.pad_fill_value = pad_fill_value

    def __call__(self, frames: np.ndarray, behavior: np.ndarray, pupil_center: np.ndarray) -> torch.Tensor:
        length = frames.shape[-1]
        input_array = np.full((5, length, self.size[1], self.size[0]), self.pad_fill_value, dtype=np.float32)

        frames = np.transpose(frames.astype(np.float32), (2, 0, 1))
        height, width = frames.shape[-2:]
        height_start = (self.size[1] - height) // 2
        width_start = (self.size[0] - width) // 2
        input_array[0, :, height_start: height_start + height, width_start: width_start + width] = frames

        input_array[1:3] = behavior[:, :, None, None]
        input_array[3:] = pupil_center[:, :, None, None]

        tensor_frames = torch.from_numpy(input_array)
        return tensor_frames


_INPUTS_PROCESSOR_REGISTRY: dict[str, Type[InputsProcessor]] = dict(
    stack_inputs=StackInputsProcessor,
)


def get_inputs_processor(name: str, processor_params: dict) -> InputsProcessor:
    assert name in _INPUTS_PROCESSOR_REGISTRY
    return _INPUTS_PROCESSOR_REGISTRY[name](**processor_params)
