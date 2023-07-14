import abc
from typing import Type

import numpy as np

import torch


def pad_frames(frames: torch.Tensor,
               size: tuple[int, int],
               pad_mode: str = "constant",
               fill_value: int = 0) -> torch.Tensor:
    height, width = frames.shape[-2:]
    height_pad = size[1] - height
    width_pad = size[0] - width
    assert height_pad >= 0 and width_pad >= 0

    top_height_pad: int = height_pad // 2
    bottom_height_pad: int = height_pad - top_height_pad
    left_width_pad: int = width_pad // 2
    right_width_pad: int = width_pad - left_width_pad
    frames = torch.nn.functional.pad(
        frames,
        [left_width_pad, right_width_pad, top_height_pad, bottom_height_pad],
        mode=pad_mode,
        value=fill_value,
    )
    return frames


class InputsProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, frames: np.ndarray, behavior: np.ndarray, pupil_center: np.ndarray) -> torch.Tensor:
        pass


class PadFramesProcessor(InputsProcessor):
    def __init__(self,
                 size: tuple[int, int],
                 pad_mode: str = "constant",
                 fill_value: int = 0):
        self.size = size
        self.pad_mode = pad_mode
        self.fill_value = fill_value

    def __call__(self, frames: np.ndarray, behavior: np.ndarray, pupil_center: np.ndarray) -> torch.Tensor:
        frames = np.transpose(frames.astype(np.float32), (2, 0, 1))
        tensor_frames = torch.from_numpy(frames)
        tensor_frames = pad_frames(tensor_frames, self.size,
                                   pad_mode=self.pad_mode,
                                   fill_value=self.fill_value)
        return tensor_frames


def draw_inputs_squares(array: np.ndarray, size: tuple[int, int]):
    width, height = size
    assert width % 2 == 0
    array = np.transpose(array, axes=(1, 0))[:, :, None, None]
    tensor = torch.from_numpy(array)
    tensor = torch.nn.functional.interpolate(tensor, (height, width // 2), mode="nearest")
    tensor = torch.concatenate((tensor[:, 0], tensor[:, 1]), dim=2)
    return tensor


class MosaicInputsProcessor(InputsProcessor):
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, frames: np.ndarray, behavior: np.ndarray, pupil_center: np.ndarray) -> torch.Tensor:
        frames = np.transpose(frames, (2, 0, 1))
        tensor_frames = torch.from_numpy(frames)
        height, width = frames.shape[-2:]
        mosaic_height = self.size[1] - height
        square_height = mosaic_height // 2
        tensor_behavior = draw_inputs_squares(behavior, size=(width, square_height))
        tensor_pupil_center = draw_inputs_squares(pupil_center, size=(width, mosaic_height - square_height))
        tensor_input = torch.concatenate((tensor_behavior, tensor_frames, tensor_pupil_center), dim=1)
        return tensor_input


_INPUTS_PROCESSOR_REGISTRY: dict[str, Type[InputsProcessor]] = dict(
    pad_frames=PadFramesProcessor,
    mosaic_inputs=MosaicInputsProcessor,
)


def get_inputs_processor(name: str, processor_params: dict) -> InputsProcessor:
    assert name in _INPUTS_PROCESSOR_REGISTRY
    return _INPUTS_PROCESSOR_REGISTRY[name](**processor_params)
