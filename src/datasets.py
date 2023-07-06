import abc
import random

import deeplake
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from src.utils import set_random_seed
from src.frames import FramesProcessor
from src.responses import ResponsesProcessor
from src.indexes import StackIndexesGenerator


class MouseVideoDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 deeplake_path: str,
                 indexes_generator: StackIndexesGenerator,
                 frames_processor: FramesProcessor,
                 responses_processor: ResponsesProcessor,
                 augmentations: nn.Module | None = None):
        self.indexes_generator = indexes_generator
        self.frames_processor = frames_processor
        self.responses_processor = responses_processor
        self.augmentations = augmentations

        self.deeplake_dataset = deeplake.load(deeplake_path, access_method="local")
        videos_shape = self.deeplake_dataset.videos.shape
        self.num_videos = videos_shape[0]
        self.num_frames = videos_shape[2]
        self.num_responses = self.deeplake_dataset.responses.shape[2]

    def get_frames(self, video_index: int, frame_indexes: list[int]) -> np.ndarray:
        frames = self.deeplake_dataset.videos[video_index, 0, frame_indexes].numpy()
        return frames

    def get_responses(self, video_index: int, frame_indexes: list[int]) -> np.ndarray:
        responses = self.deeplake_dataset.responses[video_index, :, frame_indexes].numpy()
        responses = np.transpose(responses, (1, 0))
        return responses

    def get_frames_responses(
            self,
            video_index: int,
            frame_indexes: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = self.get_frames(video_index, frame_indexes)
        responses = self.get_responses(video_index, frame_indexes)
        return frames, responses

    def process_frames_responses(self,
                                 frames: np.ndarray,
                                 responses: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self.frames_processor(frames)
        target_tensor = self.responses_processor(responses)
        return input_tensor, target_tensor

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def get_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_index, frame_indexes = self.get_frame_indexes(index)
        frames, responses = self.get_frames_responses(video_index, frame_indexes)
        tensor_frames, tensor_responses = self.process_frames_responses(frames, responses)
        if self.augmentations is not None:
            tensor_frames = self.augmentations(tensor_frames[None])[0]
        return tensor_frames, tensor_responses


class TrainMouseVideoDataset(MouseVideoDataset):
    def __init__(self,
                 deeplake_path: str,
                 indexes_generator: StackIndexesGenerator,
                 frames_processor: FramesProcessor,
                 responses_processor: ResponsesProcessor,
                 epoch_size: int,
                 augmentations: nn.Module | None = None):
        super().__init__(deeplake_path, indexes_generator, frames_processor,
                         responses_processor, augmentations=augmentations)
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def get_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos)
        frame_index = random.randrange(
            self.indexes_generator.behind,
            self.num_frames - self.indexes_generator.ahead
        )
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes


class ValMouseVideoDataset(MouseVideoDataset):
    def __init__(self,
                 deeplake_path: str,
                 indexes_generator: StackIndexesGenerator,
                 frames_processor: FramesProcessor,
                 responses_processor: ResponsesProcessor,
                 augmentations: nn.Module | None = None):
        super().__init__(deeplake_path, indexes_generator, frames_processor,
                         responses_processor, augmentations=augmentations)
        self.window_size = self.indexes_generator.ahead + self.indexes_generator.behind + 1
        self.samples_per_video = self.num_frames // self.window_size

    def __len__(self) -> int:
        return self.num_videos * self.samples_per_video

    def get_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        video_index = index // self.samples_per_video
        video_sample_index = index - video_index * self.samples_per_video
        frame_index = self.indexes_generator.behind + video_sample_index * self.window_size
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes
