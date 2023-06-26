import abc

import deeplake
import numpy as np

from torch.utils.data import Dataset

from src.frames import FramesProcessor
from src.responses import ResponsesProcessor


class MouseVideoDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 deeplake_path: str,
                 frames_processor: FramesProcessor,
                 responses_processor: ResponsesProcessor):
        self.frames_processor = frames_processor
        self.responses_processor = responses_processor

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

    def process_frames_responses(self, frames: np.ndarray, responses: np.ndarray):
        input_tensor = self.frames_processor(frames)
        target_tensor = self.responses_processor(responses)
        return input_tensor, target_tensor

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def get_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def __getitem__(self, index):
        video_index, frame_indexes = self.get_frame_indexes(index)
        frames, responses = self.get_frames_responses(video_index, frame_indexes)
        frames, responses = self.process_frames_responses(frames, responses)
        return frames, responses
