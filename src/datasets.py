import abc
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from src.mixup import Mixup
from src.data import get_mouse_data
from src.utils import set_random_seed
from src.inputs import InputsProcessor
from src.indexes import IndexesGenerator
from src.responses import ResponsesProcessor
from src import constants


class MouseVideoDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 mouse: str, split: str,
                 indexes_generator: IndexesGenerator,
                 inputs_processor: InputsProcessor,
                 responses_processor: ResponsesProcessor):
        self.mouse = mouse
        self.mouse_index = constants.mouse2index[mouse]
        self.mouse_data = get_mouse_data(mouse=mouse, split=split)
        self.indexes_generator = indexes_generator
        self.inputs_processor = inputs_processor
        self.responses_processor = responses_processor

        self.trials = self.mouse_data["trials"]
        self.num_trials = len(self.trials)
        self.trials_lengths = [t["length"] for t in self.trials]
        self.num_neurons = self.mouse_data["num_neurons"]

    def get_frames(self, trial_index: int, indexes: list[int]) -> np.ndarray:
        frames = np.load(self.trials[trial_index]["video_path"])[..., indexes]
        return frames

    def get_responses(self, trial_index: int, indexes: list[int]) -> np.ndarray:
        responses = np.load(self.trials[trial_index]["response_path"])[..., indexes]
        return responses

    def get_behavior(self, trial_index: int, indexes: list[int]) -> np.ndarray:
        behavior = np.load(self.trials[trial_index]["behavior_path"])[..., indexes]
        return behavior

    def get_pupil_center(self, trial_index: int, indexes: list[int]) -> np.ndarray:
        pupil_center = np.load(self.trials[trial_index]["pupil_center_path"])[..., indexes]
        return pupil_center

    def get_inputs_responses(
            self,
            trial_index: int,
            indexes: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frames = self.get_frames(trial_index, indexes)
        responses = self.get_responses(trial_index, indexes)
        behavior = self.get_behavior(trial_index, indexes)
        pupil_center = self.get_pupil_center(trial_index, indexes)
        return frames, behavior, pupil_center, responses

    def process_inputs_responses(self,
                                 frames: np.ndarray,
                                 behavior: np.ndarray,
                                 pupil_center: np.ndarray,
                                 responses: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self.inputs_processor(frames, behavior, pupil_center)
        target_tensor = self.responses_processor(responses)
        return input_tensor, target_tensor

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def get_sample_tensors(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trial_index, indexes = self.get_indexes(index)
        frames, behavior, pupil_center, responses = self.get_inputs_responses(trial_index, indexes)
        return self.process_inputs_responses(frames, behavior, pupil_center, responses)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_sample_tensors(index)


class TrainMouseVideoDataset(MouseVideoDataset):
    def __init__(self,
                 mouse: str,
                 indexes_generator: IndexesGenerator,
                 inputs_processor: InputsProcessor,
                 responses_processor: ResponsesProcessor,
                 epoch_size: int,
                 augmentations: nn.Module | None = None,
                 mixup: Mixup | None = None):
        super().__init__(mouse, "train", indexes_generator, inputs_processor, responses_processor)
        self.epoch_size = epoch_size
        self.augmentations = augmentations
        self.mixup = mixup

    def __len__(self) -> int:
        return self.epoch_size

    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        set_random_seed(index)
        trial_index = random.randrange(0, self.num_trials)
        num_frames = self.trials[trial_index]["length"]
        frame_index = random.randrange(
            self.indexes_generator.behind,
            num_frames - self.indexes_generator.ahead
        )
        indexes = self.indexes_generator.make_indexes(frame_index)
        return trial_index, indexes

    def get_sample_tensors(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, responses = super().get_sample_tensors(index)
        if self.augmentations is not None:
            frames = self.augmentations(frames[None])[0]
        return frames, responses

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.get_sample_tensors(index)
        if self.mixup is not None and self.mixup.use():
            random_sample = self.get_sample_tensors(index + 1)
            sample = self.mixup(sample, random_sample)
        return sample


class ValMouseVideoDataset(MouseVideoDataset):
    def __init__(self,
                 mouse: str,
                 indexes_generator: IndexesGenerator,
                 inputs_processor: InputsProcessor,
                 responses_processor: ResponsesProcessor):
        super().__init__(mouse, "val", indexes_generator, inputs_processor, responses_processor)
        self.window_size = self.indexes_generator.width
        self.samples_per_trials = [length // self.window_size for length in self.trials_lengths]
        self.num_samples = sum(self.samples_per_trials)

    def __len__(self) -> int:
        return self.num_samples

    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        trial_sample_index = index
        trial_index = 0
        for trial_index, num_trial_samples in enumerate(self.samples_per_trials):
            if trial_sample_index >= num_trial_samples:
                trial_sample_index -= num_trial_samples
            else:
                break

        frame_index = self.indexes_generator.behind + trial_sample_index * self.window_size
        indexes = self.indexes_generator.make_indexes(frame_index)
        return trial_index, indexes


class ConcatMiceVideoDataset(Dataset):
    def __init__(self, mice_datasets: list[MouseVideoDataset]):
        assert [d.mouse_index for d in mice_datasets] == constants.mice_indexes
        self.mice_datasets = mice_datasets
        self.samples_per_dataset = [len(d) for d in mice_datasets]
        self.num_samples = sum(self.samples_per_dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[list[torch.Tensor], torch.Tensor]]:
        assert 0 <= index < self.__len__()
        sample_index = index
        mouse_index = 0
        for mouse_index, num_trial_samples in enumerate(self.samples_per_dataset):
            if sample_index >= num_trial_samples:
                sample_index -= num_trial_samples
            else:
                break
        input_tensor, target_tensor = self.mice_datasets[mouse_index][sample_index]
        target_tensors = []
        for index in constants.mice_indexes:
            if index == mouse_index:
                target_tensors.append(target_tensor)
            else:
                temporal_shape = [target_tensor.shape[-1]] if len(target_tensor.shape) == 2 else []
                target_tensors.append(
                    torch.zeros(constants.num_neurons[index], *temporal_shape, dtype=torch.float32)
                )
        mice_weights = torch.zeros(constants.num_mice, dtype=torch.float32)
        mice_weights[mouse_index] = 1.0
        return input_tensor, (target_tensors, mice_weights)
