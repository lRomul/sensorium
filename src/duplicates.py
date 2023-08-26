import random
from collections import defaultdict

import imagehash
import numpy as np
from PIL import Image

from torch import Tensor

from src.data import get_length_without_nan
from src.datasets import MouseVideoDataset, ConcatMiceVideoDataset


def calculate_frame_phash(frame: np.ndarray) -> tuple[bool, ...]:
    frame = Image.fromarray(frame.astype(np.uint8), 'L')
    phash = imagehash.phash(frame).hash
    return tuple(phash.ravel().tolist())


def calculate_video_phash(video: np.ndarray, num_hash_frames: int = 5) -> tuple[bool, ...]:
    length = get_length_without_nan(video[0, 0])
    assert length >= num_hash_frames
    step = length // num_hash_frames
    frame_hashes: list[bool] = []
    for frame_index in range(step // 2, length, step)[:num_hash_frames]:
        frame_hashes += calculate_frame_phash(video[..., frame_index])
    return tuple(frame_hashes)


def get_hash_data_dict(mice_data: list[dict], num_hash_frames: int = 5) -> dict[tuple[bool, ...], set[tuple[int, int]]]:
    hash_data_dict = defaultdict(set)
    for mouse_index, mouse_data in enumerate(mice_data):
        for trial_index, trial_data in enumerate(mouse_data["trials"]):
            video = np.load(trial_data["video_path"])
            video_phash = calculate_video_phash(video, num_hash_frames)
            hash_data_dict[video_phash].add((mouse_index, trial_index))
    return dict(hash_data_dict)


def get_trial2duplicate(mice_data: list[dict], num_hash_frames: int = 5) -> dict[tuple[int, int], set[tuple[int, int]]]:
    hash_data_dict = get_hash_data_dict(mice_data, num_hash_frames=num_hash_frames)
    trial2duplicate: dict[tuple[int, int], set[tuple[int, int]]] = dict()
    for video_phash, trials_set in hash_data_dict.items():
        for trial in trials_set:
            trial2duplicate[trial] = trials_set - {trial}
    return trial2duplicate


class TrainDuplicatesMiceVideoDataset(ConcatMiceVideoDataset):
    def __init__(self,
                 mice_datasets: list[MouseVideoDataset],
                 duplicate_weight: float = 1.0,
                 num_hash_frames: int = 5):
        super().__init__(mice_datasets=mice_datasets)
        self.trial2duplicate = get_trial2duplicate(
            [mouse_dataset.mouse_data for mouse_dataset in self.mice_datasets],
            num_hash_frames=num_hash_frames,
        )
        self.duplicate_weight = duplicate_weight

    def __getitem__(self, index: int) -> tuple[Tensor, tuple[list[Tensor], Tensor]]:
        mouse_index = random.randrange(len(self.mice_datasets))
        trial_index, indexes = self.mice_datasets[mouse_index].get_indexes(index)
        mouse_sample = self.mice_datasets[mouse_index].get_sample_tensors(trial_index, indexes)
        mice_sample = self.construct_mice_sample(mouse_index, mouse_sample)

        trial_duplicate_set = self.trial2duplicate[(mouse_index, trial_index)]
        if trial_duplicate_set:
            dupl_mouse_index, dupl_trial_index = random.choice(list(trial_duplicate_set))
            _, dupl_target_tensor = self.mice_datasets[dupl_mouse_index].get_sample_tensors(dupl_trial_index, indexes)
            target_tensors, mice_weights = mice_sample[1]
            target_tensors[dupl_mouse_index] = dupl_target_tensor
            mice_weights[dupl_mouse_index] = self.duplicate_weight

        return mice_sample
