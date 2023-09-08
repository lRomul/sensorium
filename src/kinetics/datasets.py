import abc
import random
import multiprocessing
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.indexes import IndexesGenerator
from src.kinetics.frame_fetchers import NvDecFrameFetcher, OpencvFrameFetcher
from src.utils import set_random_seed
from src.kinetics import constants


def get_video_info(video_path: str | Path) -> dict[str, int | float]:
    video = cv2.VideoCapture(str(video_path))
    video_info = dict(
        frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(video.get(cv2.CAP_PROP_FPS)),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return video_info


def parse_df_row(row: dict) -> dict:
    video_name = f"{row['youtube_id']}_{row['time_start']:06}_{row['time_end']:06}.mp4"
    video_path = constants.kinetics_dir / row["split"] / video_name
    label = row['label']
    return {
        "label": label,
        "target": constants.class2target[label],
        "video_path": str(video_path),
        "video_info": get_video_info(video_path),
    }


def get_videos_data(split: str):
    data_df = pd.read_csv(constants.annotations_dir / f"{split}.csv")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        rows = data_df.to_dict('index').values()
        videos_data = list(tqdm(pool.imap(parse_df_row, rows), total=len(rows)))
    return videos_data


def process_frames(frames: Tensor, size: int, channels: int = 5):
    resized_frames = torch.nn.functional.interpolate(frames.to(torch.float32).unsqueeze(0),
                                                     scale_factor=size / max(frames.shape[-2:]), mode="area")
    padded_frames = torch.zeros(channels, resized_frames.shape[1], size, size,
                                dtype=torch.float32, device=frames.device)
    height, width = resized_frames.shape[-2:]
    height_start = (size - height) // 2
    width_start = (size - width) // 2
    padded_frames[:, :, height_start: height_start + height, width_start: width_start + width] = resized_frames
    return padded_frames


class KineticsDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 videos_data: list[dict],
                 indexes_generator: IndexesGenerator,
                 frame_size: int,
                 channels: int,
                 gpu_id: int = 0):
        self.videos_data = [
            v for v in videos_data if v["video_info"]["frame_count"] > indexes_generator.width * 2
        ]
        self.indexes_generator = indexes_generator
        self.frame_size = frame_size
        self.channels = channels
        self.gpu_id = gpu_id

        self.num_videos = len(self.videos_data)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def get_frames(self, video_index: int, frame_indexes: list[int]) -> Tensor:
        try:
            frame_fetcher = NvDecFrameFetcher(
                self.videos_data[video_index]["video_path"],
                gpu_id=self.gpu_id
            )
        except:
            frame_fetcher = OpencvFrameFetcher(
                self.videos_data[video_index]["video_path"],
                gpu_id=self.gpu_id
            )
        frames = frame_fetcher.fetch_frames(frame_indexes)
        frames = process_frames(frames, size=self.frame_size, channels=self.channels)
        return frames

    def get_target(self, video_index: int) -> Tensor:
        target = torch.tensor(self.videos_data[video_index]["target"], dtype=torch.int64)
        return target

    def get_sample_tensors(self, index: int) -> tuple[Tensor, Tensor]:
        video_index, frame_indexes = self.get_indexes(index)
        frames = self.get_frames(video_index, frame_indexes)
        target = self.get_target(video_index)
        return frames, target

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_sample_tensors(index)


class TrainKineticsDataset(KineticsDataset):
    def __init__(self,
                 videos_data: list[dict],
                 indexes_generator: IndexesGenerator,
                 frame_size: int,
                 channels: int,
                 epoch_size: int,
                 gpu_id: int = 0):
        super().__init__(videos_data, indexes_generator, frame_size, channels, gpu_id=gpu_id)
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos)
        num_frames = self.videos_data[video_index]["video_info"]["frame_count"]
        frame_index = random.randrange(
            self.indexes_generator.behind,
            num_frames - self.indexes_generator.ahead
        )
        frame_indexes = self.indexes_generator.make_indexes(frame_index)
        return video_index, frame_indexes


class ValKineticsDataset(KineticsDataset):
    def __len__(self) -> int:
        return self.num_videos

    def get_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        length = self.videos_data[index]["video_info"]["frame_count"]
        frame_indexes = self.indexes_generator.make_indexes(length // 2)
        return index, frame_indexes
