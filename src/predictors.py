from pathlib import Path

import numpy as np

import torch
import argus

from src.indexes import StackIndexesGenerator
from src.frames import get_frames_processor


class Predictor:
    def __init__(self, model_path: Path | str, device: str = "cuda:0"):
        self.model = argus.load_model(model_path, device=device, optimizer=None, loss=None)
        self.model.eval()
        self.frames_processor = get_frames_processor(*self.model.params["frames_processor"])
        self.frame_stack_size = self.model.params["frame_stack"]["size"]
        self.frame_stack_step = self.model.params["frame_stack"]["step"]
        assert self.model.params["frame_stack"]["position"] == "last"
        self.indexes_generator = StackIndexesGenerator(self.frame_stack_size,
                                                       self.frame_stack_step)
        self.num_neurons = self.model.params["num_neurons"]

    @torch.no_grad()
    def predict_trial(self, video: np.ndarray) -> np.ndarray:
        frames = self.frames_processor(video).to(self.model.device)
        length = video.shape[-1]
        responses = np.full((self.num_neurons, length), np.nan, dtype=np.float32)
        for index in range(
            self.indexes_generator.behind,
            length - self.indexes_generator.ahead
        ):
            indexes = self.indexes_generator.make_stack_indexes(index)
            prediction = self.model.predict(frames[indexes].unsqueeze(0))[0]
            responses[..., index] = prediction.cpu().numpy()
        return responses
