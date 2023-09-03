import imagehash
import numpy as np
from PIL import Image

from src.utils import get_length_without_nan


def binary_array_to_int(arr: np.ndarray) -> int:
    bit_string = ''.join(str(b) for b in 1 * arr.flatten())
    return int(bit_string, 2)


def calculate_frame_phash(frame: np.ndarray) -> int:
    frame = Image.fromarray(frame.astype(np.uint8), 'L')
    phash = imagehash.phash(frame).hash
    return binary_array_to_int(phash.ravel())


def calculate_video_phash(video: np.ndarray, num_hash_frames: int = 5) -> int:
    length = get_length_without_nan(video[0, 0])
    assert length >= num_hash_frames
    step = length // num_hash_frames
    video_hash: int = 0
    for frame_index in range(step // 2, length, step)[:num_hash_frames]:
        video_hash ^= calculate_frame_phash(video[..., frame_index])
    return video_hash
