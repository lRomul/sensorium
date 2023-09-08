import abc
import logging
from pathlib import Path
from typing import Optional, Any

import cv2

import torch

import PyNvCodec as nvc
import PytorchNvCodec as pnvc


logger = logging.getLogger(__name__)


class AbstractFrameFetcher(metaclass=abc.ABCMeta):
    def __init__(self, video_path: str | Path, gpu_id: int):
        self.video_path = Path(video_path)
        self.gpu_id = gpu_id
        self.num_frames = -1
        self.width = -1
        self.height = -1

        self._current_index = -1

    @property
    def current_index(self) -> int:
        return self._current_index

    def fetch_frame(self, index: Optional[int] = None) -> torch.Tensor:
        try:
            if index is None:
                if self._current_index < self.num_frames - 1:
                    frame = self._next_decode()
                    self._current_index += 1
                else:
                    raise RuntimeError("End of frames")
            else:
                if index < 0 or index >= self.num_frames:
                    raise RuntimeError(f"Frame index {index} out of range")
                frame = self._seek_and_decode(index)
                self._current_index = index

            frame = self._convert(frame)
        except BaseException as error:
            logger.error(
                f"Error while fetching frame {index} from '{str(self.video_path)}': {error}."
                f"Replace by empty frame."
            )
            frame = torch.zeros(self.height, self.width,
                                dtype=torch.uint8,
                                device=f"cuda:{self.gpu_id}")
        return frame

    def fetch_frames(self, indexes: list[int]) -> torch.Tensor:
        min_frame_index = min(indexes)
        max_frame_index = max(indexes)

        index2frame = dict()
        frame_indexes_set = set(indexes)
        for index in range(min_frame_index, max_frame_index + 1):
            if index not in frame_indexes_set:
                self._next_decode()
                continue
            if index == min_frame_index:
                frame_tensor = self.fetch_frame(index)
            else:
                frame_tensor = self.fetch_frame()
            index2frame[index] = frame_tensor

        frames = [index2frame[index] for index in indexes]
        return torch.stack(frames, dim=0)

    @abc.abstractmethod
    def _next_decode(self) -> Any:
        pass

    @abc.abstractmethod
    def _seek_and_decode(self, index: int) -> Any:
        pass

    @abc.abstractmethod
    def _convert(self, frame: Any) -> torch.Tensor:
        pass


class OpencvFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self.video = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _next_decode(self) -> Any:
        _, frame = self.video.read()
        return frame

    def _seek_and_decode(self, index: int) -> Any:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, frame = self.video.read()
        return frame

    def _convert(self, frame: Any) -> torch.Tensor:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.from_numpy(grayscale_frame)
        frame_tensor = frame_tensor.to(device=f"cuda:{self.gpu_id}")
        return frame_tensor


class NvDecFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self._nv_dec = nvc.PyNvDecoder(str(self.video_path), self.gpu_id)
        self.num_frames = self._nv_dec.Numframes()
        self.width = self._nv_dec.Width()
        self.height = self._nv_dec.Height()

        self._current_index = -1

        self._to_grayscale = nvc.PySurfaceConverter(
            self.width,
            self.height,
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.Y,
            self.gpu_id,
        )
        self._cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )

    def _next_decode(self) -> Any:
        nv12_surface = self._nv_dec.DecodeSingleSurface()
        return nv12_surface

    def _seek_and_decode(self, index: int) -> Any:
        seek_ctx = nvc.SeekContext(index)
        nv12_surface = self._nv_dec.DecodeSingleSurface(seek_context=seek_ctx)
        return nv12_surface

    def _convert(self, frame: Any) -> torch.Tensor:
        grayscale_surface = self._to_grayscale.Execute(frame, self._cc_ctx)
        surf_plane = grayscale_surface.PlanePtr()
        frame_tensor = pnvc.makefromDevicePtrUint8(
            surf_plane.GpuMem(),
            surf_plane.Width(),
            surf_plane.Height(),
            surf_plane.Pitch(),
            surf_plane.ElemSize(),
        )
        frame_tensor.resize_(self.height, self.width)
        return frame_tensor
