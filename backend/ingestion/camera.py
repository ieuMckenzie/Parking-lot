import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class CameraSource(ABC):
    """Base class for all camera input sources."""

    @property
    @abstractmethod
    def camera_id(self) -> str: ...

    @property
    def active(self) -> bool:
        """Whether this source may still produce frames."""
        return True

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, float] | None:
        """Return (frame, timestamp) or None if no frame available."""
        ...


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageFolderCamera(CameraSource):
    """Yields images from a folder as sequential frames."""

    def __init__(self, folder: Path, camera_id: str = "folder", fps: float = 1.0):
        self._folder = folder
        self._camera_id = camera_id
        self._fps = fps
        self._images: list[Path] = []
        self._index = 0
        self._started = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def active(self) -> bool:
        return self._started and self._index < len(self._images)

    def start(self) -> None:
        self._images = sorted(
            p for p in self._folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self._index = 0
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> tuple[np.ndarray, float] | None:
        if not self._started or self._index >= len(self._images):
            return None
        path = self._images[self._index]
        frame = cv2.imread(str(path))
        timestamp = self._index / self._fps
        self._index += 1
        if frame is None:
            return self.read()  # skip unreadable, advance index
        return frame, timestamp


class VideoCamera(CameraSource):
    """Plays a video file as a camera source."""

    def __init__(
        self,
        path: Path,
        camera_id: str = "video",
        realtime: bool = False,
    ):
        self._path = path
        self._camera_id = camera_id
        self._realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._frame_idx: int = 0
        self._start_time: float = 0.0
        self._exhausted = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def active(self) -> bool:
        return self._cap is not None and not self._exhausted

    def start(self) -> None:
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Cannot open video: {self._path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_idx = 0
        self._start_time = time.monotonic()
        self._exhausted = False

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._exhausted = True

    def read(self) -> tuple[np.ndarray, float] | None:
        if self._cap is None or self._exhausted:
            return None

        if self._realtime:
            target_time = self._frame_idx / self._fps
            elapsed = time.monotonic() - self._start_time
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

        ret, frame = self._cap.read()
        if not ret:
            self._exhausted = True
            return None

        timestamp = self._frame_idx / self._fps
        self._frame_idx += 1
        return frame, timestamp
