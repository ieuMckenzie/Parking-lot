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
