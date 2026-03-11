import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from backend.utils.logging import get_logger

log = get_logger("camera")


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
        drop_frames: bool = False,
        max_lag_seconds: float = 0.25,
    ):
        self._path = path
        self._camera_id = camera_id
        self._realtime = realtime
        self._drop_frames = drop_frames
        self._max_lag_seconds = max(0.0, max_lag_seconds)
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
            elif self._drop_frames:
                lag = elapsed - target_time
                if lag > self._max_lag_seconds:
                    # Drop old frames to keep playback close to wall clock time.
                    frames_to_skip = int((lag - self._max_lag_seconds) * self._fps)
                    for _ in range(frames_to_skip):
                        if not self._cap.grab():
                            self._exhausted = True
                            return None
                    self._frame_idx += frames_to_skip

        ret, frame = self._cap.read()
        if not ret:
            self._exhausted = True
            return None

        timestamp = self._frame_idx / self._fps
        self._frame_idx += 1
        return frame, timestamp


class ThreadedCamera(CameraSource):
    """Base for cameras that capture in a background thread with single-slot buffer."""

    def __init__(self, camera_id: str):
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def active(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def read(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            result = self._latest
            self._latest = None
        return result

    def _store_frame(self, frame: np.ndarray, timestamp: float) -> None:
        with self._lock:
            self._latest = (frame, timestamp)

    @abstractmethod
    def _capture_loop(self) -> None: ...


class RTSPCamera(ThreadedCamera):
    """RTSP camera with auto-reconnect and exponential backoff."""

    def __init__(
        self,
        url: str,
        camera_id: str = "rtsp",
        reconnect_max_delay: float = 30.0,
    ):
        super().__init__(camera_id)
        self._url = url
        self._reconnect_max_delay = reconnect_max_delay

    def _capture_loop(self) -> None:
        delay = 1.0
        while self._running:
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                log.warning("rtsp_connect_failed", url=self._url, retry_in=delay)
                time.sleep(delay)
                delay = min(delay * 2, self._reconnect_max_delay)
                continue

            log.info("rtsp_connected", url=self._url, camera_id=self._camera_id)
            delay = 1.0  # reset on success

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("rtsp_read_failed", camera_id=self._camera_id)
                    break
                self._store_frame(frame, time.time())

            cap.release()


class WebcamCamera(ThreadedCamera):
    """USB webcam by device index."""

    def __init__(self, device_index: int = 0, camera_id: str = "webcam"):
        super().__init__(camera_id)
        self._device_index = device_index

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            log.error("webcam_open_failed", device=self._device_index)
            self._running = False
            return

        log.info("webcam_opened", device=self._device_index, camera_id=self._camera_id)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._store_frame(frame, time.time())

        cap.release()
        self._running = False


def parse_source(
    spec: str,
    index: int,
    realtime: bool = False,
    drop_frames: bool = False,
    max_lag_seconds: float = 0.25,
) -> CameraSource:
    """Parse a source specifier string into a CameraSource."""
    camera_id = f"cam{index}"
    if spec.startswith("rtsp://"):
        return RTSPCamera(url=spec, camera_id=camera_id)
    elif spec.startswith("webcam:"):
        device = int(spec.split(":", 1)[1])
        return WebcamCamera(device_index=device, camera_id=camera_id)
    elif spec.startswith("images:"):
        folder = Path(spec.split(":", 1)[1])
        return ImageFolderCamera(folder=folder, camera_id=camera_id)
    else:
        return VideoCamera(
            path=Path(spec),
            camera_id=camera_id,
            realtime=realtime,
            drop_frames=drop_frames,
            max_lag_seconds=max_lag_seconds,
        )
