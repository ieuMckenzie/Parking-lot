"""Camera capture with threaded frame buffering."""

import collections
import os
import threading
import time

import cv2
import numpy as np


class CameraCapture:
    """Captures frames from a video source in a background thread."""

    def __init__(self, camera_id: str, source_spec: str, resolution: tuple | None = None, target_fps: int = 20):
        self.camera_id = camera_id
        self.source_spec = source_spec
        self.resolution = resolution
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.frame_buffer: collections.deque = collections.deque(maxlen=1)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 0.0
        self.last_frame_time: float = 0
        self.is_connected: bool = False

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        last_saved_time = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.is_connected = False
                self._connect()
                if not self.is_connected:
                    time.sleep(1)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                if isinstance(self.source_spec, str) and os.path.isfile(self.source_spec):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()
            if (now - last_saved_time) >= self.frame_interval:
                if self.resolution:
                    frame = cv2.resize(frame, self.resolution)

                if self.last_frame_time > 0:
                    self.fps = 1.0 / max(0.001, now - self.last_frame_time)
                self.last_frame_time = now

                with self.buffer_lock:
                    self.frame_buffer.append((frame, now))

                last_saved_time = now

    def _connect(self):
        try:
            src = self.source_spec
            if "usb" in src.lower():
                self.cap = cv2.VideoCapture(int(src.replace("usb", "")))
            elif src.isdigit():
                self.cap = cv2.VideoCapture(int(src))
            else:
                self.cap = cv2.VideoCapture(src)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

                if not self.is_connected:
                    print(f"System: Loaded {self.camera_id}")
                    self.is_connected = True
        except Exception:
            pass

    def get_latest_frame(self) -> tuple[np.ndarray | None, float]:
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
        return None, 0
