"""Thread-safe shared state for detections, plates, and display text."""

import os
import threading
import time
from difflib import SequenceMatcher

from parking_lot.config import DetectionConfig


class SharedState:
    """Central thread-safe store for detections, plate votes, and display text."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._seen_plates: dict[str, float] = {}
        self._plate_votes: dict[str, dict[str, int]] = {}
        self._display_text: dict[tuple, tuple] = {}
        self.latest_detections: dict[str, list[dict]] = {}
        self._last_nonempty: dict[str, list[dict]] = {}
        self._last_nonempty_time: dict[str, float] = {}
        self._static_camera_ids: set[str] = set()

    def set_static_cameras(self, camera_ids: list[str], sources: list[str]):
        with self._lock:
            self._static_camera_ids = {
                cid for cid, src in zip(camera_ids, sources) if os.path.isfile(src)
            }

    def update_detections(self, cam_id: str, detections: list[dict]):
        with self._lock:
            if not detections and cam_id in self._static_camera_ids:
                return
            self.latest_detections[cam_id] = detections
            if detections:
                self._last_nonempty[cam_id] = detections
                self._last_nonempty_time[cam_id] = time.time()

    def get_detections(self, cam_id: str) -> list[dict]:
        with self._lock:
            det = self.latest_detections.get(cam_id, [])
            if det:
                return det
            last = self._last_nonempty.get(cam_id, [])
            if not last:
                return []
            t = self._last_nonempty_time.get(cam_id, 0)
            if (time.time() - t) <= self.cfg.detection_hold_time:
                return last
            return []

    def is_similar_to_recent(self, new_text: str, current_time: float) -> tuple[bool, str | None]:
        with self._lock:
            for past_text, last_time in self._seen_plates.items():
                if (current_time - last_time) < self.cfg.log_cooldown:
                    ratio = SequenceMatcher(None, new_text, past_text).ratio()
                    if ratio >= self.cfg.similarity_thresh:
                        return True, past_text
            return False, None

    def update_seen_plate(self, plate: str, timestamp: float):
        with self._lock:
            self._seen_plates[plate] = timestamp

    def remove_seen_plate(self, plate: str):
        with self._lock:
            self._seen_plates.pop(plate, None)

    def add_plate_vote(self, original: str, candidate: str) -> int:
        with self._lock:
            if original not in self._plate_votes:
                self._plate_votes[original] = {}
            self._plate_votes[original][candidate] = self._plate_votes[original].get(candidate, 0) + 1
            return self._plate_votes[original][candidate]

    def set_display_text(self, camera_id: str, box_id: str, text: str, color: tuple):
        with self._lock:
            self._display_text[(camera_id, box_id)] = (text, color)

    def get_display_text(self, camera_id: str, box_id: str) -> tuple[str | None, tuple | None]:
        with self._lock:
            return self._display_text.get((camera_id, box_id), (None, None))

    def get_all_detections(self) -> dict[str, list[dict]]:
        """Return a snapshot of all current detections (for API)."""
        with self._lock:
            return {k: list(v) for k, v in self.latest_detections.items()}

    def get_seen_plates(self) -> dict[str, float]:
        with self._lock:
            return dict(self._seen_plates)
