import numpy as np
from unittest.mock import MagicMock
from sqlmodel import SQLModel, Session, create_engine

from backend.fusion.models import Read
from backend.fusion.tracker import TrackManager
from backend.ingestion.camera import CameraSource


class FakeCamera(CameraSource):
    """Camera that yields pre-loaded frames, then exhausts."""

    def __init__(self, frames: list[tuple[np.ndarray, float]], camera_id: str = "fake"):
        self._frames = list(frames)
        self._camera_id = camera_id
        self._index = 0
        self._started = False

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def active(self) -> bool:
        return self._started and self._index < len(self._frames)

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def read(self) -> tuple[np.ndarray, float] | None:
        if not self._started or self._index >= len(self._frames):
            return None
        frame, ts = self._frames[self._index]
        self._index += 1
        return frame, ts


class FakeDetection:
    def __init__(self, class_name, bbox, confidence):
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence

    def pad(self, ratio, frame_shape):
        x1, y1, x2, y2 = self.bbox
        return max(0, x1 - 5), max(0, y1 - 5), min(frame_shape[1], x2 + 5), min(frame_shape[0], y2 + 5)


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class FakeOCR:
    def __init__(self, text, confidence):
        self._text = text
        self._conf = confidence

    def recognize(self, crop, preprocess=True):
        return self._text, self._conf


def _make_session():
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _frame(value=128):
    return np.full((480, 640, 3), value, dtype=np.uint8)


class TestGateOrchestrator:
    def test_processes_frames_and_stops_when_exhausted(self):
        from backend.ingestion.orchestrator import GateOrchestrator

        # 3 frames produces 3 reads — meets default min_reads=3 for CONFIRMED
        cam = FakeCamera(
            frames=[(_frame(100), 0.0), (_frame(110), 1.0), (_frame(120), 2.0)],
            camera_id="cam0",
        )
        detector = FakeDetector([
            FakeDetection("USDOT", (100, 100, 200, 150), 0.9),
        ])
        ocr = FakeOCR("1234567", 0.92)
        session = _make_session()
        tm = TrackManager(timeout=5.0)  # voter uses default min_reads=3, min_confidence=0.6

        orch = GateOrchestrator(
            cameras=[cam],
            detector=detector,
            ocr=ocr,
            track_manager=tm,
            session=session,
            use_motion=False,
        )
        orch.start()

        # Should have processed frames and flushed the track on shutdown
        assert len(tm.completed) == 1
        assert tm.completed[0][1][0].value == "1234567"

    def test_multi_camera_feeds_shared_track(self):
        from backend.ingestion.orchestrator import GateOrchestrator

        cam1 = FakeCamera(frames=[(_frame(100), 0.0)], camera_id="cam0")
        cam2 = FakeCamera(frames=[(_frame(110), 0.0)], camera_id="cam1")

        detector = FakeDetector([
            FakeDetection("USDOT", (100, 100, 200, 150), 0.9),
        ])
        ocr = FakeOCR("1234567", 0.92)
        session = _make_session()
        tm = TrackManager(timeout=5.0)

        orch = GateOrchestrator(
            cameras=[cam1, cam2],
            detector=detector,
            ocr=ocr,
            track_manager=tm,
            session=session,
            use_motion=False,
        )
        orch.start()

        # Both cameras' reads should be in the same track
        assert len(tm.completed) == 1
        track = tm.completed[0][0]
        camera_ids = {r.camera_id for r in track.reads}
        assert "cam0" in camera_ids
        assert "cam1" in camera_ids

    def test_motion_filter_skips_static_frames(self):
        from backend.ingestion.orchestrator import GateOrchestrator

        # All identical frames — after warmup, motion detector should filter them
        static = _frame(128)
        frames = [(static.copy(), float(i)) for i in range(40)]
        cam = FakeCamera(frames=frames, camera_id="cam0")

        detector = FakeDetector([
            FakeDetection("USDOT", (100, 100, 200, 150), 0.9),
        ])
        ocr = FakeOCR("1234567", 0.92)
        session = _make_session()
        tm = TrackManager(timeout=5.0)

        orch = GateOrchestrator(
            cameras=[cam],
            detector=detector,
            ocr=ocr,
            track_manager=tm,
            session=session,
            use_motion=True,
            motion_threshold=0.01,
            motion_warmup=5,
        )
        orch.start()

        # Only warmup frames should have been processed (5 frames)
        # The rest should have been filtered by motion detection
        if tm.completed:
            track = tm.completed[0][0]
            # Should have far fewer reads than 40 (only warmup frames)
            assert len(track.reads) <= 5

    def test_stop_method(self):
        from backend.ingestion.orchestrator import GateOrchestrator

        # Camera with many frames — we'll stop early
        frames = [(_frame(i % 256), float(i)) for i in range(1000)]
        cam = FakeCamera(frames=frames, camera_id="cam0")

        detector = FakeDetector([])
        ocr = FakeOCR("", 0.0)
        session = _make_session()
        tm = TrackManager(timeout=5.0)

        orch = GateOrchestrator(
            cameras=[cam],
            detector=detector,
            ocr=ocr,
            track_manager=tm,
            session=session,
            use_motion=False,
        )

        import threading
        # Stop after a short delay
        def delayed_stop():
            import time
            time.sleep(0.1)
            orch.stop()

        threading.Thread(target=delayed_stop, daemon=True).start()
        orch.start()

        # Should have stopped before processing all 1000 frames
        assert cam._index < 1000
