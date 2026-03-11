import numpy as np

from backend.fusion.models import Read
from backend.fusion.pipeline import process_frame


class FakeDetection:
    def __init__(self, class_name: str, bbox: tuple, confidence: float):
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence

    def pad(self, ratio, frame_shape):
        x1, y1, x2, y2 = self.bbox
        return (
            max(0, x1 - 5), max(0, y1 - 5),
            min(frame_shape[1], x2 + 5), min(frame_shape[0], y2 + 5),
        )


class FakeDetector:
    def __init__(self, detections: list):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class FakeOCR:
    def __init__(self, results: dict[str, tuple[str, float]]):
        """results maps expected class_name to (text, confidence)."""
        self._results = results
        self._call_idx = 0

    def recognize(self, crop, preprocess=True, min_confidence=None):
        keys = list(self._results.keys())
        if self._call_idx < len(keys):
            key = keys[self._call_idx]
            self._call_idx += 1
            return self._results[key]
        return ("", 0.0)


def test_process_frame_produces_reads():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        FakeDetection("USDOT", (100, 100, 200, 150), 0.9),
    ]
    detector = FakeDetector(detections)
    ocr = FakeOCR({"USDOT": ("1234567", 0.92)})

    reads, annotations = process_frame(
        frame=frame,
        detector=detector,
        ocr=ocr,
        camera_id="cam1",
        timestamp=100.0,
        padding_ratio=0.2,
    )
    assert len(reads) == 1
    assert reads[0].text == "1234567"
    assert reads[0].class_name == "USDOT"
    assert reads[0].camera_id == "cam1"
    assert len(annotations) == 1
    assert annotations[0].class_name == "USDOT"


def test_process_frame_filters_invalid_ocr():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        FakeDetection("USDOT", (100, 100, 200, 150), 0.9),
    ]
    detector = FakeDetector(detections)
    # "AB" is too short to be a valid USDOT
    ocr = FakeOCR({"USDOT": ("AB", 0.5)})

    reads, annotations = process_frame(
        frame=frame,
        detector=detector,
        ocr=ocr,
        camera_id="cam1",
        timestamp=100.0,
        padding_ratio=0.2,
    )
    assert len(reads) == 0
    # Annotation still present even though read was filtered
    assert len(annotations) == 1


def test_process_frame_no_detections():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeDetector([])
    ocr = FakeOCR({})

    reads, annotations = process_frame(
        frame=frame,
        detector=detector,
        ocr=ocr,
        camera_id="cam1",
        timestamp=100.0,
    )
    assert reads == []
    assert annotations == []
