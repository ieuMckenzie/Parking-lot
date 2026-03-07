import numpy as np

from backend.fusion.models import Read
from backend.recognition.postprocess import postprocess


def process_frame(
    frame: np.ndarray,
    detector,
    ocr,
    camera_id: str,
    timestamp: float,
    padding_ratio: float = 0.2,
) -> list[Read]:
    """Run detection + OCR on a single frame, return validated Reads."""
    detections = detector.detect(frame)
    reads: list[Read] = []
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.pad(padding_ratio, (h, w))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        raw_text, confidence = ocr.recognize(crop)
        cleaned = postprocess(raw_text, det.class_name)
        if cleaned is None:
            continue

        reads.append(Read(
            text=cleaned,
            raw_text=raw_text,
            confidence=confidence,
            class_name=det.class_name,
            camera_id=camera_id,
            timestamp=timestamp,
        ))

    return reads
