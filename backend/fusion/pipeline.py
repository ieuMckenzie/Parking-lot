import numpy as np

from backend.config import settings
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

    small_threshold = settings.ocr.small_crop_threshold
    small_padding = settings.ocr.small_crop_padding

    for det in detections:
        # Use larger padding for small detections to give OCR more context
        bbox_w = det.bbox[2] - det.bbox[0]
        bbox_h = det.bbox[3] - det.bbox[1]
        pad = small_padding if (bbox_w < small_threshold or bbox_h < small_threshold) else padding_ratio
        x1, y1, x2, y2 = det.pad(pad, (h, w))
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
