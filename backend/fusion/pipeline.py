from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from backend.config import settings
from backend.fusion.models import Read
from backend.recognition.postprocess import postprocess


@dataclass
class Annotation:
    """Visual annotation data for a single detection on a frame."""
    bbox: tuple[int, int, int, int]
    class_name: str
    text: str
    confidence: float
    det_confidence: float = 0.0  # YOLO detection confidence (fallback when OCR fails)


class FrameResult(NamedTuple):
    reads: list[Read]
    annotations: list[Annotation]


_CONTAINER_CLASSES = {"ContainerNum", "ContainerPlate"}


def process_frame(
    frame: np.ndarray,
    detector,
    ocr,
    camera_id: str,
    timestamp: float,
    padding_ratio: float = 0.2,
    skip_ocr_classes: set[str] | None = None,
    run_ocr: bool = True,
) -> FrameResult:
    """Run detection + OCR on a single frame, return validated Reads and annotations."""
    detections = detector.detect(frame)
    reads: list[Read] = []
    annotations: list[Annotation] = []
    h, w = frame.shape[:2]

    small_threshold = settings.ocr.small_crop_threshold
    small_padding = settings.ocr.small_crop_padding
    container_min_conf = settings.ocr.container_min_confidence
    max_candidates = settings.ocr.max_candidates_per_frame

    ocr_indices: set[int]
    if len(detections) <= max_candidates:
        ocr_indices = set(range(len(detections)))
    else:
        # OCR is expensive; prioritize highest-confidence detections each frame.
        ranked = sorted(enumerate(detections), key=lambda item: item[1].confidence, reverse=True)
        ocr_indices = {idx for idx, _ in ranked[:max_candidates]}

    skip_ocr_classes = skip_ocr_classes or set()

    for idx, det in enumerate(detections):
        if (not run_ocr) or idx not in ocr_indices or det.class_name in skip_ocr_classes:
            annotations.append(Annotation(
                bbox=det.bbox,
                class_name=det.class_name,
                text="",
                confidence=det.confidence,
                det_confidence=det.confidence,
            ))
            continue

        # Use larger padding for small detections to give OCR more context
        bbox_w = det.bbox[2] - det.bbox[0]
        bbox_h = det.bbox[3] - det.bbox[1]
        pad = small_padding if (bbox_w < small_threshold or bbox_h < small_threshold) else padding_ratio
        x1, y1, x2, y2 = det.pad(pad, (h, w))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Use lower OCR threshold for container classes
        ocr_min_conf = container_min_conf if det.class_name in _CONTAINER_CLASSES else None
        raw_text, confidence = ocr.recognize(crop, min_confidence=ocr_min_conf)
        cleaned = postprocess(raw_text, det.class_name)

        # Use detection confidence as fallback when OCR produces nothing
        display_confidence = confidence if confidence > 0 else det.confidence

        # Always annotate (even rejected reads) so display shows all detections
        annotations.append(Annotation(
            bbox=det.bbox,
            class_name=det.class_name,
            text=cleaned or raw_text,
            confidence=display_confidence,
            det_confidence=det.confidence,
        ))

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

    return FrameResult(reads=reads, annotations=annotations)
