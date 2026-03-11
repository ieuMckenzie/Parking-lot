from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from backend.config import settings
from backend.utils.logging import get_logger

log = get_logger("detection")


@dataclass
class Detection:
    class_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def pad(self, ratio: float, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return padded bbox clamped to frame dimensions. frame_shape is (height, width)."""
        x1, y1, x2, y2 = self.bbox
        pad_w = int(self.width * ratio)
        pad_h = int(self.height * ratio)
        h, w = frame_shape[:2]
        return (
            max(0, x1 - pad_w),
            max(0, y1 - pad_h),
            min(w, x2 + pad_w),
            min(h, y2 + pad_h),
        )


class Detector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        device: str = "auto",
    ):
        model_path = model_path or settings.detection.model_path
        self.model = YOLO(str(model_path))
        self.confidence = confidence or settings.detection.confidence_threshold
        self.iou = iou or settings.detection.iou_threshold
        self.device = device
        self._resolved_device = "cpu" if device == "cpu" else ("0" if device == "gpu" else "auto")
        self.class_names: dict[int, str] = self.model.names
        log.info("detector_loaded", model=str(model_path), classes=self.class_names, device=self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        infer_kwargs = {"conf": self.confidence, "iou": self.iou, "verbose": False}
        if self._resolved_device != "auto":
            infer_kwargs["device"] = self._resolved_device

        try:
            results = self.model(frame, **infer_kwargs)
        except ValueError as exc:
            # Ultralytics raises ValueError when CUDA is requested but unavailable.
            if self._resolved_device == "0" and ("Invalid CUDA" in str(exc) or not torch.cuda.is_available()):
                log.warning("detector_gpu_unavailable_fallback_cpu")
                self._resolved_device = "cpu"
                infer_kwargs["device"] = "cpu"
                results = self.model(frame, **infer_kwargs)
            else:
                raise
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                detections.append(
                    Detection(
                        class_name=self.class_names[cls_id],
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                    )
                )
        return detections
