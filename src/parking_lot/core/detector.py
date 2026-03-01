"""YOLO-based object detector for truck logistics targets."""

import os
import sys

import numpy as np
from ultralytics import YOLO

from parking_lot.config import DetectionConfig


class YOLODetector:
    """Loads a YOLO model and runs inference on frames."""

    def __init__(self, model_path: str, cfg: DetectionConfig, min_thresh: float = 0.5):
        if not os.path.exists(model_path):
            sys.exit(f"ERROR: Model path is invalid: {model_path}")
        print("Initializing YOLO...")
        self.model = YOLO(model_path, task="detect")
        self.labels = self.model.names
        self.cfg = cfg
        self.min_thresh = min_thresh

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO inference on a frame.

        Returns list of detection dicts with keys: rect, bid, label, conf.
        """
        results = self.model(frame, verbose=False, iou=0.45)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()
            cls_id = int(box.cls)
            raw_label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"
            label_key = raw_label.lower().replace(" ", "").replace("_", "")
            req_conf = self.cfg.class_thresholds.get(label_key, self.min_thresh)

            if conf > req_conf:
                if any(t in label_key for t in self.cfg.target_classes) or label_key in self.cfg.target_classes:
                    bid = f"{x1}_{y1}"
                    detections.append({
                        "rect": (x1, y1, x2, y2),
                        "bid": bid,
                        "label": label_key,
                        "conf": conf,
                    })

        return detections

    def compute_crop(self, frame: np.ndarray, det: dict) -> np.ndarray | None:
        """Extract a padded crop from a frame given a detection dict."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det["rect"]

        crop_h = y2 - y1
        if crop_h < 30:
            pad_x_pct, pad_y_pct = 0.40, 0.35
        elif crop_h < 60:
            pad_x_pct, pad_y_pct = 0.30, 0.25
        else:
            pad_x_pct, pad_y_pct = 0.20, 0.15

        if det["label"] in ("licenseplate", "containerplate"):
            pad_x_pct += 0.08
            pad_y_pct += 0.07

        px = int((x2 - x1) * pad_x_pct)
        py = int((y2 - y1) * pad_y_pct)
        crop = frame[max(0, y1 - py): min(h, y2 + py), max(0, x1 - px): min(w, x2 + px)]

        if crop.size > 0:
            return crop
        return None
