"""Super-resolution upscaling for small OCR crops."""

import os

import cv2
import numpy as np

from parking_lot.config import SRConfig


class SuperResolution:
    """Loads FSRCNN/EDSR models and provides adaptive upscaling."""

    def __init__(self, cfg: SRConfig):
        self.cfg = cfg
        self.models: dict[int, cv2.dnn_superres.DnnSuperResImpl] = {}
        if cfg.enabled:
            self._load_models()

    def _load_models(self):
        for scale in (2, 3, 4):
            pb_path = os.path.join(self.cfg.model_dir, f"FSRCNN_x{scale}.pb")
            if os.path.exists(pb_path):
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(pb_path)
                sr.setModel("fsrcnn", scale)
                self.models[scale] = sr
                print(f"  Loaded SR model: FSRCNN x{scale}")
        if not self.models:
            print("  WARNING: No SR models found, falling back to cv2.resize")

    def adaptive_upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale crop to target width using the best SR scale."""
        h, w = img.shape[:2]
        if w >= self.cfg.target_width or not self.models:
            return img
        ideal_scale = self.cfg.target_width / w
        best_scale = None
        for s in sorted(self.models.keys()):
            if s >= ideal_scale:
                best_scale = s
                break
        if best_scale is None:
            best_scale = max(self.models.keys())
        return self.models[best_scale].upsample(img)

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """Unsharp mask sharpening for better OCR edge contrast."""
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
