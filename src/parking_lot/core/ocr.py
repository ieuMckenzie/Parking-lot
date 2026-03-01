"""PaddleOCR wrapper with multi-strategy pipeline."""

import re

import cv2
import numpy as np
from paddleocr import PaddleOCR

from parking_lot.config import OCRConfig
from parking_lot.core.preprocessing import preprocess_for_ocr, preprocess_light


class OCRReader:
    """Wraps PaddleOCR with a 3-strategy read pipeline."""

    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        print(f"Initializing PaddleOCR (lang={cfg.lang})...")
        self.reader = PaddleOCR(lang=cfg.lang)

    def read_text(self, processed_img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """Run PaddleOCR on a preprocessed image.

        Returns list of (box, text, confidence) tuples.
        """
        if len(processed_img.shape) == 2:
            img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            img = processed_img

        result = self.reader.ocr(img)
        if not result or result[0] is None:
            return []

        r0 = result[0]
        return self._parse_v3(r0) or self._parse_v2(r0)

    def run_pipeline(self, img_crop: np.ndarray, class_type: str) -> list[tuple[np.ndarray, str, float]]:
        """Run 3-strategy OCR pipeline on a crop.

        Strategy 1: Light preprocessing (fastest)
        Strategy 2: Heavy CLAHE preprocessing (if light is low-confidence)
        Strategy 3: Binarization (last resort for plates)
        """
        # Strategy 1: Light
        processed_light = preprocess_light(img_crop, self.cfg)
        results = self.read_text(processed_light)
        avg_conf = _avg_conf(results)

        # Strategy 2: Heavy (only if needed)
        if not results or avg_conf < 0.85:
            processed = preprocess_for_ocr(img_crop, self.cfg, use_binarization=False)
            results_heavy = self.read_text(processed)
            if _avg_conf(results_heavy) > avg_conf:
                results = results_heavy

        # Strategy 3: Binary (last resort for plates)
        is_plate = class_type in ("licenseplate", "containerplate")
        if is_plate and (not results or _avg_conf(results) < 0.60):
            processed_bin = preprocess_for_ocr(img_crop, self.cfg, use_binarization=True)
            results_bin = self.read_text(processed_bin)
            if _avg_conf(results_bin) > _avg_conf(results):
                results = results_bin

        return results

    def _parse_v3(self, r0) -> list[tuple[np.ndarray, str, float]]:
        """Parse PaddleOCR 3.x dict-style output."""
        def _get(key, default=None):
            if hasattr(r0, "get") and callable(r0.get):
                return r0.get(key, default)
            return getattr(r0, key, default)

        texts = _get("rec_texts") or _get("rec_text")
        if texts is None:
            return []

        if hasattr(texts, "__iter__") and not isinstance(texts, str):
            texts = list(texts)
        else:
            texts = [texts] if texts else []

        if not texts:
            return []

        scores = _get("rec_scores") or _get("rec_score")
        if scores is None or (hasattr(scores, "__len__") and len(scores) != len(texts)):
            scores = [0.5] * len(texts)
        else:
            scores = list(scores)

        polys = _get("dt_polys") or _get("rec_polys") or _get("rec_boxes")
        if polys is not None and hasattr(polys, "__len__") and len(polys) == len(texts):
            polys = list(polys)
        else:
            polys = None

        dummy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        out = []
        for i, text in enumerate(texts):
            text = str(text).strip() if text else ""
            if not text:
                continue
            try:
                conf = float(scores[i]) if i < len(scores) else 0.5
            except (ValueError, TypeError, IndexError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            if polys is not None and i < len(polys):
                try:
                    box = np.array(polys[i], dtype=np.float64)
                    if box.ndim == 1 and box.size >= 4:
                        box = box.reshape(-1, 2)
                    if box.ndim < 2 or box.size < 4:
                        box = dummy.copy()
                except (ValueError, TypeError):
                    box = dummy.copy()
            else:
                box = dummy.copy()
            out.append((box, text, conf))
        return out

    def _parse_v2(self, r0) -> list[tuple[np.ndarray, str, float]]:
        """Parse PaddleOCR 2.x list-style output."""
        out = []
        for line in list(r0):
            if not line or len(line) < 2:
                continue
            try:
                box = np.array(line[0], dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if box.ndim == 1 and box.size >= 4:
                box = box.reshape(-1, 2)
            if box.ndim < 2 or box.size < 4:
                continue
            second = line[1]
            if isinstance(second, (list, tuple)) and len(second) >= 2:
                text = str(second[0]).strip()
                try:
                    conf = float(second[1])
                except (ValueError, TypeError):
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))
            else:
                text, conf = str(second).strip(), 0.5
            out.append((box, text, conf))
        return out


def stitch_results(results: list[tuple[np.ndarray, str, float]]) -> list[tuple[str, float]]:
    """Sort OCR results by position and produce stitched + individual candidates."""
    by_y = sorted(results, key=_bbox_y0)
    stitched = ""
    for _, text, _ in by_y:
        stitched += re.sub(r"[^A-Z0-9\-]", "", text.upper())

    candidates = [(stitched, _avg_conf(results))]
    for _, text, conf in results:
        candidates.append((re.sub(r"[^A-Z0-9\-]", "", text.upper()), conf))
    return candidates


def _avg_conf(results: list[tuple]) -> float:
    if not results:
        return 0.0
    return sum(c for _, _, c in results) / len(results)


def _bbox_y0(item):
    b = item[0]
    try:
        return float(b[0][1]) if getattr(b, "ndim", 2) >= 2 and len(b) > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0
