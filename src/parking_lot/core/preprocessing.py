"""Image preprocessing functions for OCR and display."""

import cv2
import numpy as np

from parking_lot.config import OCRConfig


def preprocess_for_ocr(img: np.ndarray, cfg: OCRConfig, use_binarization: bool = False) -> np.ndarray:
    """Heavy preprocessing for difficult images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = gray.shape[0]
    if height < cfg.min_height:
        scale = cfg.min_height / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cfg.interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if use_binarization:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return gray


def preprocess_light(img: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    """Minimal preprocessing: grayscale + resize. Best for PaddleOCR v4."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    if height < cfg.min_height:
        scale = cfg.min_height / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cfg.interp)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return gray


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Visualization-only enhancement. Does not affect OCR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)
