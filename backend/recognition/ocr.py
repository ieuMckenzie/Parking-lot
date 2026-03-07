import numpy as np
import cv2
from paddleocr import TextRecognition

from backend.config import settings
from backend.utils.logging import get_logger

log = get_logger("recognition")


class OCREngine:
    def __init__(
        self,
        model_name: str | None = None,
        padding_ratio: float | None = None,
    ):
        model_name = model_name or settings.ocr.rec_model
        self.padding_ratio = padding_ratio if padding_ratio is not None else settings.ocr.padding_ratio
        self.model = TextRecognition(model_name=model_name)
        log.info("ocr_loaded", model=model_name)

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Grayscale -> CLAHE -> unsharp mask -> resize to 48px height."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        target_h = 48
        h, w = sharpened.shape[:2]
        if h > 0:
            scale = target_h / h
            sharpened = cv2.resize(sharpened, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

        # Convert back to 3-channel for PaddleOCR
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def recognize(self, crop: np.ndarray, preprocess: bool = True) -> tuple[str, float]:
        """Run OCR on a single crop. Returns (text, confidence)."""
        if preprocess:
            crop = self.preprocess(crop)

        output = self.model.predict(input=crop, batch_size=1)
        for res in output:
            text = res["rec_text"]
            score = float(res["rec_score"])
            return text, score
        return "", 0.0

    def recognize_batch(self, crops: list[np.ndarray], preprocess: bool = True) -> list[tuple[str, float]]:
        """Run OCR on multiple crops. Returns list of (text, confidence)."""
        if not crops:
            return []

        processed = [self.preprocess(c) if preprocess else c for c in crops]
        output = self.model.predict(input=processed, batch_size=len(processed))

        results = []
        for res in output:
            results.append((res["rec_text"], float(res["rec_score"])))
        return results
