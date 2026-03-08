import numpy as np
import cv2
from paddleocr import TextRecognition

from backend.config import settings
from backend.utils.logging import get_logger

log = get_logger("recognition")

# Mean brightness below which we skip grayscale preprocessing
# (dark crops lose all contrast when converted to grayscale + CLAHE)
_DARK_THRESHOLD = 80


class OCREngine:
    def __init__(
        self,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
        padding_ratio: float | None = None,
    ):
        model_name = model_name or settings.ocr.rec_model
        self.padding_ratio = padding_ratio if padding_ratio is not None else settings.ocr.padding_ratio
        self.model = TextRecognition(model_name=model_name)
        log.info("ocr_loaded", model=model_name)

        # Lazy-load fallback (server model) only when needed
        self._fallback_model_name = fallback_model_name or settings.ocr.fallback_rec_model
        self._fallback_model: TextRecognition | None = None

    def _get_fallback(self) -> TextRecognition | None:
        if not self._fallback_model_name:
            return None
        if self._fallback_model is None:
            self._fallback_model = TextRecognition(model_name=self._fallback_model_name)
            log.info("ocr_fallback_loaded", model=self._fallback_model_name)
        return self._fallback_model

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for OCR. Skips grayscale pipeline on dark crops."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()

        if mean_brightness < _DARK_THRESHOLD:
            # Dark crop: boost brightness/contrast instead of CLAHE
            # which tends to amplify noise in dark images
            bright = cv2.convertScaleAbs(gray, alpha=2.0, beta=40)
            enhanced = bright
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        target_h = 48
        h, w = sharpened.shape[:2]
        if h > 0:
            scale = target_h / h
            sharpened = cv2.resize(sharpened, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _run_model(self, model: TextRecognition, crop: np.ndarray) -> tuple[str, float]:
        output = model.predict(input=crop, batch_size=1)
        for res in output:
            return res["rec_text"], float(res["rec_score"])
        return "", 0.0

    def recognize(self, crop: np.ndarray, preprocess: bool = True) -> tuple[str, float]:
        """Run OCR on a single crop. Falls back to server model if primary returns empty."""
        processed = self.preprocess(crop) if preprocess else crop

        text, score = self._run_model(self.model, processed)
        if text and score > 0:
            return text, score

        # Fallback: try server model
        fallback = self._get_fallback()
        if fallback is not None:
            text, score = self._run_model(fallback, processed)
            if text and score > 0:
                log.debug("ocr_fallback_used", text=text, score=score)
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
