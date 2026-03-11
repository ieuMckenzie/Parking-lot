import numpy as np
from paddleocr import PaddleOCR

from backend.config import settings
from backend.utils.logging import get_logger

log = get_logger("recognition")


class OCREngine:
    def __init__(
        self,
        det_model: str | None = None,
        rec_model: str | None = None,
        padding_ratio: float | None = None,
        min_confidence: float | None = None,
    ):
        det_model = det_model or settings.ocr.det_model
        rec_model = rec_model or settings.ocr.rec_model
        self.padding_ratio = padding_ratio if padding_ratio is not None else settings.ocr.padding_ratio
        self.min_confidence = min_confidence if min_confidence is not None else settings.ocr.min_confidence

        self.model = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        log.info("ocr_loaded", det_model=det_model, rec_model=rec_model)

    def recognize(self, crop: np.ndarray, preprocess: bool = True) -> tuple[str, float]:
        """Run full OCR pipeline on a crop. Returns (text, confidence).

        The preprocess parameter is accepted for backward compatibility but is
        a no-op — PP-OCRv5 handles its own preprocessing internally.
        """
        output = self.model.predict(input=crop)

        texts: list[str] = []
        scores: list[float] = []

        for res in output:
            rec_texts = res["rec_texts"]
            rec_scores = res["rec_scores"]
            for text, score in zip(rec_texts, rec_scores):
                if score >= self.min_confidence and text.strip():
                    texts.append(text.strip())
                    scores.append(score)

        if not texts:
            return "", 0.0

        combined_text = " ".join(texts)
        avg_confidence = sum(scores) / len(scores)
        return combined_text, avg_confidence
