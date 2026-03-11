import os
import platform

import numpy as np

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
        device: str = "auto",
    ):
        det_model = det_model or settings.ocr.det_model
        rec_model = rec_model or settings.ocr.rec_model
        self.padding_ratio = padding_ratio if padding_ratio is not None else settings.ocr.padding_ratio
        self.min_confidence = min_confidence if min_confidence is not None else settings.ocr.min_confidence

        # PaddleX model source checks can add startup latency and noisy logs.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        # Windows + PaddlePaddle 3.x can fail in oneDNN executor for OCR models.
        # Keep conservative defaults whenever runtime resolves to CPU on Windows.
        ocr_runtime_kwargs: dict[str, object] = {}
        is_windows = platform.system().lower().startswith("win")

        def _apply_windows_cpu_safety_flags() -> None:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

        if is_windows and device in {"auto", "cpu"}:
            _apply_windows_cpu_safety_flags()
            ocr_runtime_kwargs = {
                "device": "cpu",
                "enable_mkldnn": False,
                "enable_cinn": False,
                "enable_hpi": False,
            }
        elif device in {"cpu", "gpu"}:
            if device == "gpu":
                import paddle

                if not paddle.is_compiled_with_cuda():
                    log.warning("ocr_gpu_unavailable_fallback_cpu")
                    if is_windows:
                        _apply_windows_cpu_safety_flags()
                        ocr_runtime_kwargs = {
                            "device": "cpu",
                            "enable_mkldnn": False,
                            "enable_cinn": False,
                            "enable_hpi": False,
                        }
                    else:
                        ocr_runtime_kwargs = {"device": "cpu"}
                else:
                    ocr_runtime_kwargs = {"device": "gpu"}
            else:
                ocr_runtime_kwargs = {"device": "cpu"}

        from paddleocr import PaddleOCR

        self.model = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **ocr_runtime_kwargs,
        )
        runtime_device = ocr_runtime_kwargs.get("device", "auto")
        log.info("ocr_loaded", det_model=det_model, rec_model=rec_model, device=runtime_device, requested_device=device)

    def recognize(self, crop: np.ndarray, preprocess: bool = True, min_confidence: float | None = None) -> tuple[str, float]:
        """Run full OCR pipeline on a crop. Returns (text, confidence).

        The preprocess parameter is accepted for backward compatibility but is
        a no-op — PP-OCRv5 handles its own preprocessing internally.
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        output = self.model.predict(input=crop)

        texts: list[str] = []
        scores: list[float] = []

        for res in output:
            rec_texts = res["rec_texts"]
            rec_scores = res["rec_scores"]
            for text, score in zip(rec_texts, rec_scores):
                if score >= threshold and text.strip():
                    texts.append(text.strip())
                    scores.append(score)

        if not texts:
            return "", 0.0

        combined_text = " ".join(texts)
        avg_confidence = sum(scores) / len(scores)
        return combined_text, avg_confidence
