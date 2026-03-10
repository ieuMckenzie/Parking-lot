import numpy as np
from unittest.mock import MagicMock, patch

from backend.recognition.ocr import OCREngine


class FakePredictResult:
    """Mimics a single PaddleOCR predict result item."""
    def __init__(self, lines):
        """lines: list of (text, score) tuples."""
        self._lines = lines

    def __getitem__(self, key):
        if key == "rec_text":
            return [t for t, s in self._lines]
        if key == "rec_score":
            return [s for t, s in self._lines]
        raise KeyError(key)


def _make_engine():
    """Create OCREngine with model loading bypassed."""
    engine = object.__new__(OCREngine)
    engine.padding_ratio = 0.2
    engine.min_confidence = 0.5
    engine.model = MagicMock()
    return engine


class TestRecognize:
    def test_single_line_above_threshold(self):
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([("ABC1234", 0.95)]),
        ]

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == "ABC1234"
        assert conf == 0.95

    def test_multi_line_concatenated(self):
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([("USDOT", 0.9), ("1234567", 0.85)]),
        ]

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == "USDOT 1234567"
        assert abs(conf - 0.875) < 0.01  # average of 0.9 and 0.85

    def test_filters_below_threshold(self):
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([("GOOD", 0.8), ("GARBAGE", 0.2)]),
        ]

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == "GOOD"
        assert conf == 0.8

    def test_all_below_threshold_returns_empty(self):
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([("JUNK", 0.1), ("NOISE", 0.3)]),
        ]

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == ""
        assert conf == 0.0

    def test_empty_prediction_returns_empty(self):
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([]),
        ]

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == ""
        assert conf == 0.0

    def test_no_results_returns_empty(self):
        engine = _make_engine()
        engine.model.predict.return_value = []

        text, conf = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == ""
        assert conf == 0.0

    def test_preprocess_param_ignored(self):
        """preprocess parameter exists for backward compat but is a no-op."""
        engine = _make_engine()
        engine.model.predict.return_value = [
            FakePredictResult([("TEXT", 0.9)]),
        ]

        text1, _ = engine.recognize(np.zeros((50, 50, 3), dtype=np.uint8), preprocess=True)
        engine.model.predict.reset_mock()
        engine.model.predict.return_value = [
            FakePredictResult([("TEXT", 0.9)]),
        ]
        text2, _ = engine.recognize(np.zeros((50, 50, 3), dtype=np.uint8), preprocess=False)
        assert text1 == text2
