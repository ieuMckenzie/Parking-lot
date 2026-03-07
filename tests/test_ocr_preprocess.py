import numpy as np
import cv2

from backend.recognition.ocr import OCREngine


class TestPreprocess:
    """Test OCREngine.preprocess without loading the model."""

    @staticmethod
    def _make_engine():
        """Create an OCREngine with model loading bypassed."""
        engine = object.__new__(OCREngine)
        engine.padding_ratio = 0.2
        return engine

    def test_output_shape_height_48(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        assert result.shape[0] == 48

    def test_output_preserves_aspect_ratio(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        # 200/100 = 2.0 aspect ratio, scaled to h=48 → w=96
        assert result.shape[0] == 48
        assert result.shape[1] == 96

    def test_output_is_3_channel(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_dtype_uint8(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        assert result.dtype == np.uint8

    def test_small_crop_scales_up(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (10, 30, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        assert result.shape[0] == 48
        # 30/10 * 48 = 144
        assert result.shape[1] == 144

    def test_already_48_height(self):
        engine = self._make_engine()
        crop = np.random.randint(0, 255, (48, 120, 3), dtype=np.uint8)
        result = engine.preprocess(crop)
        assert result.shape[0] == 48
        assert result.shape[1] == 120
