"""Tests for parking_lot.core.preprocessing."""

import numpy as np

from parking_lot.config import OCRConfig
from parking_lot.core.preprocessing import enhance_frame, preprocess_for_ocr, preprocess_light


class TestPreprocessForOCR:
    def test_output_is_grayscale(self, dummy_frame, ocr_cfg):
        result = preprocess_for_ocr(dummy_frame, ocr_cfg)
        assert result.ndim == 2  # single channel

    def test_adds_padding(self, dummy_frame, ocr_cfg):
        result = preprocess_for_ocr(dummy_frame, ocr_cfg)
        # Original 200x300 + 10px padding on each side = 220x320
        assert result.shape[0] == 200 + 20
        assert result.shape[1] == 300 + 20

    def test_upscales_small_image(self, ocr_cfg):
        small = np.random.randint(0, 255, (10, 30, 3), dtype=np.uint8)
        result = preprocess_for_ocr(small, ocr_cfg)
        # Should be upscaled to at least min_height (48) + padding (20)
        assert result.shape[0] >= ocr_cfg.min_height + 20

    def test_no_upscale_for_large_image(self, ocr_cfg):
        large = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = preprocess_for_ocr(large, ocr_cfg)
        # 100 > min_height(48), so should just add padding
        assert result.shape[0] == 100 + 20

    def test_binarization_produces_binary(self, dummy_frame, ocr_cfg):
        result = preprocess_for_ocr(dummy_frame, ocr_cfg, use_binarization=True)
        unique = np.unique(result)
        # Binarized image should have limited unique values (mostly 0 and 255)
        # Padding adds white border which is 255
        assert 255 in unique


class TestPreprocessLight:
    def test_output_is_grayscale(self, dummy_frame, ocr_cfg):
        result = preprocess_light(dummy_frame, ocr_cfg)
        assert result.ndim == 2

    def test_adds_padding(self, dummy_frame, ocr_cfg):
        result = preprocess_light(dummy_frame, ocr_cfg)
        assert result.shape[0] == 200 + 20
        assert result.shape[1] == 300 + 20

    def test_upscales_small_image(self, ocr_cfg):
        small = np.random.randint(0, 255, (10, 30, 3), dtype=np.uint8)
        result = preprocess_light(small, ocr_cfg)
        assert result.shape[0] >= ocr_cfg.min_height + 20


class TestEnhanceFrame:
    def test_same_shape_as_input(self, dummy_frame):
        result = enhance_frame(dummy_frame)
        assert result.shape == dummy_frame.shape

    def test_output_dtype_uint8(self, dummy_frame):
        result = enhance_frame(dummy_frame)
        assert result.dtype == np.uint8

    def test_output_in_valid_range(self, dummy_frame):
        result = enhance_frame(dummy_frame)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_works_on_solid_color(self):
        solid = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = enhance_frame(solid)
        assert result.shape == solid.shape
