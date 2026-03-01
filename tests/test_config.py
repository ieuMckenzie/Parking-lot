"""Tests for parking_lot.config dataclasses."""

import os

from parking_lot.config import (
    CameraConfig,
    DetectionConfig,
    OCRConfig,
    ScannerConfig,
    SRConfig,
    ValidationConfig,
    _project_root,
)


def test_project_root_is_absolute():
    root = _project_root()
    assert os.path.isabs(root)


def test_project_root_contains_src():
    root = _project_root()
    assert os.path.isdir(os.path.join(root, "src", "parking_lot"))


def test_ocr_config_defaults():
    cfg = OCRConfig()
    assert cfg.min_height == 48
    assert cfg.thresh == 0.40
    assert cfg.lang == "en"


def test_sr_config_default_model_dir():
    cfg = SRConfig()
    assert "sr_models" in cfg.model_dir
    assert os.path.isabs(cfg.model_dir)


def test_detection_config_has_all_target_classes():
    cfg = DetectionConfig()
    for cls in cfg.target_classes:
        assert cls in cfg.class_thresholds


def test_validation_config_defaults():
    cfg = ValidationConfig()
    assert cfg.auth_refresh_interval == 5.0
    assert cfg.max_confusable_indices == 6
    assert "authorized.txt" in cfg.authorized_file


def test_camera_config_defaults():
    cfg = CameraConfig()
    assert cfg.resolution == (1920, 1080)
    assert cfg.target_fps == 20
    assert cfg.feed_enhance is True


def test_scanner_config_nests_all_sub_configs():
    cfg = ScannerConfig()
    assert isinstance(cfg.ocr, OCRConfig)
    assert isinstance(cfg.sr, SRConfig)
    assert isinstance(cfg.detection, DetectionConfig)
    assert isinstance(cfg.validation, ValidationConfig)
    assert isinstance(cfg.camera, CameraConfig)


def test_scanner_config_custom_values():
    cfg = ScannerConfig(
        model_path="/tmp/model.pt",
        sources=["usb0", "usb1"],
        min_thresh=0.7,
        num_ocr_workers=4,
    )
    assert cfg.model_path == "/tmp/model.pt"
    assert len(cfg.sources) == 2
    assert cfg.min_thresh == 0.7
    assert cfg.num_ocr_workers == 4
