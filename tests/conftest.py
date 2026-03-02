"""Shared test fixtures."""

import os
import queue
import tempfile
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from parking_lot.config import (
    CameraConfig,
    DetectionConfig,
    OCRConfig,
    ScannerConfig,
    SRConfig,
    ValidationConfig,
)
from parking_lot.core.validation import TextValidator
from parking_lot.engine.events import EventBus
from parking_lot.engine.logger import CSVLogger
from parking_lot.engine.state import SharedState


@pytest.fixture
def detection_cfg():
    return DetectionConfig()


@pytest.fixture
def validation_cfg(tmp_path):
    """ValidationConfig pointing at a temp authorized file."""
    auth_file = tmp_path / "authorized.txt"
    auth_file.write_text("8XST826\n6ZDT416\n8ZKNO39\n8ANK512\n")
    return ValidationConfig(authorized_file=str(auth_file))


@pytest.fixture
def validator(validation_cfg):
    v = TextValidator(validation_cfg)
    v.refresh_authorized()
    return v


@pytest.fixture
def state(detection_cfg):
    return SharedState(detection_cfg)


@pytest.fixture
def ocr_cfg():
    return OCRConfig()


@pytest.fixture
def csv_logger(tmp_path):
    path = str(tmp_path / "test_log.csv")
    return CSVLogger(filename=path)


@pytest.fixture
def dummy_frame():
    """A 200x300 BGR dummy frame."""
    return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def small_crop():
    """A small 20x40 BGR crop that triggers upscaling."""
    return np.random.randint(0, 255, (20, 40, 3), dtype=np.uint8)


@pytest.fixture
def mock_engine(tmp_path):
    """A mock ScannerEngine for API tests — no real YOLO/OCR needed."""
    engine = MagicMock()
    engine.cfg = ScannerConfig(
        model_path="fake.pt",
        sources=["usb0"],
        camera=CameraConfig(),
        sr=SRConfig(enabled=False),
    )
    engine.cameras = {"cam0": MagicMock()}
    engine.get_uptime.return_value = 42.5
    engine.get_queue_sizes.return_value = {"yolo_queue": 0, "ocr_queue": 1}
    engine.get_camera_status.return_value = [
        {"id": "cam0", "source": "usb0", "connected": True, "fps": 15.3},
    ]
    engine.get_snapshot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # State
    engine.state = SharedState(DetectionConfig())
    engine.state.update_detections("cam0", [
        {"rect": (10, 20, 100, 80), "bid": "10_20", "label": "licenseplate", "conf": 0.92},
    ])

    # Logger
    logger = CSVLogger(filename=str(tmp_path / "test_log.csv"))
    logger.log("cam0", "ABC1234", "licenseplate", 0.95)
    engine.logger = logger

    # Validator
    auth_file = tmp_path / "authorized.txt"
    auth_file.write_text("ABC1234\n")
    engine.cfg.validation.authorized_file = str(auth_file)
    engine.validator = TextValidator(ValidationConfig(authorized_file=str(auth_file)))
    engine.validator.refresh_authorized()

    # Event bus
    engine.events = EventBus()

    engine.stop = MagicMock()
    return engine
