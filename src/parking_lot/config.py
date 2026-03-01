"""Central configuration for the parking lot scanner."""

import os
from dataclasses import dataclass, field

import cv2


def _project_root() -> str:
    """Return the project root (two levels up from this file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class OCRConfig:
    min_height: int = 48
    use_binarization: bool = False
    mag_ratio: float = 1.5
    best_effort_min_conf: float = 0.30
    interp: int = cv2.INTER_LINEAR
    thresh: float = 0.40
    lang: str = "en"


@dataclass
class SRConfig:
    target_width: int = 280
    model_dir: str = field(default_factory=lambda: os.path.join(_project_root(), "models", "paddle", "sr_models"))
    enabled: bool = True


@dataclass
class DetectionConfig:
    similarity_thresh: float = 0.60
    log_cooldown: float = 60.0
    detection_hold_time: float = 0.5
    stability_votes_required: int = 10
    class_thresholds: dict = field(default_factory=lambda: {
        "usdot": 0.50,
        "trailernum": 0.60,
        "containernum": 0.40,
        "containerplate": 0.70,
        "licenseplate": 0.70,
    })
    target_classes: list = field(default_factory=lambda: [
        "usdot", "trailernum", "containerplate", "licenseplate", "containernum",
    ])


@dataclass
class ValidationConfig:
    plate_regex: str = r"^[A-Z0-9\-]{4,15}$"
    container_regex: str = r"^[A-Z]{4}[0-9]{6,7}$"
    trailer_regex: str = r"^[A-Z0-9\-]{1,10}$"
    authorized_file: str = field(default_factory=lambda: os.path.join(_project_root(), "data", "authorized.txt"))
    auth_refresh_interval: float = 5.0
    max_confusable_indices: int = 6


@dataclass
class CameraConfig:
    resolution: tuple = (1920, 1080)
    target_fps: int = 20
    feed_enhance: bool = True


@dataclass
class ScannerConfig:
    """Top-level config aggregating all sub-configs."""
    model_path: str = ""
    sources: list = field(default_factory=list)
    min_thresh: float = 0.5
    num_ocr_workers: int = 2
    ocr_debug: bool = False
    use_gpu: bool = False
    record: bool = False
    grid_cols: int | None = None

    ocr: OCRConfig = field(default_factory=OCRConfig)
    sr: SRConfig = field(default_factory=SRConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
