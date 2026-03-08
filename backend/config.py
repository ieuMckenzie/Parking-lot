from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CameraConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAMERA_")

    rtsp_urls: list[str] = Field(
        default=["rtsp://localhost:8554/cam1", "rtsp://localhost:8554/cam2", "rtsp://localhost:8554/cam3"],
        description="RTSP URLs for gate cameras",
    )
    capture_fps: int = Field(default=5, description="Target capture FPS per camera")


class DetectionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTION_")

    model_path: str = Field(default="yolov8n.pt", description="YOLO model weights path")
    confidence_threshold: float = Field(default=0.25, description="Minimum detection confidence")
    iou_threshold: float = Field(default=0.45, description="NMS IoU threshold")


class OCRConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_")

    det_model: str = Field(default="PP-OCRv5_mobile_det", description="PaddleOCR detection model name")
    rec_model: str = Field(default="PP-OCRv5_mobile_rec", description="PaddleOCR recognition model name")
    fallback_rec_model: str = Field(default="PP-OCRv5_server_rec", description="Fallback model when primary returns empty")
    padding_ratio: float = Field(default=0.2, description="Bbox padding ratio before OCR")
    small_crop_padding: float = Field(default=0.5, description="Extra padding for small detections (< 150px)")
    small_crop_threshold: int = Field(default=150, description="Detections smaller than this (pixels) get extra padding")


class FusionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FUSION_")

    window_seconds: float = Field(default=10.0, description="Inactivity timeout to close a track")
    min_reads: int = Field(default=3, description="Minimum reads for consensus")
    min_confidence: float = Field(default=0.6, description="Minimum total confidence for consensus")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GateVision"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'gatevision.db'}",
        description="Database connection URL",
    )
    csv_log_path: str = Field(
        default="",
        description="Path to CSV file for logging detections (empty = disabled)",
    )

    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    ocr: OCRConfig = OCRConfig()
    fusion: FusionConfig = FusionConfig()


settings = Settings()
