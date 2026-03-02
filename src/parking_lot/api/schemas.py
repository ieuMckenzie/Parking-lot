"""Pydantic request/response models for the API."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    yolo_queue_size: int
    ocr_queue_size: int
    num_cameras: int
    sse_subscribers: int


class CameraStatus(BaseModel):
    id: str
    source: str
    connected: bool
    fps: float


class CameraListResponse(BaseModel):
    cameras: list[CameraStatus]


class Detection(BaseModel):
    rect: tuple[int, int, int, int]
    bid: str
    label: str
    conf: float


class DetectionsResponse(BaseModel):
    detections: dict[str, list[Detection]]


class LogEntry(BaseModel):
    Timestamp: str
    Camera_ID: str
    Value: str
    Data_Type: str
    Confidence: str


class LogResponse(BaseModel):
    entries: list[LogEntry]
    total: int


class ConfigResponse(BaseModel):
    min_thresh: float
    ocr_thresh: float
    similarity_thresh: float
    log_cooldown: float
    detection_hold_time: float
    sr_enabled: bool
    sr_target_width: int
    feed_enhance: bool
    num_ocr_workers: int


class ThresholdsUpdate(BaseModel):
    min_thresh: float | None = None
    ocr_thresh: float | None = None
    similarity_thresh: float | None = None
    log_cooldown: float | None = None


class AuthorizedPlateRequest(BaseModel):
    plate: str


class AuthorizedListResponse(BaseModel):
    plates: list[str]


class MessageResponse(BaseModel):
    message: str
