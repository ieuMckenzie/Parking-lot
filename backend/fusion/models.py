from dataclasses import dataclass
from enum import Enum


class FusionStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class Read:
    """A single OCR read from one frame/camera."""

    text: str
    raw_text: str
    confidence: float
    class_name: str
    camera_id: str
    timestamp: float


@dataclass
class FusionResult:
    """Result of fusing multiple reads for one text class."""

    class_name: str
    value: str
    confidence: float
    num_reads: int
    consensus_ratio: float
    status: FusionStatus
