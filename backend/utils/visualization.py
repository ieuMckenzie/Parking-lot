"""Draw detection annotations and status overlays on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from backend.fusion.pipeline import Annotation

# BGR colors per detection class
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "LicensePlate": (0, 200, 0),       # green
    "USDOT": (200, 100, 0),            # blue
    "TrailerNum": (0, 150, 255),        # orange
    "ContainerNum": (180, 50, 180),     # purple
    "ContainerPlate": (200, 200, 0),    # cyan
}
DEFAULT_COLOR = (200, 200, 200)  # gray


def draw_annotations(
    frame: np.ndarray,
    annotations: list[Annotation],
) -> np.ndarray:
    """Draw bounding boxes and OCR labels on a frame. Returns the annotated copy."""
    out = frame.copy()

    for ann in annotations:
        color = CLASS_COLORS.get(ann.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = ann.bbox

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: "ClassName: TEXT (0.95)"
        label = f"{ann.class_name}: {ann.text} ({ann.confidence:.2f})"

        # Background rectangle for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(out, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


def draw_status(
    frame: np.ndarray,
    decision: str | None = None,
    fps: float | None = None,
    track_reads: int = 0,
) -> np.ndarray:
    """Draw status bar at top of frame with FPS, track info, and decision banner."""
    out = frame
    h, w = out.shape[:2]

    # Semi-transparent status bar at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # FPS counter
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Track read count
    if track_reads > 0:
        cv2.putText(out, f"Reads: {track_reads}", (w // 2 - 40, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Decision banner
    if decision:
        banner_colors = {
            "APPROVED": (0, 180, 0),
            "DENIED": (0, 0, 200),
            "FLAGGED": (0, 180, 255),
        }
        banner_color = banner_colors.get(decision, (128, 128, 128))
        banner_h = 50
        banner_y = h - banner_h
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, banner_y), (w, h), banner_color, -1)
        cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)
        cv2.putText(out, decision, (w // 2 - 80, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return out
