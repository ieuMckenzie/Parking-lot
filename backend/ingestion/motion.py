import cv2
import numpy as np


class MotionDetector:
    """Detects motion using MOG2 background subtraction."""

    def __init__(self, threshold: float = 0.01, warmup_frames: int = 30):
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self._frame_count = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=200, detectShadows=False,
        )
        self._roi_mask: np.ndarray | None = None

    def set_roi(self, mask: np.ndarray) -> None:
        """Set a binary mask (255=include, 0=ignore) for the region of interest."""
        self._roi_mask = mask

    def has_motion(self, frame: np.ndarray) -> bool:
        """Return True if motion exceeds threshold (or during warmup)."""
        self._frame_count += 1
        fg_mask = self._subtractor.apply(frame)

        if self._frame_count <= self.warmup_frames:
            return True

        if self._roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, self._roi_mask)

        motion_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        return bool(motion_ratio >= self.threshold)

    def reset(self) -> None:
        """Reset the background model and frame count."""
        self._frame_count = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=200, detectShadows=False,
        )
