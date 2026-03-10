import numpy as np
from backend.ingestion.motion import MotionDetector


def _static_frame(value=128):
    """A solid-color frame (no motion when repeated)."""
    return np.full((100, 100, 3), value, dtype=np.uint8)


def _noisy_frame(seed=42):
    """A random frame (guaranteed different from static)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)


class TestMotionDetector:
    def test_warmup_always_returns_true(self):
        md = MotionDetector(threshold=0.01, warmup_frames=5)
        frame = _static_frame()
        for _ in range(5):
            assert md.has_motion(frame) is True

    def test_no_motion_after_warmup(self):
        md = MotionDetector(threshold=0.01, warmup_frames=5)
        frame = _static_frame()
        # Burn through warmup
        for _ in range(5):
            md.has_motion(frame)
        # Same frame again — no motion
        assert md.has_motion(frame) is False

    def test_motion_detected(self):
        md = MotionDetector(threshold=0.01, warmup_frames=3)
        static = _static_frame()
        # Warmup with static background
        for _ in range(3):
            md.has_motion(static)
        # Now inject a very different frame
        moving = _noisy_frame()
        assert md.has_motion(moving) is True

    def test_threshold_sensitivity(self):
        # Very high threshold — even large changes don't trigger
        md = MotionDetector(threshold=0.99, warmup_frames=3)
        static = _static_frame()
        for _ in range(3):
            md.has_motion(static)
        moving = _noisy_frame()
        assert md.has_motion(moving) is False

    def test_roi_mask(self):
        md = MotionDetector(threshold=0.01, warmup_frames=3)
        # ROI mask: only look at top-left 10x10 corner
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :10] = 255
        md.set_roi(mask)

        static = _static_frame()
        for _ in range(3):
            md.has_motion(static)

        # Change only the bottom-right (outside ROI) — no motion
        changed = static.copy()
        changed[50:, 50:] = 0
        assert md.has_motion(changed) is False

    def test_reset(self):
        md = MotionDetector(threshold=0.01, warmup_frames=3)
        static = _static_frame()
        for _ in range(5):
            md.has_motion(static)
        md.reset()
        # After reset, should be back in warmup
        assert md.has_motion(static) is True
