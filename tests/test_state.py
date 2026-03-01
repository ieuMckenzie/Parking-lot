"""Tests for parking_lot.engine.state.SharedState."""

import threading
import time

from parking_lot.config import DetectionConfig
from parking_lot.engine.state import SharedState


SAMPLE_DET = {"rect": (10, 20, 100, 80), "bid": "10_20", "label": "licenseplate", "conf": 0.9}


class TestUpdateAndGetDetections:
    def test_store_and_retrieve(self, state):
        state.update_detections("cam0", [SAMPLE_DET])
        result = state.get_detections("cam0")
        assert len(result) == 1
        assert result[0]["label"] == "licenseplate"

    def test_empty_returns_empty(self, state):
        assert state.get_detections("cam99") == []

    def test_overwrite_detections(self, state):
        state.update_detections("cam0", [SAMPLE_DET])
        new_det = {"rect": (50, 50, 150, 150), "bid": "50_50", "label": "usdot", "conf": 0.8}
        state.update_detections("cam0", [new_det])
        result = state.get_detections("cam0")
        assert len(result) == 1
        assert result[0]["label"] == "usdot"

    def test_clear_detections(self, state):
        state.update_detections("cam0", [SAMPLE_DET])
        state.update_detections("cam0", [])
        # With hold time, might still return old detections briefly
        # Wait for hold time to expire
        time.sleep(state.cfg.detection_hold_time + 0.1)
        assert state.get_detections("cam0") == []


class TestDetectionHoldTime:
    def test_hold_time_returns_last_nonempty(self):
        cfg = DetectionConfig(detection_hold_time=0.5)
        state = SharedState(cfg)
        state.update_detections("cam0", [SAMPLE_DET])
        state.update_detections("cam0", [])
        # Within hold time — should still see old detections
        result = state.get_detections("cam0")
        assert len(result) == 1

    def test_hold_time_expires(self):
        cfg = DetectionConfig(detection_hold_time=0.05)
        state = SharedState(cfg)
        state.update_detections("cam0", [SAMPLE_DET])
        state.update_detections("cam0", [])
        time.sleep(0.1)
        assert state.get_detections("cam0") == []


class TestStaticCameras:
    def test_static_camera_ignores_empty_update(self, state, tmp_path):
        # Create a real file to simulate a static source
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake image data")

        state.set_static_cameras(["img0"], [str(img_file)])
        state.update_detections("img0", [SAMPLE_DET])
        # Empty update on static camera should be ignored
        state.update_detections("img0", [])
        assert len(state.get_detections("img0")) == 1


class TestSimilarityCheck:
    def test_similar_plate_detected(self, state):
        now = time.time()
        state.update_seen_plate("ABC1234", now)
        is_similar, match = state.is_similar_to_recent("ABC1234", now + 1)
        assert is_similar is True
        assert match == "ABC1234"

    def test_different_plate_not_similar(self, state):
        now = time.time()
        state.update_seen_plate("ABC1234", now)
        is_similar, _ = state.is_similar_to_recent("XYZ9999", now + 1)
        assert is_similar is False

    def test_expired_plate_not_similar(self):
        cfg = DetectionConfig(log_cooldown=1.0)
        state = SharedState(cfg)
        now = time.time()
        state.update_seen_plate("ABC1234", now - 2.0)  # 2 seconds ago
        is_similar, _ = state.is_similar_to_recent("ABC1234", now)
        assert is_similar is False

    def test_close_match_is_similar(self, state):
        now = time.time()
        state.update_seen_plate("ABC1234", now)
        # Off by one char — should still be similar (ratio ~0.86)
        is_similar, _ = state.is_similar_to_recent("ABC1235", now + 1)
        assert is_similar is True


class TestPlateVotes:
    def test_first_vote_returns_one(self, state):
        assert state.add_plate_vote("ABC1234", "ABC1234") == 1

    def test_votes_accumulate(self, state):
        state.add_plate_vote("ABC1234", "ABC1234")
        assert state.add_plate_vote("ABC1234", "ABC1234") == 2
        assert state.add_plate_vote("ABC1234", "ABC1234") == 3

    def test_different_candidates_tracked_separately(self, state):
        state.add_plate_vote("ABC1234", "ABC1234")
        state.add_plate_vote("ABC1234", "ABC1235")
        assert state.add_plate_vote("ABC1234", "ABC1234") == 2
        assert state.add_plate_vote("ABC1234", "ABC1235") == 2


class TestDisplayText:
    def test_set_and_get(self, state):
        state.set_display_text("cam0", "10_20", "ABC1234", (0, 255, 0))
        text, color = state.get_display_text("cam0", "10_20")
        assert text == "ABC1234"
        assert color == (0, 255, 0)

    def test_missing_returns_none(self, state):
        text, color = state.get_display_text("cam0", "missing")
        assert text is None
        assert color is None


class TestSeenPlates:
    def test_update_and_get(self, state):
        now = time.time()
        state.update_seen_plate("ABC1234", now)
        plates = state.get_seen_plates()
        assert "ABC1234" in plates

    def test_remove_seen_plate(self, state):
        state.update_seen_plate("ABC1234", time.time())
        state.remove_seen_plate("ABC1234")
        assert "ABC1234" not in state.get_seen_plates()

    def test_remove_nonexistent_no_crash(self, state):
        state.remove_seen_plate("DOESNOTEXIST")  # should not raise


class TestGetAllDetections:
    def test_returns_snapshot(self, state):
        state.update_detections("cam0", [SAMPLE_DET])
        state.update_detections("cam1", [])
        all_dets = state.get_all_detections()
        assert "cam0" in all_dets
        assert len(all_dets["cam0"]) == 1


class TestThreadSafety:
    def test_concurrent_updates_no_crash(self, state):
        """Hammer the state from multiple threads to check for deadlocks/crashes."""
        errors = []

        def writer(cam_id):
            try:
                for _ in range(100):
                    state.update_detections(cam_id, [SAMPLE_DET])
                    state.set_display_text(cam_id, "10_20", "ABC", (255, 255, 255))
                    state.update_seen_plate(f"PLATE-{cam_id}", time.time())
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    state.get_detections("cam0")
                    state.get_display_text("cam0", "10_20")
                    state.is_similar_to_recent("ABC1234", time.time())
                    state.get_all_detections()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=writer, args=(f"cam{i}",)))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"
