import cv2
import numpy as np
import pytest


def _create_image(path, color=(255, 0, 0)):
    """Helper: write a small solid-color image."""
    img = np.full((100, 100, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestImageFolderCamera:
    def test_reads_images_in_sorted_order(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        _create_image(tmp_path / "b.png", (0, 255, 0))
        _create_image(tmp_path / "a.png", (0, 0, 255))
        _create_image(tmp_path / "c.png", (255, 0, 0))

        cam = ImageFolderCamera(folder=tmp_path, camera_id="test")
        cam.start()

        frames = []
        while True:
            result = cam.read()
            if result is None:
                break
            frame, ts = result
            frames.append((frame, ts))

        assert len(frames) == 3
        # Sorted order: a.png (idx 0), b.png (idx 1), c.png (idx 2)
        assert frames[0][1] == 0.0  # timestamp = index / fps
        assert frames[1][1] == pytest.approx(1.0)
        assert frames[2][1] == pytest.approx(2.0)
        # a.png is red (BGR: 0,0,255)
        assert frames[0][0][50, 50, 2] == 255

    def test_skips_non_image_files(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        _create_image(tmp_path / "a.png")
        (tmp_path / "readme.txt").write_text("not an image")

        cam = ImageFolderCamera(folder=tmp_path, camera_id="test")
        cam.start()

        result = cam.read()
        assert result is not None
        assert cam.read() is None  # only one image

    def test_empty_folder(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        cam = ImageFolderCamera(folder=tmp_path, camera_id="test")
        cam.start()
        assert cam.read() is None

    def test_active_property(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        _create_image(tmp_path / "a.png")
        cam = ImageFolderCamera(folder=tmp_path, camera_id="test")

        assert not cam.active  # not started yet
        cam.start()
        assert cam.active
        cam.read()  # consume the one image
        assert not cam.active

    def test_camera_id(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        cam = ImageFolderCamera(folder=tmp_path, camera_id="folder1")
        assert cam.camera_id == "folder1"

    def test_stop(self, tmp_path):
        from backend.ingestion.camera import ImageFolderCamera

        _create_image(tmp_path / "a.png")
        cam = ImageFolderCamera(folder=tmp_path, camera_id="test")
        cam.start()
        cam.stop()
        assert not cam.active


def _create_test_video(path, num_frames=10, fps=30.0):
    """Helper: write a small test video."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (100, 100))
    for i in range(num_frames):
        frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoCamera:
    def test_reads_all_frames(self, tmp_path):
        from backend.ingestion.camera import VideoCamera

        vid = tmp_path / "test.avi"
        _create_test_video(vid, num_frames=5, fps=10.0)

        cam = VideoCamera(path=vid, camera_id="vid1")
        cam.start()

        frames = []
        while True:
            result = cam.read()
            if result is None:
                break
            frames.append(result)

        assert len(frames) == 5
        # Timestamps based on frame index / fps
        assert frames[0][1] == pytest.approx(0.0)
        assert frames[1][1] == pytest.approx(0.1)

    def test_active_property(self, tmp_path):
        from backend.ingestion.camera import VideoCamera

        vid = tmp_path / "test.avi"
        _create_test_video(vid, num_frames=2)

        cam = VideoCamera(path=vid, camera_id="vid1")
        assert not cam.active
        cam.start()
        assert cam.active
        cam.read()
        cam.read()
        cam.read()  # returns None, exhausted
        assert not cam.active

    def test_stop_releases_capture(self, tmp_path):
        from backend.ingestion.camera import VideoCamera

        vid = tmp_path / "test.avi"
        _create_test_video(vid, num_frames=5)

        cam = VideoCamera(path=vid, camera_id="vid1")
        cam.start()
        cam.stop()
        assert not cam.active
        assert cam.read() is None

    def test_invalid_path_raises(self, tmp_path):
        from backend.ingestion.camera import VideoCamera

        cam = VideoCamera(path=tmp_path / "nonexistent.avi", camera_id="vid1")
        with pytest.raises(RuntimeError, match="Cannot open video"):
            cam.start()

    def test_camera_id(self, tmp_path):
        from backend.ingestion.camera import VideoCamera

        cam = VideoCamera(path=tmp_path / "x.avi", camera_id="myvid")
        assert cam.camera_id == "myvid"
