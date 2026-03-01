"""Tests for parking_lot.capture.grid.GridDisplay."""

import math

import numpy as np

from parking_lot.capture.grid import GridDisplay


class TestGridDimensions:
    def test_single_camera(self):
        grid = GridDisplay(["cam0"], cell_size=(320, 240))
        assert grid.cols == 1
        assert grid.rows == 1
        assert grid.canvas_width == 320
        assert grid.canvas_height == 240

    def test_two_cameras_auto_layout(self):
        grid = GridDisplay(["cam0", "cam1"], cell_size=(320, 240))
        assert grid.cols == 2  # ceil(sqrt(2)) = 2
        assert grid.rows == 1

    def test_four_cameras_auto_layout(self):
        grid = GridDisplay(["cam0", "cam1", "cam2", "cam3"], cell_size=(320, 240))
        assert grid.cols == 2
        assert grid.rows == 2
        assert grid.canvas_width == 640
        assert grid.canvas_height == 480

    def test_three_cameras_auto_layout(self):
        grid = GridDisplay(["cam0", "cam1", "cam2"], cell_size=(320, 240))
        assert grid.cols == 2  # ceil(sqrt(3)) = 2
        assert grid.rows == 2  # ceil(3/2) = 2

    def test_custom_grid_cols(self):
        grid = GridDisplay(["cam0", "cam1", "cam2", "cam3"], grid_cols=4, cell_size=(320, 240))
        assert grid.cols == 4
        assert grid.rows == 1


class TestCompose:
    def test_compose_returns_correct_shape(self):
        grid = GridDisplay(["cam0"], cell_size=(320, 240))
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        canvas = grid.compose({"cam0": frame}, {"cam0": 15.0})
        assert canvas.shape == (240, 320, 3)

    def test_compose_with_no_signal(self):
        grid = GridDisplay(["cam0", "cam1"], cell_size=(320, 240))
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        # cam1 has no frame
        canvas = grid.compose({"cam0": frame, "cam1": None}, {"cam0": 15.0})
        assert canvas.shape == (240, 640, 3)
        # cam1 area should not be all black (NO SIGNAL text drawn)
        cam1_region = canvas[:, 320:]
        # At least some pixels should be non-zero from the text
        assert cam1_region.any()

    def test_compose_multi_camera(self):
        ids = ["cam0", "cam1", "cam2", "cam3"]
        grid = GridDisplay(ids, cell_size=(160, 120))
        frames = {cid: np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for cid in ids}
        fps = {cid: 20.0 for cid in ids}
        canvas = grid.compose(frames, fps)
        assert canvas.shape == (240, 320, 3)  # 2x2 grid

    def test_compose_resizes_mismatched_frame(self):
        grid = GridDisplay(["cam0"], cell_size=(320, 240))
        # Feed a larger frame — compose should resize it
        big_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        canvas = grid.compose({"cam0": big_frame}, {"cam0": 30.0})
        assert canvas.shape == (240, 320, 3)

    def test_fps_text_below_threshold(self):
        grid = GridDisplay(["cam0"], cell_size=(320, 240))
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        # fps < 1.0 should show just cam_id, no FPS number
        canvas = grid.compose({"cam0": frame}, {"cam0": 0.5})
        # Just check it doesn't crash; text content verified visually
        assert canvas.shape == (240, 320, 3)
