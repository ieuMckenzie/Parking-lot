"""Grid display for multi-camera views."""

import math

import cv2
import numpy as np


class GridDisplay:
    """Composes multiple camera frames into a single grid canvas."""

    def __init__(self, camera_ids: list[str], grid_cols: int | None = None, cell_size: tuple = (640, 480)):
        self.camera_ids = camera_ids
        self.cell_size = cell_size
        self.cols = grid_cols if grid_cols else math.ceil(math.sqrt(len(camera_ids)))
        self.rows = math.ceil(len(camera_ids) / self.cols)
        self.canvas_width = self.cols * cell_size[0]
        self.canvas_height = self.rows * cell_size[1]

    def compose(self, frames: dict[str, np.ndarray | None], fps_data: dict[str, float]) -> np.ndarray:
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        for i, cam_id in enumerate(self.camera_ids):
            r, c = divmod(i, self.cols)
            x, y = c * self.cell_size[0], r * self.cell_size[1]
            if frames.get(cam_id) is not None:
                img = cv2.resize(frames[cam_id], self.cell_size)
                canvas[y: y + self.cell_size[1], x: x + self.cell_size[0]] = img

                fps = fps_data.get(cam_id, 0)
                fps_txt = f"{cam_id}" + (f" | FPS: {fps:.1f}" if fps > 1.0 else "")
                cv2.putText(canvas, fps_txt, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(canvas, "NO SIGNAL", (x + 50, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas
