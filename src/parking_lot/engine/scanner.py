"""ScannerEngine: central orchestrator for the parking lot scanner."""

import os
import queue
import threading
import time
import warnings

import cv2
import numpy as np

from parking_lot.config import ScannerConfig
from parking_lot.capture.camera import CameraCapture
from parking_lot.capture.grid import GridDisplay
from parking_lot.core.detector import YOLODetector
from parking_lot.core.ocr import OCRReader, stitch_results
from parking_lot.core.preprocessing import enhance_frame
from parking_lot.core.super_resolution import SuperResolution
from parking_lot.core.validation import TextValidator
from parking_lot.engine.logger import CSVLogger
from parking_lot.engine.state import SharedState

warnings.filterwarnings("ignore")

YOLO_PRINT_COOLDOWN = 5.0
OCR_ERROR_COOLDOWN = 10.0


class ScannerEngine:
    """Owns all threads, queues, cameras, and worker instances.

    Usage:
        engine = ScannerEngine(config)
        engine.start()          # non-blocking: spawns daemon threads
        engine.run_display_loop()  # blocking: CLI display (cv2.imshow)
        # OR: just query engine.state / engine.logger from FastAPI
    """

    def __init__(self, cfg: ScannerConfig):
        self.cfg = cfg
        self.start_time = time.time()

        # Core components
        self.detector = YOLODetector(cfg.model_path, cfg.detection, cfg.min_thresh)
        self.ocr_reader = OCRReader(cfg.ocr)
        self.sr = SuperResolution(cfg.sr)
        self.validator = TextValidator(cfg.validation)
        self.state = SharedState(cfg.detection)
        self.logger = CSVLogger()

        # Queues
        self.yolo_queue: queue.Queue = queue.Queue(maxsize=4)
        self.ocr_queue: queue.Queue = queue.Queue(maxsize=10 * max(1, len(cfg.sources)))

        # Cameras
        self.camera_ids: list[str] = []
        self.cameras: dict[str, CameraCapture] = {}
        self._setup_cameras()

        # Display
        self.grid: GridDisplay | None = None

        # Internal
        self._yolo_reported: dict = {}
        self._running = False

    def _setup_cameras(self):
        for i, src in enumerate(self.cfg.sources):
            if os.path.isfile(src):
                cid = os.path.basename(src)
            else:
                cid = f"cam{i}"
            self.camera_ids.append(cid)
            cam = CameraCapture(cid, src, self.cfg.camera.resolution, self.cfg.camera.target_fps)
            self.cameras[cid] = cam

        self.state.set_static_cameras(self.camera_ids, self.cfg.sources)

    def start(self):
        """Non-blocking start: spawns all worker threads and cameras."""
        self._running = True

        # Start cameras
        for cam in self.cameras.values():
            cam.start()

        # Start YOLO worker
        threading.Thread(target=self._yolo_worker, daemon=True).start()

        # Start OCR workers
        print(f"Starting {self.cfg.num_ocr_workers} OCR workers...")
        for i in range(self.cfg.num_ocr_workers):
            threading.Thread(target=self._ocr_worker, args=(i,), daemon=True).start()

        # Wait for cameras to connect
        time.sleep(2)

        # Setup grid display
        self.grid = GridDisplay(self.camera_ids, self.cfg.grid_cols, self.cfg.camera.resolution)
        print("\nSystem Ready. Press 'q' to quit.")

    def stop(self):
        """Stop all cameras and workers."""
        self._running = False
        for cam in self.cameras.values():
            cam.stop()

    def run_display_loop(self):
        """Blocking display loop for CLI mode (cv2.imshow)."""
        recorder = None
        if self.cfg.record and self.grid:
            recorder = cv2.VideoWriter(
                "output.avi",
                cv2.VideoWriter_fourcc(*"MJPG"),
                30,
                (self.grid.canvas_width, self.grid.canvas_height),
            )

        try:
            while self._running:
                frames = {}
                fps_map = {}

                for cid, cam in self.cameras.items():
                    f, _ = cam.get_latest_frame()
                    if f is None:
                        frames[cid] = None
                        continue

                    frame = f.copy()
                    if self.cfg.camera.feed_enhance:
                        frame = enhance_frame(frame)
                    fps_map[cid] = cam.fps

                    if not self.yolo_queue.full():
                        self.yolo_queue.put((frame, cid))

                    detections = self.state.get_detections(cid)
                    for det in detections:
                        x1, y1, x2, y2 = det["rect"]
                        bid = det["bid"]
                        txt, col = self.state.get_display_text(cid, bid)
                        if txt is None:
                            txt = "Scanning..."
                            col = (255, 165, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

                    frames[cid] = frame

                if self.grid:
                    final_img = self.grid.compose(frames, fps_map)
                    cv2.imshow("Truck Logistics Scanner (PaddleOCR)", final_img)
                    if recorder:
                        recorder.write(final_img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop()
            if recorder:
                recorder.release()
            cv2.destroyAllWindows()

    # --- Worker threads ---

    def _yolo_worker(self):
        print("YOLO Worker Started")
        while self._running:
            try:
                frame, cam_id = self.yolo_queue.get(timeout=1)
                now = time.time()

                # Cleanup old reported keys
                for key in list(self._yolo_reported.keys()):
                    if now - self._yolo_reported[key] > 60:
                        del self._yolo_reported[key]

                detections = self.detector.detect(frame)

                for det in detections:
                    report_key = (cam_id, det["bid"], det["label"])
                    if report_key not in self._yolo_reported or (now - self._yolo_reported[report_key]) >= YOLO_PRINT_COOLDOWN:
                        print(f"YOLO: {det['label']} {det['conf']:.2f} ({cam_id})")
                        self._yolo_reported[report_key] = now

                self.state.update_detections(cam_id, detections)

                # OCR handoff
                for det in detections:
                    txt, _ = self.state.get_display_text(cam_id, det["bid"])
                    if txt is None:
                        if not self.ocr_queue.full():
                            crop = self.detector.compute_crop(frame, det)
                            if crop is not None:
                                self.ocr_queue.put((crop, det["conf"], det["bid"], cam_id, det["label"]))

                self.yolo_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"YOLO Error: {e}")

    def _ocr_worker(self, wid: int):
        last_error = None
        last_error_time = 0

        while self._running:
            try:
                self.validator.refresh_authorized()
                data = self.ocr_queue.get(timeout=1)
                img_crop, conf_yolo, box_id, camera_id, class_type = data

                # Super-resolution + sharpen
                img_crop = self.sr.adaptive_upscale(img_crop)
                img_crop = SuperResolution.sharpen(img_crop)

                # Run 3-strategy OCR pipeline
                results = self.ocr_reader.run_pipeline(img_crop, class_type)

                # Stitch and match
                candidates = stitch_results(results)

                found_match = False
                final_text = None
                final_conf = 0

                for raw, cand_conf in candidates:
                    if not raw:
                        continue
                    corrected = self.validator.smart_correction(raw)
                    is_valid = self.validator.is_valid_format(corrected, class_type)

                    if is_valid and cand_conf >= self.cfg.ocr.thresh:
                        found_match = True
                        final_text = corrected
                        final_conf = cand_conf
                        break

                # Best effort
                if not found_match and candidates:
                    best_cand = max(candidates, key=lambda x: x[1])
                    if best_cand[1] >= self.cfg.ocr.best_effort_min_conf:
                        found_match = True
                        final_text = best_cand[0]
                        final_conf = best_cand[1]

                if found_match and final_text:
                    color = (0, 255, 0) if self.validator.is_authorized(final_text) else (255, 255, 255)
                    self.state.set_display_text(camera_id, box_id, final_text, color)
                    similar, _ = self.state.is_similar_to_recent(final_text, time.time())

                    if not similar:
                        self.logger.log(camera_id, final_text, class_type, final_conf)
                        self.state.update_seen_plate(final_text, time.time())
                        if self.cfg.ocr_debug:
                            print(f"LOGGED: {final_text} ({class_type})")

                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                now = time.time()
                if e != last_error or (now - last_error_time) >= OCR_ERROR_COOLDOWN:
                    print(f"OCR Error: {e}")
                    last_error, last_error_time = e, now

    # --- API helpers ---

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_camera_status(self) -> list[dict]:
        statuses = []
        for cid, cam in self.cameras.items():
            statuses.append({
                "id": cid,
                "source": cam.source_spec,
                "connected": cam.is_connected,
                "fps": round(cam.fps, 1),
            })
        return statuses

    def get_snapshot(self, camera_id: str) -> np.ndarray | None:
        cam = self.cameras.get(camera_id)
        if cam is None:
            return None
        frame, _ = cam.get_latest_frame()
        return frame

    def get_queue_sizes(self) -> dict:
        return {
            "yolo_queue": self.yolo_queue.qsize(),
            "ocr_queue": self.ocr_queue.qsize(),
        }
