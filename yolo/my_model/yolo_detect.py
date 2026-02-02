import os
import sys
import argparse
import glob
import time
import csv
import datetime
import re
import warnings
import threading
import queue
import itertools
import math
import collections
from difflib import SequenceMatcher

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
''''
to run with multiple cameras use
python yolo_detect.py --model my_model.pt --source usb0,usb1 

'''
# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

SIMILARITY_THRESH = 0.60
OCR_THRESH = 0.80
LOG_COOLDOWN = 60.0
PLATE_REGEX = r'^[A-Z0-9]{5,8}$'
AUTHORIZED_FILE = 'authorized.txt'
STABILITY_VOTES_REQUIRED = 10

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Comma-separated sources: usb0,usb1 or video files', required=True)
parser.add_argument('--thresh', help='Confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH', default=None)
parser.add_argument('--record', help='Record video', action='store_true')
parser.add_argument('--grid-cols', help='Grid columns (auto if not set)', type=int, default=None)
parser.add_argument('--skip-frames', help='Run YOLO every N frames (default: 2)', type=int, default=2)
parser.add_argument('--ocr-workers', help='Number of OCR worker threads (default: 3)', type=int, default=3)
args = parser.parse_args()

model_path = args.model
min_thresh = args.thresh
user_res = args.resolution
record = args.record
grid_cols = args.grid_cols
skip_frames = max(1, args.skip_frames)
num_ocr_workers = max(1, args.ocr_workers)

multi_cam = len(img_source) > 1

# --- INITIALIZATION ---
if not os.path.exists(model_path):
    sys.exit('ERROR: Model path is invalid.')

model = YOLO(model_path, task='detect')
labels = model.names

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

log_filename = 'parking_log.csv'
authorized_plates = set()
last_auth_update = 0

# --- THREAD-SAFE SHARED STATE ---
class SharedState:
    """Thread-safe container for shared state across all workers"""

    def __init__(self):
        self._lock = threading.RLock()
        self._seen_plates = {}
        self._plate_votes = {}
        self._display_text = {}

    def is_similar_to_recent(self, new_text, current_time):
        """Check if plate was recently seen"""
        with self._lock:
            for past_text, last_time in self._seen_plates.items():
                if (current_time - last_time) < LOG_COOLDOWN:
                    ratio = SequenceMatcher(None, new_text, past_text).ratio()
                    if ratio >= SIMILARITY_THRESH:
                        return True, past_text
            return False, None

    def update_seen_plate(self, plate, timestamp):
        with self._lock:
            self._seen_plates[plate] = timestamp

    def remove_seen_plate(self, plate):
        with self._lock:
            if plate in self._seen_plates:
                del self._seen_plates[plate]

    def get_plate_votes(self, plate):
        with self._lock:
            return self._plate_votes.get(plate, {}).copy()

    def add_plate_vote(self, original, candidate):
        with self._lock:
            if original not in self._plate_votes:
                self._plate_votes[original] = {}
            self._plate_votes[original][candidate] = \
                self._plate_votes[original].get(candidate, 0) + 1
            return self._plate_votes[original][candidate]

    def clear_plate_votes(self, plate):
        with self._lock:
            if plate in self._plate_votes:
                self._plate_votes[plate].clear()

    def delete_plate_votes(self, plate):
        with self._lock:
            if plate in self._plate_votes:
                del self._plate_votes[plate]

    def set_display_text(self, camera_id, box_id, text, color):
        with self._lock:
            self._display_text[(camera_id, box_id)] = (text, color)

    def get_display_text(self, camera_id, box_id):
        with self._lock:
            return self._display_text.get((camera_id, box_id), (None, None))

    def clear_display_text_for_camera(self, camera_id):
        with self._lock:
            keys_to_remove = [k for k in self._display_text if k[0] == camera_id]
            for k in keys_to_remove:
                del self._display_text[k]

shared_state = SharedState()

# --- THREAD-SAFE CSV LOGGER ---
class CSVLogger:
    """Thread-safe CSV logging with camera ID support"""

    def __init__(self, filename):
        self.filename = filename
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Camera_ID', 'License_Plate', 'Status', 'OCR_Confidence'])

    def log(self, camera_id, plate, status, ocr_conf):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, camera_id, plate, status, f"{ocr_conf:.2f}"])

csv_logger = CSVLogger(log_filename)

# --- TEXT FILE LOADING ---
def update_authorized_list():
    global last_auth_update, authorized_plates
    if time.time() - last_auth_update > 5:
        if os.path.exists(AUTHORIZED_FILE):
            try:
                with open(AUTHORIZED_FILE, 'r') as f:
                    new_set = {line.strip().upper() for line in f if line.strip()}
                if new_set != authorized_plates:
                    print(f"Reloaded Authorized List: {len(new_set)} plates.")
                    authorized_plates = new_set
            except Exception as e:
                print(f"Error reading text file: {e}")
        last_auth_update = time.time()

def is_authorized(plate):
    return plate in authorized_plates

# --- SMART CORRECTION (B/8 and Q/O) ---
def smart_correction(text):
    if is_authorized(text):
        return text

    confusables = {
        'B': '8', '8': 'B',
        'Q': 'O', 'O': 'Q'
    }

    indices = [i for i, char in enumerate(text) if char in confusables]

    if not indices:
        return text

    chars = list(text)
    options = [[c, confusables[c]] for c in [text[i] for i in indices]]

    for combo in itertools.product(*options):
        for i, idx in enumerate(indices):
            chars[idx] = combo[i]
        candidate = "".join(chars)

        if is_authorized(candidate):
            return candidate

    return text

# --- HELPER FUNCTIONS ---
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    if width < 300:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = thresh.shape
    border = 30
    canvas = np.full((h + border*2, w + border*2), 255, dtype=np.uint8)
    canvas[border:h+border, border:w+border] = thresh
    return canvas

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# --- CAMERA CAPTURE CLASS ---
class CameraCapture:
    """Per-camera capture thread with frame buffer"""

    def __init__(self, camera_id, source_spec, resolution=None):
        self.camera_id = camera_id
        self.source_spec = source_spec
        self.resolution = resolution
        self.frame_buffer = collections.deque(maxlen=2)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.connected = False
        self.cap = None
        self.thread = None
        self.last_frame_time = 0
        self.fps = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        reconnect_attempts = 0
        max_reconnect = 10

        while self.running:
            if not self.connected:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect:
                    print(f"[{self.camera_id}] Max reconnect attempts, waiting...")
                    time.sleep(10.0)
                    reconnect_attempts = 0

                self._try_connect()
                if not self.connected:
                    time.sleep(1.0 * min(reconnect_attempts, 5))
                continue

            reconnect_attempts = 0
            ret, frame = self.cap.read()

            if not ret:
                self.connected = False
                print(f"[{self.camera_id}] Disconnected, attempting reconnect...")
                continue

            now = time.time()
            if self.last_frame_time > 0:
                self.fps = 1.0 / max(0.001, now - self.last_frame_time)
            self.last_frame_time = now

            if self.resolution:
                frame = cv2.resize(frame, self.resolution)

            with self.buffer_lock:
                self.frame_buffer.append((frame, now))

    def _try_connect(self):
        try:
            source_lower = self.source_spec.lower()
            is_rtsp = source_lower.startswith('rtsp://')

            if 'usb' in source_lower:
                idx = int(source_lower.replace('usb', ''))
                self.cap = cv2.VideoCapture(idx)
            elif self.source_spec.startswith('/dev/'):
                self.cap = cv2.VideoCapture(self.source_spec)
            elif self.source_spec.isdigit():
                self.cap = cv2.VideoCapture(int(self.source_spec))
            elif is_rtsp:
                # RTSP stream - use optimized settings
                # Try FFMPEG backend with TCP transport for reliability
                self.cap = cv2.VideoCapture(self.source_spec, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    # Minimize buffer to reduce latency (get latest frame, not queued)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Set a shorter timeout for faster failure detection
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            else:
                # Video file or other source
                self.cap = cv2.VideoCapture(self.source_spec)

            if self.cap.isOpened():
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.connected = True
                source_type = "RTSP stream" if is_rtsp else self.source_spec
                print(f"[{self.camera_id}] Connected to {source_type}")
            else:
                print(f"[{self.camera_id}] Failed to open {self.source_spec}")
        except Exception as e:
            print(f"[{self.camera_id}] Connection error: {e}")

    def get_latest_frame(self):
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
        return None, 0

# --- GRID DISPLAY CLASS ---
class GridDisplay:
    """Compose multiple camera frames into a single grid display"""

    def __init__(self, camera_ids, grid_cols=None, cell_size=(640, 480)):
        self.camera_ids = camera_ids
        self.num_cameras = len(camera_ids)
        self.cell_size = cell_size

        if grid_cols is None:
            self.cols = math.ceil(math.sqrt(self.num_cameras))
        else:
            self.cols = grid_cols
        self.rows = math.ceil(self.num_cameras / self.cols)

        self.canvas_width = self.cols * cell_size[0]
        self.canvas_height = self.rows * cell_size[1]

        self.positions = {}
        for i, cam_id in enumerate(camera_ids):
            row = i // self.cols
            col = i % self.cols
            x = col * cell_size[0]
            y = row * cell_size[1]
            self.positions[cam_id] = (x, y)

    def compose(self, frames, fps_data=None):
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        for cam_id in self.camera_ids:
            x, y = self.positions[cam_id]

            if cam_id in frames and frames[cam_id] is not None:
                frame = frames[cam_id]
                resized = cv2.resize(frame, self.cell_size)
                canvas[y:y+self.cell_size[1], x:x+self.cell_size[0]] = resized

                cv2.putText(canvas, cam_id, (x+10, y+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if fps_data and cam_id in fps_data:
                    fps_text = f"FPS: {fps_data[cam_id]:.1f}"
                    cv2.putText(canvas, fps_text, (x+10, y+60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(canvas, f"{cam_id}: DISCONNECTED",
                           (x+50, y+self.cell_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return canvas

# --- OCR QUEUE ---
sources = [s.strip() for s in args.source.split(',')]
num_cameras = len(sources)
ocr_queue = queue.Queue(maxsize=10 * num_cameras)

# --- OCR WORKER FUNCTION ---
def ocr_worker(worker_id):
    while True:
        try:
            update_authorized_list()

            data = ocr_queue.get(timeout=1)
            img_crop, conf_yolo, box_id, camera_id = data

            processed_plate = preprocess_for_ocr(img_crop)
            ocr_results = reader.readtext(processed_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            for _, text, ocr_conf in ocr_results:
                raw_plate = clean_text(text)

                if not re.match(PLATE_REGEX, raw_plate):
                    continue
                if ocr_conf < OCR_THRESH:
                    continue

                clean_plate = smart_correction(raw_plate)
                current_time = time.time()

                is_duplicate, match_name = shared_state.is_similar_to_recent(clean_plate, current_time)

                if is_authorized(clean_plate):
                    auth_status = "AUTHORIZED"
                    display_color = (0, 255, 0)
                else:
                    auth_status = "UNAUTHORIZED"
                    display_color = (0, 0, 255)

                display_text = f"{clean_plate} ({auth_status[:4]})"
                key = (cam_id, box_id)
                if is_duplicate:
                    if clean_plate == match_name:
                        shared_state.clear_plate_votes(match_name)
                    else:
                        vote_count = shared_state.add_plate_vote(match_name, clean_plate)

                        if vote_count >= STABILITY_VOTES_REQUIRED:
                            print(f"[{camera_id}] CORRECTION: '{match_name}' -> '{clean_plate}'")

                            shared_state.remove_seen_plate(match_name)
                            shared_state.update_seen_plate(clean_plate, current_time)
                            shared_state.delete_plate_votes(match_name)

                            match_name = clean_plate
                            if is_authorized(clean_plate):
                                auth_status = "AUTHORIZED"
                                display_color = (0, 255, 0)
                            else:
                                auth_status = "UNAUTHORIZED"
                                display_color = (0, 0, 255)

                            csv_logger.log(camera_id, clean_plate, f"{auth_status} (CORRECTED)", ocr_conf)
                            print(f"[{camera_id}] LOG UPDATED: {clean_plate}")

                            display_text = f"{clean_plate} ({auth_status[:4]})"
                            shared_state.set_display_text(camera_id, box_id, display_text, display_color)

                    shared_state.update_seen_plate(match_name, current_time)
                    shared_state.set_display_text(camera_id, box_id, f"{match_name} ({auth_status[:4]})", display_color)
                else:
                    csv_logger.log(camera_id, clean_plate, auth_status, ocr_conf)

                    log_icon = "OK" if auth_status == "AUTHORIZED" else "!!"
                    print(f"[{camera_id}] {log_icon} LOGGED: {clean_plate} [{auth_status}]")

                    shared_state.update_seen_plate(clean_plate, current_time)
                    shared_state.set_display_text(camera_id, box_id, display_text, display_color)

                break

            ocr_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[OCR Worker {worker_id}] Error: {e}")

# --- START OCR WORKERS ---
print(f"Starting {num_ocr_workers} OCR worker(s)...")
ocr_workers = []
for i in range(num_ocr_workers):
    t = threading.Thread(target=ocr_worker, args=(i,), daemon=True)
    t.start()
    ocr_workers.append(t)

# --- PARSE RESOLUTION ---
resolution = None
if user_res:
    resW, resH = map(int, user_res.split('x'))
    resolution = (resW, resH)
else:
    resolution = (640, 480)

# --- START CAMERA CAPTURE THREADS ---
camera_ids = [f"cam{i}" for i in range(num_cameras)]
cameras = {}

print(f"Starting {num_cameras} camera capture thread(s)...")
for cam_id, source in zip(camera_ids, sources):
    cam = CameraCapture(cam_id, source, resolution)
    cam.start()
    cameras[cam_id] = cam

# Wait for cameras to connect
time.sleep(2.0)

# --- INITIALIZE GRID DISPLAY ---
grid_display = GridDisplay(camera_ids, grid_cols, resolution)

# --- RECORDING SETUP ---
recorder = None
if record:
    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (grid_display.canvas_width, grid_display.canvas_height)
    )

# --- FRAME COUNTERS ---
frame_counters = {cam_id: 0 for cam_id in camera_ids}
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]

last_cleanup_time = time.time()
CLEANUP_INTERVAL = 3.0

print(f"\nParking Scanner Started")
print(f"  Cameras: {num_cameras}")
print(f"  Grid: {grid_display.rows}x{grid_display.cols}")
print(f"  OCR Workers: {num_ocr_workers}")
print(f"  Frame Skip: every {skip_frames} frame(s)")
print(f"  Controls: 's' = screenshot, 'q' = quit\n")

# --- MAIN LOOP ---
try:
    while True:
        frames = {}
        fps_data = {}

        for cam_id, cam in cameras.items():
            frame, timestamp = cam.get_latest_frame()

            if frame is None:
                frames[cam_id] = None
                continue

            frame = frame.copy()
            fps_data[cam_id] = cam.fps

            frame_counters[cam_id] += 1
            run_yolo = (frame_counters[cam_id] % skip_frames == 0)

            if run_yolo:
                results = model(frame, verbose=False)
                detections = results[0].boxes

                for i in range(len(detections)):
                    xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    conf = detections[i].conf.item()

                    if conf > min_thresh:
                        if 'plate' in classname.lower():
                            box_id = f"{xmin}_{ymin}"

                            text_label, color = shared_state.get_display_text(cam_id, box_id)

                            if text_label is None:
                                text_label = "Scanning..."
                                color = (255, 100, 0)

                                if not ocr_queue.full():
                                    h_img, w_img = frame.shape[:2]
                                    pad_x = int((xmax - xmin) * 0.08)
                                    pad_y = int((ymax - ymin) * 0.08)
                                    crop_xmin = max(0, xmin - pad_x)
                                    crop_ymin = max(0, ymin - pad_y)
                                    crop_xmax = min(w_img, xmax + pad_x)
                                    crop_ymax = min(h_img, ymax + pad_y)

                                    plate_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()

                                    if plate_crop.size > 0:
                                        ocr_queue.put((plate_crop, conf, box_id, cam_id))

                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                            cv2.putText(frame, text_label, (xmin, ymin-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            color = bbox_colors[classidx % 5]
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            frames[cam_id] = frame

        # Periodic cleanup of old display text
        if time.time() - last_cleanup_time > CLEANUP_INTERVAL:
            for cam_id in camera_ids:
                shared_state.clear_display_text_for_camera(cam_id)
            last_cleanup_time = time.time()

        # Compose and display grid
        grid_frame = grid_display.compose(frames, fps_data)
        cv2.imshow('Parking Scanner', grid_frame)

        if record and recorder:
            recorder.write(grid_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Screenshot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, grid_frame)
            print(f"Screenshot saved: {filename}")

finally:
    print("\nShutting down...")
    for cam in cameras.values():
        cam.stop()
    if record and recorder:
        recorder.release()
    cv2.destroyAllWindows()
    print("Done.")
