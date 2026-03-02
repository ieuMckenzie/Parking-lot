# Truck logistics scanner using PaddleOCR (Optimized for Speed/Accuracy)
# Run: python paddle_detect.py --model my_model.pt --source s3.png
# Install: pip install paddlepaddle paddleocr ultralytics opencv-python
import os
import sys
import argparse
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
from paddleocr import PaddleOCR
from ultralytics import YOLO

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

SIMILARITY_THRESH = 0.60
OCR_THRESH = 0.40  # Slightly higher for Paddle v4
LOG_COOLDOWN = 60.0
DETECTION_HOLD_TIME = 0.5

# OPTIMIZED DEFAULTS
OCR_MIN_HEIGHT = 48         # Paddle expects ~32-48px height. 220 was too slow.
OCR_USE_BINARIZATION = False
OCR_MAG_RATIO = 1.5         # 1.5x is usually sufficient for 1080p
OCR_BEST_EFFORT_MIN_CONF = 0.30
OCR_UPSCALE = 2.0           # 2.0x is safer than 3.0x (artifacts)
OCR_SMALL_CROP_HEIGHT = 40
OCR_UPSCALE_SMALL = 3.0     # Only upscale tiny crops significantly
# INTER_LINEAR is much faster than LANCZOS4 and sufficient for OCR
OCR_INTERP = cv2.INTER_LINEAR 

FEED_ENHANCE = True         # Keep True for human visualization

PLATE_REGEX = r'^[A-Z0-9\-]{4,15}$'
CONTAINER_REGEX = r'^[A-Z]{4}[0-9]{6,7}$' # Standard ISO is 4 letters + 6 digits + 1 check
TRAILERNUM_REGEX = r'^[A-Z0-9\-]{1,10}$'

AUTHORIZED_FILE = 'authorized.txt'
STABILITY_VOTES_REQUIRED = 10

# Keys must be lowercase
CLASS_THRESHOLDS = {
    'usdot': 0.50,             
    'trailernum': 0.60,
    'containernum': 0.40,
    'containerplate': 0.70,
    'licenseplate': 0.70
}

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Comma-separated sources (0, rtsp://..., file.mp4)', required=True)
parser.add_argument('--thresh', help='Confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH', default=None)
parser.add_argument('--record', help='Record video', action='store_true')
parser.add_argument('--grid-cols', help='Grid columns', type=int, default=None)
parser.add_argument('--ocr-workers', help='Number of OCR threads', type=int, default=2)
parser.add_argument('--ocr-debug', help='Print OCR diagnostics', action='store_true')
parser.add_argument('--use-gpu', help='Enable GPU for PaddleOCR', action='store_true')
parser.add_argument('--no-enhance-feed', help='Disable frame enhancement', action='store_true')
args = parser.parse_args()

model_path = args.model
global_min_thresh = args.thresh
user_res = args.resolution
record = args.record
grid_cols = args.grid_cols
num_ocr_workers = max(1, args.ocr_workers)
ocr_debug = args.ocr_debug
use_gpu = args.use_gpu

if args.no_enhance_feed:
    FEED_ENHANCE = False
sources = [s.strip() for s in args.source.split(',') if s.strip()]
num_cameras = len(sources)

# --- INITIALIZATION ---
if not os.path.exists(model_path):
    sys.exit('ERROR: Model path is invalid.')

print("Initializing YOLO...")
model = YOLO(model_path, task='detect')
labels = model.names

print(f"Initializing PaddleOCR (GPU={use_gpu})...")
# PaddleOCR 3.x: only lang is supported; use_gpu/show_log/det_* are not in this API
reader = PaddleOCR(lang='en')

log_filename = 'truck_logistics.csv'
authorized_plates = set()
last_auth_update = 0

# --- SHARED STATE ---
class SharedState:
    def __init__(self):
        self._lock = threading.RLock()
        self._seen_plates = {}
        self._plate_votes = {}
        self._display_text = {}
        self.latest_detections = {}
        self._last_nonempty = {}
        self._last_nonempty_time = {}
        self._static_camera_ids = set()

    def set_static_cameras(self, camera_ids, sources):
        with self._lock:
            self._static_camera_ids = {cid for cid, src in zip(camera_ids, sources) if os.path.isfile(src)}

    def update_detections(self, cam_id, detections):
        with self._lock:
            if not detections and cam_id in self._static_camera_ids:
                return
            self.latest_detections[cam_id] = detections
            if detections:
                self._last_nonempty[cam_id] = detections
                self._last_nonempty_time[cam_id] = time.time()

    def get_detections(self, cam_id):
        with self._lock:
            det = self.latest_detections.get(cam_id, [])
            if det:
                return det
            last = self._last_nonempty.get(cam_id, [])
            if not last:
                return []
            t = self._last_nonempty_time.get(cam_id, 0)
            if (time.time() - t) <= DETECTION_HOLD_TIME:
                return last
            return []

    def is_similar_to_recent(self, new_text, current_time):
        with self._lock:
            for past_text, last_time in self._seen_plates.items():
                if (current_time - last_time) < LOG_COOLDOWN:
                    ratio = SequenceMatcher(None, new_text, past_text).ratio()
                    if ratio >= SIMILARITY_THRESH:
                        return True, past_text
            return False, None

    def update_seen_plate(self, plate, timestamp):
        with self._lock: self._seen_plates[plate] = timestamp

    def remove_seen_plate(self, plate):
        with self._lock:
            if plate in self._seen_plates: del self._seen_plates[plate]

    def add_plate_vote(self, original, candidate):
        with self._lock:
            if original not in self._plate_votes: self._plate_votes[original] = {}
            self._plate_votes[original][candidate] = self._plate_votes[original].get(candidate, 0) + 1
            return self._plate_votes[original][candidate]

    def set_display_text(self, camera_id, box_id, text, color):
        with self._lock: self._display_text[(camera_id, box_id)] = (text, color)

    def get_display_text(self, camera_id, box_id):
        with self._lock: return self._display_text.get((camera_id, box_id), (None, None))

shared_state = SharedState()

# --- CSV LOGGER ---
class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(['Timestamp', 'Camera_ID', 'Value', 'Data_Type', 'Confidence'])

    def log(self, camera_id, value, data_type, ocr_conf):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.filename, 'a', newline='') as f:
                csv.writer(f).writerow([timestamp, camera_id, value, data_type, f"{ocr_conf:.2f}"])

csv_logger = CSVLogger(log_filename)

def update_authorized_list():
    global last_auth_update, authorized_plates
    if time.time() - last_auth_update > 5:
        if os.path.exists(AUTHORIZED_FILE):
            try:
                with open(AUTHORIZED_FILE, 'r') as f:
                    authorized_plates = {line.strip().upper() for line in f if line.strip()}
            except: pass
        last_auth_update = time.time()

def is_authorized(plate): return plate in authorized_plates

def smart_correction(text):
    if is_authorized(text): return text
    
    confusables = {
        'B': '8', '8': 'B',
        'Q': '0', '0': 'Q', 'D': '0', 'O': '0',
        'S': '5', '5': 'S',
        'Z': '2', '2': 'Z',
        'G': '6', '6': 'G',
        'I': '1', '1': 'I'
    }
    
    indices = [i for i, char in enumerate(text) if char in confusables]
    if not indices: return text
    
    chars = list(text)
    options = [[c, confusables[c]] for c in [text[i] for i in indices]]
    
    # Limit combinations to prevent freeze on long strings
    if len(options) > 6: options = options[:6]

    for combo in itertools.product(*options):
        for i, idx in enumerate(indices): chars[idx] = combo[i]
        candidate = "".join(chars)
        
        if is_authorized(candidate): return candidate
        if re.match(CONTAINER_REGEX, candidate.replace('-', '')) or re.match(PLATE_REGEX, candidate):
            return candidate
            
    return text

# --- OPTIMIZED PREPROCESSING ---
def preprocess_for_ocr(img, use_binarization=False):
    """Heavy preprocessing for difficult images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height = gray.shape[0]
    if height < OCR_MIN_HEIGHT:
        scale = OCR_MIN_HEIGHT / height
        # INTER_LINEAR is faster
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=OCR_INTERP)
    
    # REMOVED fastNlMeansDenoising (Too slow for real-time)

    # Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    if use_binarization:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return gray

def preprocess_for_ocr_light(img):
    """Minimal preprocessing: grayscale + resize. Best for OCR v4."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    if height < OCR_MIN_HEIGHT:
        scale = OCR_MIN_HEIGHT / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=OCR_INTERP)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return gray

def _run_readtext(processed_img):
    if len(processed_img.shape) == 2:
        img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    else:
        img = processed_img
    result = reader.ocr(img)
    if not result or result[0] is None:
        return []
    r0 = result[0]

    # PaddleOCR 3.x: result[0] is dict with rec_texts, rec_scores, dt_polys/rec_polys
    def _get(key, default=None):
        if hasattr(r0, 'get') and callable(r0.get):
            return r0.get(key, default)
        return getattr(r0, key, default)
    texts = _get('rec_texts') or _get('rec_text')
    if texts is not None:
        if hasattr(texts, '__iter__') and not isinstance(texts, str):
            texts = list(texts)
        else:
            texts = [texts] if texts else []
        if texts:
            scores = _get('rec_scores') or _get('rec_score')
            if scores is None or (hasattr(scores, '__len__') and len(scores) != len(texts)):
                scores = [0.5] * len(texts)
            else:
                scores = list(scores)
            polys = _get('dt_polys') or _get('rec_polys') or _get('rec_boxes')
            if polys is not None and hasattr(polys, '__len__') and len(polys) == len(texts):
                polys = list(polys)
            else:
                polys = None
            out = []
            dummy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
            for i, text in enumerate(texts):
                text = str(text).strip() if text else ""
                if not text:
                    continue
                try:
                    conf = float(scores[i]) if i < len(scores) else 0.5
                except (ValueError, TypeError, IndexError):
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))
                if polys is not None and i < len(polys):
                    try:
                        box = np.array(polys[i], dtype=np.float64)
                        if box.ndim == 1 and box.size >= 4:
                            box = box.reshape(-1, 2)
                        if box.ndim < 2 or box.size < 4:
                            box = dummy.copy()
                    except (ValueError, TypeError):
                        box = dummy.copy()
                else:
                    box = dummy.copy()
                out.append((box, text, conf))
            return out

    # 2.x-style list of [box, (text, conf)]
    out = []
    for line in list(r0):
        if not line or len(line) < 2:
            continue
        try:
            box = np.array(line[0], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if box.ndim == 1 and box.size >= 4:
            box = box.reshape(-1, 2)
        if box.ndim < 2 or box.size < 4:
            continue
        second = line[1]
        if isinstance(second, (list, tuple)) and len(second) >= 2:
            text = str(second[0]).strip()
            try:
                conf = float(second[1])
            except (ValueError, TypeError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
        else:
            text, conf = str(second).strip(), 0.5
        out.append((box, text, conf))
    return out

def _bbox_y0(item):
    b = item[0]
    try:
        return float(b[0][1]) if getattr(b, 'ndim', 2) >= 2 and len(b) > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0
def _bbox_x0(item):
    b = item[0]
    try:
        return float(b[0][0]) if getattr(b, 'ndim', 2) >= 2 and len(b) > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0

def clean_text(text): return re.sub(r'[^A-Z0-9\-]', '', text.upper())

def enhance_frame(frame):
    """Visualization only. Does not affect OCR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)

# --- CAMERA CAPTURE CLASS ---
class CameraCapture:
    def __init__(self, camera_id, source_spec, resolution=None, target_fps=20):
        self.camera_id = camera_id
        self.source_spec = source_spec
        self.resolution = resolution
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.frame_buffer = collections.deque(maxlen=1)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.cap = None
        self.fps = 0.0
        self.last_frame_time = 0
        self.is_connected = False

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

    def _capture_loop(self):
        last_saved_time = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.is_connected = False
                self._connect()
                if not self.is_connected:
                    time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01) # Avoid busy loop on static files
                if isinstance(self.source_spec, str) and os.path.isfile(self.source_spec):
                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop file
                continue
            
            now = time.time()
            if (now - last_saved_time) >= self.frame_interval:
                if self.resolution:
                    frame = cv2.resize(frame, self.resolution)

                if self.last_frame_time > 0:
                    self.fps = 1.0 / max(0.001, now - self.last_frame_time)
                self.last_frame_time = now

                with self.buffer_lock:
                    self.frame_buffer.append((frame, now))
                
                last_saved_time = now

    def _connect(self):
        try:
            src = self.source_spec
            if 'usb' in src.lower(): self.cap = cv2.VideoCapture(int(src.replace('usb', '')))
            elif src.isdigit(): self.cap = cv2.VideoCapture(int(src))
            else: self.cap = cv2.VideoCapture(src)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps) 
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
                if not self.is_connected:
                    print(f"System: Loaded {self.camera_id}")
                    self.is_connected = True
        except: pass           
    
    def get_latest_frame(self):
        with self.buffer_lock:
            if self.frame_buffer: return self.frame_buffer[-1]
        return None, 0

# --- YOLO WORKER ---
yolo_input_queue = queue.Queue(maxsize=4)
_yolo_reported = {}
YOLO_PRINT_COOLDOWN = 5.0

def yolo_worker():
    global _yolo_reported
    print("YOLO Worker Started")
    while True:
        try:
            frame, cam_id = yolo_input_queue.get(timeout=1)
            now = time.time()
            for key in list(_yolo_reported.keys()):
                if now - _yolo_reported[key] > 60: del _yolo_reported[key]
            
            results = model(frame, verbose=False, iou=0.45)[0] # Added iou param for better NMS
            detections = []
            
            def _normalize_label(rl): return rl.lower().replace(' ', '').replace('_', '')
            targets = ['usdot', 'trailernum', 'containerplate', 'licenseplate', 'containernum']
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls_id = int(box.cls)
                raw_label = labels[cls_id] if cls_id < len(labels) else "unknown"
                label_key = _normalize_label(raw_label)
                req_conf = CLASS_THRESHOLDS.get(label_key, global_min_thresh)
                
                if conf > req_conf:
                    if any(t in label_key for t in targets) or label_key in targets:
                        bid = f"{x1}_{y1}"
                        detections.append({
                            'rect': (x1, y1, x2, y2),
                            'bid': bid,
                            'label': label_key,
                            'conf': conf
                        })
                        report_key = (cam_id, bid, label_key)
                        if report_key not in _yolo_reported or (now - _yolo_reported[report_key]) >= YOLO_PRINT_COOLDOWN:
                            print(f"YOLO: {label_key} {conf:.2f} ({cam_id})")
                            _yolo_reported[report_key] = now
            
            shared_state.update_detections(cam_id, detections)
            
            # OCR Handoff
            for det in detections:
                bid = det['bid']
                txt, _ = shared_state.get_display_text(cam_id, bid)
                if txt is None:
                    if not ocr_queue.full():
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = det['rect']
                        
                        # Dynamic padding - scale inversely with crop size
                        crop_h = y2 - y1
                        if crop_h < 30:
                            pad_x_pct, pad_y_pct = 0.40, 0.35
                        elif crop_h < 60:
                            pad_x_pct, pad_y_pct = 0.30, 0.25
                        else:
                            pad_x_pct, pad_y_pct = 0.20, 0.15
                        if det['label'] in ('licenseplate', 'containerplate'):
                            pad_x_pct += 0.08
                            pad_y_pct += 0.07
                        
                        px, py = int((x2-x1) * pad_x_pct), int((y2-y1) * pad_y_pct)
                        crop = frame[max(0, y1-py):min(h, y2+py), max(0, x1-px):min(w, x2+px)]
                        
                        if crop.size > 0:
                            ocr_queue.put((crop, det['conf'], bid, cam_id, det['label']))

            yolo_input_queue.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"YOLO Error: {e}")

# --- GRID DISPLAY ---
class GridDisplay:
    def __init__(self, camera_ids, grid_cols=None, cell_size=(640, 480)):
        self.camera_ids = camera_ids
        self.cell_size = cell_size
        self.cols = grid_cols if grid_cols else math.ceil(math.sqrt(len(camera_ids)))
        self.rows = math.ceil(len(camera_ids) / self.cols)
        self.canvas_width = self.cols * cell_size[0]
        self.canvas_height = self.rows * cell_size[1]

    def compose(self, frames, fps_data):
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        for i, cam_id in enumerate(self.camera_ids):
            r, c = divmod(i, self.cols)
            x, y = c * self.cell_size[0], r * self.cell_size[1]
            if frames.get(cam_id) is not None:
                img = cv2.resize(frames[cam_id], self.cell_size)
                canvas[y:y+self.cell_size[1], x:x+self.cell_size[0]] = img
                
                fps = fps_data.get(cam_id, 0)
                fps_txt = f"{cam_id}" + (f" | FPS: {fps:.1f}" if fps > 1.0 else "")
                cv2.putText(canvas, fps_txt, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(canvas, "NO SIGNAL", (x+50, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas

# --- OCR WORKER (OPTIMIZED FLOW) ---
ocr_queue = queue.Queue(maxsize=10 * num_cameras)
_last_ocr_error = None
_last_ocr_error_time = 0
OCR_ERROR_COOLDOWN = 10.0

def ocr_worker(wid):
    global _last_ocr_error, _last_ocr_error_time
    while True:
        try:
            update_authorized_list()
            data = ocr_queue.get(timeout=1)
            img_crop, conf_yolo, box_id, camera_id, class_type = data

            # Adaptive upscale
            h_crop = img_crop.shape[0] if hasattr(img_crop, 'shape') else 0
            upscale = OCR_UPSCALE_SMALL if (h_crop > 0 and h_crop < OCR_SMALL_CROP_HEIGHT) else OCR_UPSCALE
            if upscale != 1.0:
                img_crop = cv2.resize(img_crop, None, fx=upscale, fy=upscale, interpolation=OCR_INTERP)

            def get_avg_conf(res):
                if not res: return 0.0
                return sum(c for _, _, c in res) / len(res)

            # --- STRATEGY 1: LIGHT (Fastest) ---
            processed_light = preprocess_for_ocr_light(img_crop)
            results = _run_readtext(processed_light)
            avg_conf = get_avg_conf(results)

            # --- STRATEGY 2: HEAVY (Only if needed) ---
            if not results or avg_conf < 0.85:
                processed = preprocess_for_ocr(img_crop, use_binarization=False)
                results_heavy = _run_readtext(processed)
                if get_avg_conf(results_heavy) > avg_conf:
                    results = results_heavy

            # --- STRATEGY 3: BINARY (Last resort for plates) ---
            is_plate = class_type in ('licenseplate', 'containerplate')
            if is_plate and (not results or get_avg_conf(results) < 0.60):
                processed_bin = preprocess_for_ocr(img_crop, use_binarization=True)
                results_bin = _run_readtext(processed_bin)
                if get_avg_conf(results_bin) > get_avg_conf(results):
                    results = results_bin

            # --- STITCHING & MATCHING ---
            # Sort vertically then horizontally
            by_y = sorted(results, key=_bbox_y0)
            stitched = ""
            for _, text, _ in by_y:
                stitched += re.sub(r'[^A-Z0-9\-]', '', text.upper())

            candidates = [(stitched, get_avg_conf(results))]
            for _, text, conf in results:
                candidates.append((re.sub(r'[^A-Z0-9\-]', '', text.upper()), conf))

            found_match = False
            final_text = None
            final_conf = 0
            
            for raw, cand_conf in candidates:
                if not raw: continue
                corrected = smart_correction(raw)
                
                is_valid = False
                if 'containernum' in class_type or ('container' in class_type and 'plate' not in class_type):
                    is_valid = bool(re.match(CONTAINER_REGEX, corrected.replace('-', '')))
                elif 'trailernum' in class_type:
                    is_valid = bool(re.match(TRAILERNUM_REGEX, corrected))
                else:
                    is_valid = bool(re.match(PLATE_REGEX, corrected))
                
                if is_valid and cand_conf >= OCR_THRESH:
                    found_match = True
                    final_text = corrected
                    final_conf = cand_conf
                    break

            # Best effort logic
            if not found_match and candidates:
                # Filter candidates that at least loosely match regex
                # (Simple heuristic check)
                best_cand = max(candidates, key=lambda x: x[1])
                if best_cand[1] >= OCR_BEST_EFFORT_MIN_CONF:
                    found_match = True
                    final_text = best_cand[0]
                    final_conf = best_cand[1]

            if found_match and final_text:
                color = (0, 255, 0) if is_authorized(final_text) else (255, 255, 255)
                shared_state.set_display_text(camera_id, box_id, final_text, color)
                similar, _ = shared_state.is_similar_to_recent(final_text, time.time())
                
                if not similar:
                    csv_logger.log(camera_id, final_text, class_type, final_conf)
                    shared_state.update_seen_plate(final_text, time.time())
                    if ocr_debug: print(f"LOGGED: {final_text} ({class_type})")

            ocr_queue.task_done()
        except queue.Empty: continue
        except Exception as e:
            now = time.time()
            if e != _last_ocr_error or (now - _last_ocr_error_time) >= OCR_ERROR_COOLDOWN:
                print(f"OCR Error: {e}")
                _last_ocr_error, _last_ocr_error_time = e, now

# --- STARTUP ---
print(f"Starting {num_ocr_workers} OCR workers...")
for i in range(num_ocr_workers): threading.Thread(target=ocr_worker, args=(i,), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()

if user_res:
    resW, resH = map(int, user_res.split('x'))
    resolution = (resW, resH)
else: resolution = (1920, 1080)

camera_ids = []
for i, src in enumerate(sources):
    if os.path.isfile(src): camera_ids.append(os.path.basename(src))
    else: camera_ids.append(f"cam{i}")

shared_state.set_static_cameras(camera_ids, sources)

cameras = {}
for cid, src in zip(camera_ids, sources):
    cam = CameraCapture(cid, src, resolution, target_fps=20)
    cam.start()
    cameras[cid] = cam

time.sleep(2)
grid = GridDisplay(camera_ids, grid_cols, resolution)
if record: recorder = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (grid.canvas_width, grid.canvas_height))

print("\nSystem Ready (Async Mode). Press 'q' to quit.")

# --- MAIN LOOP ---
try:
    while True:
        frames = {}
        fps_map = {}
        
        for cid, cam in cameras.items():
            f, _ = cam.get_latest_frame()
            if f is None: 
                frames[cid] = None
                continue
            
            frame = f.copy()
            if FEED_ENHANCE: frame = enhance_frame(frame)
            fps_map[cid] = cam.fps
            
            if not yolo_input_queue.full():
                yolo_input_queue.put((frame, cid))

            detections = shared_state.get_detections(cid)

            for det in detections:
                x1, y1, x2, y2 = det['rect']
                bid = det['bid']
                txt, col = shared_state.get_display_text(cid, bid)
                if txt is None:
                    txt = "Scanning..."
                    col = (255, 165, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            frames[cid] = frame

        final_img = grid.compose(frames, fps_map)
        cv2.imshow('Truck Logistics Scanner (PaddleOCR)', final_img)
        if record and recorder: recorder.write(final_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    for c in cameras.values(): c.stop()
    if record and recorder: recorder.release()
    cv2.destroyAllWindows()