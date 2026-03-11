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

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

SIMILARITY_THRESH = 0.60
OCR_THRESH = 0.35
LOG_COOLDOWN = 60.0
DETECTION_HOLD_TIME = 0.5
# Low-res defaults: all feeds/images assumed low-res for better OCR
OCR_MIN_HEIGHT = 220
OCR_USE_BINARIZATION = False
OCR_MAG_RATIO = 3.0
OCR_BEST_EFFORT_MIN_CONF = 0.25
# Upscale the crop before OCR (3.0 = triple size for low-res)
OCR_UPSCALE = 3.0
# Sharper upscaling for low-res (LANCZOS4); use INTER_CUBIC if LANCZOS4 is slow
OCR_INTERP = cv2.INTER_LANCZOS4 if hasattr(cv2, 'INTER_LANCZOS4') else cv2.INTER_CUBIC
# Enhance full frame (contrast + light sharpen) for clearer YOLO/OCR input
FEED_ENHANCE = True

PLATE_REGEX = r'^[A-Z0-9\-]{4,15}$'
CONTAINER_REGEX = r'^[A-Z]{4}[0-9]{5,8}$'
# FMCSA/49 CFR: trailer number alphanumeric, max 10 chars per trailer
TRAILERNUM_REGEX = r'^[A-Z0-9\-]{1,10}$'

AUTHORIZED_FILE = 'authorized.txt'
STABILITY_VOTES_REQUIRED = 10

# FIX: Keys must be lowercase to match the logic below
CLASS_THRESHOLDS = {
    'usdot': 0.50,             
    'trailernum': 0.60,
    'containernum': 0.40,
    'containerplate': 0.75,
    'licenseplate': 0.75
}

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Comma-separated sources', required=True)
parser.add_argument('--thresh', help='Confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH', default=None)
parser.add_argument('--record', help='Record video', action='store_true')
parser.add_argument('--grid-cols', help='Grid columns', type=int, default=None)
parser.add_argument('--ocr-workers', help='Number of OCR threads', type=int, default=3)
parser.add_argument('--ocr-debug', help='Print OCR diagnostics (raw results, candidates, why no match)', action='store_true')
parser.add_argument('--ocr-upscale', help='Upscale crop by this factor before OCR (e.g. 2.0 = double size)', type=float, default=None)
parser.add_argument('--no-enhance-feed', help='Disable frame enhancement (contrast + sharpen) for a raw feed', action='store_true')
parser.add_argument('--draw-all', help='Draw all YOLO classes above threshold, even if not in target class list', action='store_true')
args = parser.parse_args()

model_path = args.model
global_min_thresh = args.thresh
user_res = args.resolution
record = args.record
grid_cols = args.grid_cols
num_ocr_workers = max(1, args.ocr_workers)
ocr_debug = args.ocr_debug
draw_all = args.draw_all
if args.ocr_upscale is not None:
    OCR_UPSCALE = max(0.25, min(4.0, args.ocr_upscale))
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

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)

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

    def clear_plate_votes(self, plate):
        with self._lock:
            if plate in self._plate_votes: self._plate_votes[plate].clear()

    def delete_plate_votes(self, plate):
        with self._lock:
            if plate in self._plate_votes: del self._plate_votes[plate]

    def set_display_text(self, camera_id, box_id, text, color):
        with self._lock: self._display_text[(camera_id, box_id)] = (text, color)

    def get_display_text(self, camera_id, box_id):
        with self._lock: return self._display_text.get((camera_id, box_id), (None, None))

    def clear_display_text_for_camera(self, camera_id):
        with self._lock:
            keys_to_remove = [k for k in self._display_text if k[0] == camera_id]
            for k in keys_to_remove: del self._display_text[k]

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

# FIX: Expanded Smart Correction
def smart_correction(text):
    if is_authorized(text): return text
    
    # Expanded confusion map
    confusables = {
        'B': '8', '8': 'B',
        'Q': 'O', 'O': 'Q', 'D': 'O', '0': 'O',
        'S': '5', '5': 'S',
        'Z': '2', '2': 'Z',
        'G': '6', '6': 'G',
        'I': '1', '1': 'I'
    }
    
    indices = [i for i, char in enumerate(text) if char in confusables]
    if not indices: return text
    
    chars = list(text)
    options = [[c, confusables[c]] for c in [text[i] for i in indices]]
    
    for combo in itertools.product(*options):
        for i, idx in enumerate(indices): chars[idx] = combo[i]
        candidate = "".join(chars)
        
        # Priority 1: Check Authorized List
        if is_authorized(candidate): return candidate
        
        # Priority 2: Check Regex (container: strip hyphens for match)
        if re.match(CONTAINER_REGEX, candidate.replace('-', '')) or re.match(PLATE_REGEX, candidate):
            return candidate
            
    return text

# FIX: Better Preprocessing
def preprocess_for_ocr(img, use_binarization=None):
    if use_binarization is None:
        use_binarization = OCR_USE_BINARIZATION
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize: Upscale if too small so EasyOCR can detect each character
    height = gray.shape[0]
    if height < OCR_MIN_HEIGHT:
        scale = OCR_MIN_HEIGHT / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=OCR_INTERP)
    
    # 3. Denoise (slightly reduced strength when using binarization)
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    
    # 4. Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 5. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # 6. Optional binarization for plate/container text
    if use_binarization:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 7. Padding
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return gray

def preprocess_for_ocr_light(img):
    """Minimal preprocessing: grayscale, upscale, padding. No denoise/CLAHE/sharpen so EasyOCR sees raw contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    if height < OCR_MIN_HEIGHT:
        scale = OCR_MIN_HEIGHT / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=OCR_INTERP)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return gray

def _run_readtext(processed_img):
    return reader.readtext(
        processed_img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        decoder='greedy',
        paragraph=False,
        mag_ratio=OCR_MAG_RATIO,
        adjust_contrast=0.5,
        contrast_ths=0.1,
        width_ths=0.6,
        rotation_info=[90, 180, 270],
        text_threshold=0.5,
        low_text=0.3
    )

def clean_text(text): return re.sub(r'[^A-Z0-9\-]', '', text.upper())

def enhance_frame(frame):
    """Improve clarity: contrast (CLAHE on luminance) + light sharpen. Use on full frame before YOLO/display."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    out = cv2.filter2D(out, -1, kernel)
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
        self.is_connected = False # FIX: Prevents spamming
        self.static_image = None

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()

    def _capture_loop(self):
        last_saved_time = 0
        while self.running:
            if self.static_image is not None:
                now = time.time()
                if (now - last_saved_time) >= self.frame_interval:
                    frame = self.static_image.copy()
                    if self.resolution:
                        frame = cv2.resize(frame, self.resolution)

                    if self.last_frame_time > 0:
                        self.fps = 1.0 / max(0.001, now - self.last_frame_time)
                    self.last_frame_time = now

                    with self.buffer_lock:
                        self.frame_buffer.append((frame, now))

                    last_saved_time = now
                time.sleep(0.005)
                continue

            if not self.cap or not self.cap.isOpened():
                self.is_connected = False
                self._connect()
                if not self.is_connected:
                    time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # FIX: Static files just sleep instead of release
                time.sleep(0.01)
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

            if os.path.isfile(src):
                ext = os.path.splitext(src)[1].lower()
                if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}:
                    img = cv2.imread(src)
                    if img is not None:
                        self.static_image = img
                        if not self.is_connected:
                            print(f"System: Loaded {self.camera_id}")
                            self.is_connected = True
                    return

            if 'usb' in src.lower(): self.cap = cv2.VideoCapture(int(src.replace('usb', '')))
            elif src.isdigit(): self.cap = cv2.VideoCapture(int(src))
            else: self.cap = cv2.VideoCapture(src)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps) 
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
                # FIX: Only print once
                if not self.is_connected:
                    print(f"System: Loaded {self.camera_id}")
                    self.is_connected = True
        except: 
            pass           
    
    def get_latest_frame(self):
        with self.buffer_lock:
            if self.frame_buffer: return self.frame_buffer[-1]
        return None, 0

# --- YOLO WORKER ---
yolo_input_queue = queue.Queue(maxsize=4)
_yolo_reported = {}
YOLO_PRINT_COOLDOWN = 5.0
TARGET_LABEL_HINTS = ['usdot', 'trailernum', 'containernum', 'containerplate', 'licenseplate']

def yolo_worker():
    global _yolo_reported
    print("YOLO Worker Started")
    while True:
        try:
            frame, cam_id = yolo_input_queue.get(timeout=1)
            now = time.time()
            for key in list(_yolo_reported.keys()):
                if now - _yolo_reported[key] > 60:
                    del _yolo_reported[key]
            
            results = model(frame, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls_id = int(box.cls)
                
                raw_label = labels[cls_id] if cls_id < len(labels) else "unknown"
                label_key = raw_label.lower().replace(' ', '_').replace('-', '_')
                label_norm = re.sub(r'[^a-z0-9]', '', raw_label.lower())
                req_conf = CLASS_THRESHOLDS.get(label_norm, CLASS_THRESHOLDS.get(label_key, global_min_thresh))
                is_target = any(t in label_norm for t in TARGET_LABEL_HINTS)
                
                if conf > req_conf:
                    if is_target or draw_all:
                        bid = f"{x1}_{y1}"
                        detections.append({
                            'rect': (x1, y1, x2, y2),
                            'bid': bid,
                            'label': label_key,
                            'raw_label': raw_label,
                            'is_target': is_target,
                            'conf': conf
                        })
                        report_key = (cam_id, bid, label_key)
                        if report_key not in _yolo_reported or (now - _yolo_reported[report_key]) >= YOLO_PRINT_COOLDOWN:
                            print(f"YOLO: {label_key} {conf:.2f} ({cam_id})")
                            _yolo_reported[report_key] = now
            
            shared_state.update_detections(cam_id, detections)
            
            # OCR Handoff
            for det in detections:
                if not det.get('is_target', True):
                    continue
                bid = det['bid']
                txt, _ = shared_state.get_display_text(cam_id, bid)
                if txt is None:
                    if not ocr_queue.full():
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = det['rect']
                        px = int((x2-x1) * (0.20 if 'number' in det['label'] else 0.18))
                        py = int((y2-y1) * 0.15)
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
                
                # Show FPS only if > 0 (avoids clutter on static images)
                fps = fps_data.get(cam_id, 0)
                fps_txt = f"{cam_id}" + (f" | FPS: {fps:.1f}" if fps > 1.0 else "")
                
                cv2.putText(canvas, fps_txt, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(canvas, "NO SIGNAL", (x+50, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas

# --- OCR WORKER ---
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

            if OCR_UPSCALE != 1.0 and OCR_UPSCALE > 0:
                img_crop = cv2.resize(img_crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=OCR_INTERP)

            processed = preprocess_for_ocr(img_crop)
            processed_light = preprocess_for_ocr_light(img_crop)
            results = _run_readtext(processed)
            results_light = _run_readtext(processed_light)
            def stitch_len(r):
                s = ""
                for _, text, _ in r:
                    s += re.sub(r'[^A-Z0-9\-]', '', text.upper())
                return len(s), sum(c for _, _, c in r) / len(r) if r else 0
            len_full, conf_full = stitch_len(results)
            len_light, conf_light = stitch_len(results_light)
            if len_light > len_full or (len_light == len_full and conf_light > conf_full):
                results = results_light
            if not results and OCR_USE_BINARIZATION:
                processed_no_bin = preprocess_for_ocr(img_crop, use_binarization=False)
                results = _run_readtext(processed_no_bin)

            # --- VERTICAL STITCH: sort by Y (top to bottom) ---
            by_y = sorted(results, key=lambda x: x[0][0][1])
            stitched_vertical = ""
            conf_sum = 0
            conf_count = 0
            for _, text, conf in by_y:
                part = re.sub(r'[^A-Z0-9\-]', '', text.upper())
                if part:
                    stitched_vertical += part
                    conf_sum += conf
                    conf_count += 1
            avg_conf = (conf_sum / conf_count) if conf_count > 0 else 0

            # --- HORIZONTAL STITCH: sort by X (left to right) ---
            by_x = sorted(results, key=lambda x: x[0][0][0])
            stitched_horizontal = ""
            for _, text, _ in by_x:
                part = re.sub(r'[^A-Z0-9\-]', '', text.upper())
                if part:
                    stitched_horizontal += part

            # Build candidate list: stitched (vertical, horizontal) + individual lines; dedupe
            candidates_with_conf = [(stitched_vertical, avg_conf), (stitched_horizontal, avg_conf)]
            for _, text, conf in results:
                raw_part = re.sub(r'[^A-Z0-9\-]', '', text.upper())
                if raw_part:
                    candidates_with_conf.append((raw_part, conf))
            seen = set()
            unique_candidates = []
            for cand, c in candidates_with_conf:
                if cand not in seen and len(cand) >= 1:
                    seen.add(cand)
                    unique_candidates.append((cand, c))

            if ocr_debug:
                print(f"[OCR debug] {camera_id} {box_id} {class_type}: results_empty={len(results)==0}, raw_count={len(results)}")
                for i, (bbox, text, conf) in enumerate(results):
                    print(f"  raw[{i}] text={repr(text)} conf={conf:.2f}")
                print(f"  stitched_vertical={repr(stitched_vertical)} stitched_horizontal={repr(stitched_horizontal)} avg_conf={avg_conf:.2f}")
                print(f"  unique_candidates={len(unique_candidates)}")

            found_match = False
            final_text = None
            final_conf = 0
            regex_valid_candidates = []
            for raw, cand_conf in unique_candidates:
                cleaned = clean_text(raw)
                if not cleaned:
                    if ocr_debug:
                        print(f"  candidate {repr(raw)} -> cleaned empty, skip")
                    continue
                corrected = smart_correction(cleaned)
                is_valid = False
                if 'containernum' in class_type or ('container' in class_type and 'plate' not in class_type):
                    norm = corrected.replace('-', '')
                    is_valid = bool(re.match(CONTAINER_REGEX, norm))
                elif 'trailernum' in class_type:
                    is_valid = bool(re.match(TRAILERNUM_REGEX, corrected))
                else:
                    is_valid = bool(re.match(PLATE_REGEX, corrected))
                if is_valid:
                    regex_valid_candidates.append((corrected, cand_conf))
                if is_valid and cand_conf >= OCR_THRESH:
                    found_match = True
                    final_text = corrected
                    final_conf = cand_conf
                    if ocr_debug:
                        print(f"  ACCEPTED: corrected={corrected} conf={cand_conf:.2f} (>= OCR_THRESH)")
                    break
                if ocr_debug:
                    print(f"  candidate raw={repr(raw)} cleaned={cleaned} corrected={corrected} is_valid={is_valid} conf={cand_conf:.2f}")

            if not found_match and regex_valid_candidates:
                best = max(regex_valid_candidates, key=lambda x: x[1])
                if best[1] >= OCR_BEST_EFFORT_MIN_CONF:
                    found_match = True
                    final_text = best[0]
                    final_conf = best[1]
                    if ocr_debug:
                        print(f"  BEST_EFFORT: corrected={final_text} conf={final_conf:.2f}")

            if not found_match and ocr_debug:
                if not results:
                    print(f"  NO_MATCH reason: EasyOCR returned empty")
                elif not regex_valid_candidates:
                    print(f"  NO_MATCH reason: no candidate matched regex")
                else:
                    print(f"  NO_MATCH reason: best regex-valid conf {max(c[1] for c in regex_valid_candidates):.2f} < {OCR_BEST_EFFORT_MIN_CONF}")

            if found_match and final_text:
                color = (0, 255, 0) if is_authorized(final_text) else (255, 255, 255)
                shared_state.set_display_text(camera_id, box_id, final_text, color)
                shared_state.add_plate_vote(box_id, final_text)
                similar, _ = shared_state.is_similar_to_recent(final_text, time.time())
                if not similar:
                    if 'containernum' in class_type:
                        data_type = 'containernum'
                    elif 'containerplate' in class_type:
                        data_type = 'containerplate'
                    elif 'licenseplate' in class_type:
                        data_type = 'licenseplate'
                    elif 'usdot' in class_type:
                        data_type = 'usdot'
                    elif 'trailernum' in class_type:
                        data_type = 'trailernum'
                    else:
                        data_type = 'container' if 'container' in class_type else 'plate'
                    csv_logger.log(camera_id, final_text, data_type, final_conf)
                    shared_state.update_seen_plate(final_text, time.time())

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
else: resolution = (640, 480)

# FIX: Smart ID Generation (Uses filenames)
camera_ids = []
for i, src in enumerate(sources):
    if os.path.isfile(src):
        camera_ids.append(os.path.basename(src))
    else:
        camera_ids.append(f"cam{i}")

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
            if FEED_ENHANCE:
                frame = enhance_frame(frame)
            fps_map[cid] = cam.fps
            
            if not yolo_input_queue.full():
                yolo_input_queue.put((frame, cid))

            detections = shared_state.get_detections(cid)

            for det in detections:
                x1, y1, x2, y2 = det['rect']
                bid = det['bid']
                txt, col = shared_state.get_display_text(cid, bid)
                if txt is None:
                    if det.get('is_target', True):
                        txt = "Scanning..."
                        col = (255, 165, 0)
                    else:
                        txt = f"{det.get('raw_label', det['label'])}: {det['conf']:.2f}"
                        col = (0, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            frames[cid] = frame

        final_img = grid.compose(frames, fps_map)
        cv2.imshow('Truck Logistics Scanner (Async)', final_img)
        if record and recorder: recorder.write(final_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    for c in cameras.values(): c.stop()
    if record and recorder: recorder.release()
    cv2.destroyAllWindows()