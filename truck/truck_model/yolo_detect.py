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
OCR_THRESH = 0.80
LOG_COOLDOWN = 60.0

PLATE_REGEX = r'^[A-Z0-9\-]{4,15}$' 
CONTAINER_REGEX = r'^[A-Z]{4}[0-9]{6,7}$'

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
args = parser.parse_args()

model_path = args.model
global_min_thresh = args.thresh
user_res = args.resolution
record = args.record
grid_cols = args.grid_cols
num_ocr_workers = max(1, args.ocr_workers)
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

    def update_detections(self, cam_id, detections):
        with self._lock:
            self.latest_detections[cam_id] = detections

    def get_detections(self, cam_id):
        with self._lock:
            return self.latest_detections.get(cam_id, [])

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
        
        # Priority 2: Check Regex
        if re.match(CONTAINER_REGEX, candidate) or re.match(PLATE_REGEX, candidate):
            return candidate
            
    return text

# FIX: Better Preprocessing
def preprocess_for_ocr(img):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize: Upscale if too small
    height = gray.shape[0]
    if height < 60: 
        scale = 60 / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 3. Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 5. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # 6. Padding
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return gray

def clean_text(text): return re.sub(r'[^A-Z0-9\-]', '', text.upper())

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

def yolo_worker():
    print("YOLO Worker Started")
    while True:
        try:
            frame, cam_id = yolo_input_queue.get(timeout=1)
            
            results = model(frame, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls_id = int(box.cls)
                
                raw_label = labels[cls_id] if cls_id < len(labels) else "unknown"
                label_key = raw_label.lower().replace(' ', '_')
                req_conf = CLASS_THRESHOLDS.get(label_key, global_min_thresh)
                
                if conf > req_conf:
                    # FIX: Targets must be lowercase now
                    targets = ['usdot', 'trailernum', 'containerplate', 'licenseplate', 'containernum']
                    if any(t in label_key for t in targets):
                        detections.append({
                            'rect': (x1, y1, x2, y2),
                            'bid': f"{x1}_{y1}",
                            'label': label_key,
                            'conf': conf
                        })
            
            shared_state.update_detections(cam_id, detections)
            
            # OCR Handoff
            for det in detections:
                bid = det['bid']
                txt, _ = shared_state.get_display_text(cam_id, bid)
                if txt is None:
                    if not ocr_queue.full():
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = det['rect']
                        px = int((x2-x1) * (0.15 if 'number' in det['label'] else 0.08))
                        py = int((y2-y1) * 0.08)
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

def ocr_worker(wid):
    while True:
        try:
            update_authorized_list()
            data = ocr_queue.get(timeout=1)
            img_crop, conf_yolo, box_id, camera_id, class_type = data

            processed = preprocess_for_ocr(img_crop)
            
            # Get raw details (box coordinates + text) to find vertical stacks
            results = reader.readtext(
                processed, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                decoder='greedy',
                paragraph=False, # Keep false to get individual boxes
                mag_ratio=1.5
            )

            # --- NEW: VERTICAL STITCHING LOGIC ---
            # 1. Sort results by Y-coordinate (Top to Bottom)
            results.sort(key=lambda x: x[0][0][1]) 

            stitched_text = ""
            current_conf = 0
            count = 0
            
            # 2. Naive Stitching: Just join everything found in this crop
            # (Since YOLO already cropped the area tight, this is usually safe)
            for _, text, conf in results:
                clean_char = re.sub(r'[^A-Z0-9]', '', text.upper())
                if clean_char:
                    stitched_text += clean_char
                    current_conf += conf
                    count += 1
            
            # Average confidence of the stitched parts
            if count > 0:
                final_conf = current_conf / count
            else:
                final_conf = 0

            # 3. Now run your existing validation on the STITCHED string
            candidates = [stitched_text] 
            
            # Also keep original non-stitched results in case it was actually horizontal
            for _, text, conf in results:
                candidates.append(re.sub(r'[^A-Z0-9]', '', text.upper()))

            found_match = False
            for raw in candidates:
                if found_match: break
                
                # ... (Insert your existing Validation & Correction Logic here) ...
                
                # TEST: Vertical Container Regex (Standard format still applies once stitched)
                is_valid = False
                if 'container' in class_type:
                    if re.match(CONTAINER_REGEX, raw): is_valid = True
                elif 'plate' in class_type:
                    if re.match(PLATE_REGEX, raw): is_valid = True
                
                if is_valid and final_conf >= OCR_THRESH:
                    # ... (Your existing Logging/Voting Code) ...
                    found_match = True
            
            ocr_queue.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"OCR Error: {e}")

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
        cv2.imshow('Truck Logistics Scanner (Async)', final_img)
        if record and recorder: recorder.write(final_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    for c in cameras.values(): c.stop()
    if record and recorder: recorder.release()
    cv2.destroyAllWindows()